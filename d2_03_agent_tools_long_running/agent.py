import asyncio
import argparse
import os
import sys
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk import runners as adk_runners
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool


def _load_local_env() -> None:
    """Load environment variables from a sibling .env file if present."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
    except OSError as exc:
        # Surface a concise warning without interrupting CLI execution.
        print(f"Warning: failed to read {env_path.name}: {exc}", file=sys.stderr)


_load_local_env()

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

LARGE_ORDER_THRESHOLD = 5


class ShippingAgent(LlmAgent):
    """Local subclass keeps module path anchored for runner metadata."""


@dataclass
class PendingConfirmation:
    call_id: str
    hint: str
    payload: Optional[Dict[str, Any]]
    original_tool: Optional[str]
    original_args: Optional[Dict[str, Any]]


def place_shipping_order(
    num_containers: int, destination: str, tool_context: ToolContext
) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 containers (LARGE_ORDER_THRESHOLD).

    Args:
        num_containers: Number of containers to ship
        destination: Shipping destination

    Returns:
        Dictionary with order status
    """

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: Small orders (≤5 containers) auto-approve
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: This is the first time this tool is called. Large orders need human approval - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {num_containers} containers to {destination}. Do you want to approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return {  # This is sent to the Agent
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order approved: {num_containers} containers to {destination}",
        }
    else:
        return {
            "status": "rejected",
            "message": f"Order rejected: {num_containers} containers to {destination}",
        }


# Create shipping agent with pausable tool
shipping_agent = ShippingAgent(
    name="shipping_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a shipping coordinator assistant.
  
  When users request to ship containers:
   1. Use the place_shipping_order tool with the number of containers and destination
   2. If the order status is 'pending', inform the user that approval is required
   3. After receiving the final result, provide a clear summary including:
      - Order status (approved/rejected)
      - Order ID (if available)
      - Number of containers and destination
   4. Keep responses concise but informative
  """,
    tools=[FunctionTool(func=place_shipping_order)],
)

# Wrap the agent in a resumable app - THIS IS THE KEY FOR LONG-RUNNING OPERATIONS!
shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()

# Create runner with the resumable app
shipping_runner = Runner(
    app=shipping_app,  # Pass the app instead of the agent
    session_service=session_service,
)


def _close_runner() -> None:
    """Close the runner, reporting cleanup issues to stderr."""

    try:
        asyncio.run(shipping_runner.close())
    except RuntimeError as exc:
        print(f"Warning: failed to close runner cleanly: {exc}", file=sys.stderr)


async def _ensure_session(user_id: str, session_id: str) -> None:
    """Create the session lazily if it does not already exist."""

    session = await session_service.get_session(
        app_name=shipping_runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:
        await session_service.create_session(
            app_name=shipping_runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )


def _maybe_render_tool_result(event, verbose: bool) -> None:
    """Print tool outcomes when no text response is emitted."""

    if verbose:
        return
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return
    if any(getattr(part, "text", None) for part in parts):
        return

    for part in parts:
        function_response = getattr(part, "function_response", None)
        if not function_response:
            continue
        response = getattr(function_response, "response", None)
        if isinstance(response, dict):
            status = response.get("status")
            message = response.get("message")
            if status and message:
                print(f"{event.author} > {message} (status: {status})")
            elif status:
                print(f"{event.author} > Tool status: {status}")
            elif message:
                print(f"{event.author} > {message}")
            else:
                print(f"{event.author} > Tool result: {response}")
        elif response is not None:
            print(f"{event.author} > Tool result: {response}")


def main() -> None:
    """Run the shipping coordinator agent from the command line."""

    parser = argparse.ArgumentParser(
        description="Run the shipping coordinator agent against a single or interactive prompt.",
    )
    parser.add_argument(
        "-m",
        "--message",
        help="Optional one-shot prompt to send to the agent. Leave blank for interactive mode.",
    )
    parser.add_argument(
        "--session",
        help=(
            "Optional session identifier. Reuse this when resuming long-running flows, "
            "otherwise a new session will be generated."
        ),
    )
    parser.add_argument(
        "--user",
        default="cli_user",
        help="Identifier attached to events emitted by this CLI user (default: cli_user).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show tool call details emitted by the agent.",
    )
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    pending_confirmations: "OrderedDict[str, PendingConfirmation]" = OrderedDict()

    def _collect_confirmation_requests(event: Any) -> None:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None)
        if not parts:
            return
        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call and function_call.name == "adk_request_confirmation":
                raw_args = getattr(function_call, "args", {}) or {}
                if isinstance(raw_args, dict):
                    args_dict = raw_args
                else:
                    try:
                        args_dict = dict(raw_args)
                    except Exception:  # noqa: BLE001 - defensive conversion
                        args_dict = {}
                original_raw = args_dict.get("originalFunctionCall") or {}
                if isinstance(original_raw, dict):
                    original_call = original_raw
                else:
                    try:
                        original_call = dict(original_raw)
                    except Exception:  # noqa: BLE001 - fallback when conversion fails
                        original_call = {}

                tool_conf_raw = args_dict.get("toolConfirmation") or {}
                if isinstance(tool_conf_raw, dict):
                    tool_conf = tool_conf_raw
                else:
                    try:
                        tool_conf = dict(tool_conf_raw)
                    except Exception:  # noqa: BLE001 - fallback when conversion fails
                        tool_conf = {}

                payload_value = tool_conf.get("payload") if isinstance(tool_conf, dict) else None
                payload = payload_value if isinstance(payload_value, dict) else None
                pending_confirmations[function_call.id] = PendingConfirmation(
                    call_id=function_call.id,
                    hint=str(tool_conf.get("hint", "")).strip(),
                    payload=payload,
                    original_tool=original_call.get("name"),
                    original_args=original_call.get("args") if isinstance(original_call.get("args"), dict) else None,
                )

            function_response = getattr(part, "function_response", None)
            if function_response and function_response.name == "adk_request_confirmation":
                pending_confirmations.pop(function_response.id, None)

    def _send_content(content: types.Content) -> None:
        asyncio.run(_ensure_session(args.user, session_id))
        try:
            for event in shipping_runner.run(
                user_id=args.user,
                session_id=session_id,
                new_message=content,
            ):
                adk_runners.print_event(event, verbose=args.verbose)
                _maybe_render_tool_result(event, args.verbose)
                _collect_confirmation_requests(event)
        except Exception as exc:  # noqa: BLE001 - surface runtime issues for CLI users
            print(f"Error while running agent: {exc}", file=sys.stderr)

    def _dispatch_text(user_message: str) -> None:
        content = types.Content(role="user", parts=[types.Part(text=user_message)])
        _send_content(content)

    def _send_confirmation_response(request: PendingConfirmation, *, confirmed: bool) -> None:
        response_payload: Dict[str, Any] = {"confirmed": confirmed}
        if request.payload is not None:
            response_payload["payload"] = request.payload
        pending_confirmations.pop(request.call_id, None)
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_confirmation",
                        id=request.call_id,
                        response=response_payload,
                    )
                )
            ],
        )
        _send_content(content)

    def _describe_request(request: PendingConfirmation) -> str:
        payload = request.payload or {}
        description_parts: list[str] = []
        if "num_containers" in payload:
            description_parts.append(f"{payload['num_containers']} containers")
        if "destination" in payload:
            destination_text = f"to {payload['destination']}"
            if description_parts:
                description_parts[-1] = f"{description_parts[-1]} {destination_text}"
            else:
                description_parts.append(destination_text)
        extras = {k: v for k, v in payload.items() if k not in {"num_containers", "destination"}}
        if extras:
            description_parts.append(str(extras))
        if not description_parts and request.original_args:
            description_parts.append(str(request.original_args))
        return " ".join(description_parts)

    def _handle_pending_confirmations() -> None:
        while pending_confirmations:
            call_id, request = next(iter(pending_confirmations.items()))
            details = _describe_request(request)
            print()
            print("Confirmation required:")
            if request.original_tool:
                print(f"  Tool: {request.original_tool}")
            if details:
                print(f"  Details: {details}")
            if request.hint:
                print(f"  Hint: {request.hint}")
            while True:
                try:
                    choice = input("Approve this tool call? [y/n]: ").strip().lower()
                except EOFError:
                    choice = ""
                if choice in {"y", "yes"}:
                    _send_confirmation_response(request, confirmed=True)
                    break
                if choice in {"n", "no"}:
                    _send_confirmation_response(request, confirmed=False)
                    break
                print("Please reply with 'y' to approve or 'n' to reject.")

    if args.message:
        _dispatch_text(args.message)
        _handle_pending_confirmations()
        _close_runner()
        return

    print(
        "Starting interactive session with shipping agent. Type 'exit' or 'quit' to leave.",
    )
    print(f"Session ID: {session_id}")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            _dispatch_text(user_input)
            _handle_pending_confirmations()
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        _close_runner()


if __name__ == "__main__":
    main()
