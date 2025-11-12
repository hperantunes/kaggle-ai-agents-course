import asyncio
import argparse
import os
import sys
import uuid
from pathlib import Path

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

    def dispatch(user_message: str) -> None:
        asyncio.run(_ensure_session(args.user, session_id))
        content = types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )

        try:
            for event in shipping_runner.run(
                user_id=args.user,
                session_id=session_id,
                new_message=content,
            ):
                adk_runners.print_event(event, verbose=args.verbose)
                _maybe_render_tool_result(event, args.verbose)
        except Exception as exc:  # noqa: BLE001 - surface runtime issues for CLI users
            print(f"Error while running agent: {exc}", file=sys.stderr)

    if args.message:
        dispatch(args.message)
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

            dispatch(user_input)
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        _close_runner()


if __name__ == "__main__":
    main()
