import argparse
import asyncio
import base64
import binascii
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def load_env() -> None:
    """Populate os.environ with values from a local .env file when present."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


async def persist_image_artifact(
    image_base64: str,
    *,
    mime_type: str = "image/png",
    filename: Optional[str] = None,
    tool_context: ToolContext,
) -> dict:
    """Saves a base64-encoded image as an artifact so ADK Web can render it."""
    if not image_base64:
        return {
            "status": "error",
            "message": "image_base64 is required to persist an image artifact.",
        }

    try:
        image_bytes = base64.b64decode(image_base64)
    except (binascii.Error, ValueError) as exc:
        return {
            "status": "error",
            "message": f"Failed to decode base64 image payload: {exc}",
        }

    extension = mime_type.split("/")[-1] if "/" in mime_type else "bin"
    artifact_name = filename or f"mcp-image-{uuid.uuid4()}.{extension}"
    artifact_part = types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=image_bytes,
        )
    )

    try:
        version = await tool_context.save_artifact(artifact_name, artifact_part)
    except ValueError as exc:
        return {
            "status": "error",
            "message": (
                "Artifact service is unavailable in this environment; "
                "cannot persist the image."
            ),
            "details": str(exc),
        }

    artifact_uri = f"artifact://{artifact_name}"

    return {
        "status": "success",
        "artifact": artifact_name,
        "artifact_uri": artifact_uri,
        "version": version,
        "message": (
            "Image persisted as an artifact. Reference artifact_uri in your "
            "final reply so ADK Web can render it."
        ),
    }


mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction=(
        "To show images you must call the MCP getTinyImage tool to retrieve the"
        " base64 payload, then immediately call persist_image_artifact with the"
        " returned data so ADK Web can render the attachment. Reference the"
        " returned artifact_uri in Markdown using ![alt text](artifact://...)"
        " when describing the image."
    ),
    tools=[mcp_image_server, persist_image_artifact],
)


async def run_prompt(prompt: str, verbose: bool) -> object:
    runner = InMemoryRunner(agent=image_agent)
    return await runner.run_debug(prompt, verbose=verbose)


def format_response(response: object) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    try:
        return json.dumps(response, indent=2, default=str)
    except (TypeError, ValueError):
        return str(response)


async def async_main(prompt: str, verbose: bool) -> None:
    load_env()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY must be set in the environment or .env file."
        )

    response = await run_prompt(prompt, verbose=verbose)
    print(format_response(response))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tiny image agent with an ad-hoc prompt."
    )
    parser.add_argument("prompt", help="Prompt to send to the agent")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose runner output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args.prompt, verbose=args.verbose))


if __name__ == "__main__":
    main()
