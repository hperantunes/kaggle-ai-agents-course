import base64
import binascii
import uuid
from typing import Optional

from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


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


# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

# Create image agent with MCP integration
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

root_agent = image_agent
