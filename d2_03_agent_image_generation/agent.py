import base64
import binascii
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.tools.tool_context import ToolContext


retry_config = types.HttpRetryOptions(
	attempts=5,
	exp_base=7,
	initial_delay=1,
	http_status_codes=[429, 500, 503, 504],
)


async def persist_image_artifact(
	image_base64: str,
	*,
	mime_type: str = "image/png",
	filename: Optional[str] = None,
	tool_context: ToolContext,
) -> dict:
	"""Stores a base64 image as an artifact so ADK Web can display it."""

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


_BULK_COUNT_KEYS: Sequence[str] = (
	"image_count",
	"num_images",
	"count",
	"n",
	"quantity",
	"batch_size",
	"images",
	"samples",
)


def _coerce_count(value: object) -> Optional[int]:
	if isinstance(value, bool):
		return None
	if isinstance(value, (int, float)):
		return int(value)
	if isinstance(value, str):
		try:
			return int(float(value.strip()))
		except ValueError:
			return None
	if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
		return len(value)  # type: ignore[arg-type]
	return None


def requires_bulk_confirmation(**kwargs) -> bool:
	"""Returns True when a tool call appears to request multiple images."""

	for key in _BULK_COUNT_KEYS:
		if key in kwargs:
			count = _coerce_count(kwargs[key])
			if count is not None:
				return count > 1

	prompts = kwargs.get("prompts")
	if isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
		return len(prompts) > 1

	prompt = kwargs.get("prompt")
	if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
		return len(prompt) > 1

	return False


@dataclass
class _ImageServerConfig:
	id: str
	label: str
	summary: str
	command: str
	args: Sequence[str]
	required_env: Sequence[str] = ()
	optional_env: Sequence[str] = ()
	tool_filter: Optional[Sequence[str]] = None
	tool_name_prefix: Optional[str] = None
	timeout: int = 60


def _build_stdio_connection(
	*,
	command: str,
	args: Sequence[str],
	required_env: Sequence[str],
	optional_env: Sequence[str],
	timeout: int,
) -> tuple[Optional[StdioConnectionParams], Optional[str]]:
	env: dict[str, str] = {}
	missing = [key for key in required_env if not os.environ.get(key)]
	if missing:
		return None, f"set {'/'.join(missing)} to enable"

	for key in required_env:
		env[key] = os.environ[key]

	for key in optional_env:
		value = os.environ.get(key)
		if value:
			env[key] = value

	server_params = StdioServerParameters(
		command=command,
		args=list(args),
		env=env or None,
	)

	connection = StdioConnectionParams(
		server_params=server_params,
		timeout=timeout,
	)
	return connection, None


IMAGE_SERVER_CONFIGS: Sequence[_ImageServerConfig] = (
	_ImageServerConfig(
		id="everything",
		label="Everything demo server",
		summary="Provides a miniature sample image via getTinyImage for quick smoke tests.",
		command="npx",
		args=("-y", "@modelcontextprotocol/server-everything"),
		tool_filter=("getTinyImage",),
		tool_name_prefix="everything_",
		timeout=30,
	),
	_ImageServerConfig(
		id="everart",
		label="EverArt (Flux/SD3.5/Recraft)",
		summary="High-quality art generation across multiple models. Requires EVERART_API_KEY.",
		command="npx",
		args=("-y", "@modelcontextprotocol/server-everart"),
		required_env=("EVERART_API_KEY",),
		tool_name_prefix="everart_",
		timeout=60,
	),
	_ImageServerConfig(
		id="replicate",
		label="Replicate model hub",
		summary="Access to community image models (e.g., Flux, SDXL) via Replicate API token.",
		command="npx",
		args=("-y", "mcp-replicate"),
		required_env=("REPLICATE_API_TOKEN",),
		tool_name_prefix="replicate_",
		timeout=60,
	),
)


def _initialize_toolsets() -> tuple[list[McpToolset], list[str], list[str]]:
	active_toolsets: list[McpToolset] = []
	active_descriptions: list[str] = []
	disabled_descriptions: list[str] = []

	for config in IMAGE_SERVER_CONFIGS:
		connection, disable_reason = _build_stdio_connection(
			command=config.command,
			args=config.args,
			required_env=config.required_env,
			optional_env=config.optional_env,
			timeout=config.timeout,
		)

		if not connection:
			disabled_descriptions.append(
				f"{config.label} (`{config.id}`) – {config.summary} [{disable_reason}]"
			)
			continue

		toolset = McpToolset(
			connection_params=connection,
			tool_filter=list(config.tool_filter) if config.tool_filter else None,
			tool_name_prefix=config.tool_name_prefix,
			require_confirmation=requires_bulk_confirmation,
		)
		active_toolsets.append(toolset)
		active_descriptions.append(
			f"{config.label} (`{config.id}`) – {config.summary}"
		)

	return active_toolsets, active_descriptions, disabled_descriptions


IMAGE_TOOLSETS, ACTIVE_SERVER_DESCRIPTIONS, DISABLED_SERVER_DESCRIPTIONS = (
	_initialize_toolsets()
)

if not IMAGE_TOOLSETS:
	raise RuntimeError(
		"No MCP image servers could be initialized. Ensure Node.js/npx are installed "
		"and required API keys are configured."
	)


ACTIVE_SERVER_GUIDE = (
	"\n".join(f"- {description}" for description in ACTIVE_SERVER_DESCRIPTIONS)
	if ACTIVE_SERVER_DESCRIPTIONS
	else "- None currently active."
)

DISABLED_SERVER_GUIDE = (
	"\n".join(f"- {description}" for description in DISABLED_SERVER_DESCRIPTIONS)
	if DISABLED_SERVER_DESCRIPTIONS
	else "- None."
)


instruction = (
	"You are an energetic AI art director. Help users pick an image generation "
	"provider, gather a detailed prompt, and run the appropriate MCP tool.\n"
	"\n"
	"Active MCP image servers:\n"
	f"{ACTIVE_SERVER_GUIDE}\n"
	"\n"
	"Disabled or awaiting configuration:\n"
	f"{DISABLED_SERVER_GUIDE}\n"
	"\n"
	"Guidelines:\n"
	"- Default to the first active server unless the user specifies otherwise.\n"
	"- Clarify desired style, aspect ratio, and number of outputs before calling a tool.\n"
	"- When the user requests multiple images, set the tool arguments (e.g., image_count) "
	"accordingly. Calls requesting more than one image trigger a confirmation pause; wait "
	"for approval before proceeding.\n"
	"- After receiving base64 image data from any tool, immediately call persist_image_artifact "
	"so ADK Web can render it, then reference the artifact URI with Markdown syntax.\n"
	"- If a tool returns URLs instead of base64 data, surface the links and offer to download "
	"them if needed.\n"
	"- If no configured server can satisfy the request, explain what setup is missing and "
	"offer guidance rather than fabricating results."
)


root_agent = LlmAgent(
	model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
	name="mcp_image_agent",
	instruction=instruction,
	tools=[*IMAGE_TOOLSETS, persist_image_artifact],
)
