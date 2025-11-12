import argparse
import asyncio
import base64
import binascii
import os
import queue
import sys
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Coroutine

from google.adk import runners as adk_runners
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

# Usage: activate the project virtualenv, then run `python agent.py` for an
# interactive art session or `python agent.py --message "Generate a neon fox"`
# for a one-shot prompt.


class RunnerBridge:
	"""Sync-friendly wrapper around the asynchronous ADK runner."""

	def __init__(self, runner: Runner) -> None:
		self._runner = runner
		self._loop = asyncio.new_event_loop()
		self._thread = threading.Thread(
			target=self._loop_worker,
			name="image-agent-loop",
			daemon=True,
		)
		self._thread.start()
		self._closed = False

	def _loop_worker(self) -> None:
		asyncio.set_event_loop(self._loop)
		self._loop.run_forever()

	def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
		if self._closed:
			raise RuntimeError("RunnerBridge is already closed.")
		future = asyncio.run_coroutine_threadsafe(coro, self._loop)
		try:
			return future.result()
		except KeyboardInterrupt:
			future.cancel()
			raise

	def stream_events(
		self,
		*,
		user_id: str,
		session_id: str,
		content: types.Content,
	) -> Generator[Any, None, None]:
		if self._closed:
			raise RuntimeError("RunnerBridge is already closed.")

		event_queue: "queue.Queue[Any]" = queue.Queue()

		async def _invoke() -> None:
			try:
				async for event in self._runner.run_async(
					user_id=user_id,
					session_id=session_id,
					new_message=content,
				):
					event_queue.put(event)
			except BaseException as exc:  # noqa: BLE001 - surface async failures
				event_queue.put(exc)
			finally:
				event_queue.put(None)

		asyncio.run_coroutine_threadsafe(_invoke(), self._loop)

		pending_exc: Optional[BaseException] = None
		while True:
			item = event_queue.get()
			if item is None:
				break
			if isinstance(item, BaseException):
				pending_exc = item
				continue
			yield item

		if pending_exc is not None:
			raise pending_exc

	def close(self) -> None:
		if self._closed:
			return

		try:
			future = asyncio.run_coroutine_threadsafe(self._runner.close(), self._loop)
			future.result()
		except RuntimeError as exc:
			print(f"Warning: failed to close runner cleanly: {exc}", file=sys.stderr)
		finally:
			self._loop.call_soon_threadsafe(self._loop.stop)
			self._thread.join()
			self._loop.close()
			self._closed = True



def _looks_like_image(image_bytes: bytes, mime_type: str) -> bool:
	header = image_bytes[:12]
	if not header:
		return False
	mime_lower = mime_type.lower()
	if mime_lower == "image/png":
		return header.startswith(b"\x89PNG\r\n\x1a\n")
	if mime_lower in {"image/jpeg", "image/jpg"}:
		return header.startswith(b"\xff\xd8\xff")
	if mime_lower == "image/webp":
		return header.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
	if mime_lower == "image/gif":
		return header.startswith((b"GIF87a", b"GIF89a"))
	return True


def _safe_b64decode(data: str) -> bytes:
	"""Decode base64 content while tolerating missing padding."""
	padding = (-len(data)) % 4
	if padding:
		data += "=" * padding
	return base64.b64decode(data)


def _summarize_inline_image(part: Dict[str, Any]) -> Optional[str]:
	"""Produce a short message describing an inline MCP image payload."""
	data = part.get("data")
	if not isinstance(data, str) or not data:
		return None
	mime_type = str(part.get("mimeType") or part.get("mime_type") or "image/png")
	try:
		image_bytes = _safe_b64decode(data)
	except (binascii.Error, ValueError) as exc:
		return f"Inline image payload failed to decode: {exc}"

	if not _looks_like_image(image_bytes, mime_type):
		status = "header unrecognized"
	else:
		status = "looks like a valid image"

	sample = data[:60] + ("..." if len(data) > 60 else "")
	return (
		f"Inline image summary: {len(image_bytes)} bytes ({mime_type}, {status}). "
		f"Sample base64: {sample}"
	)


def _coerce_to_dict(value: Any) -> Dict[str, Any]:
	if isinstance(value, dict):
		return value
	try:
		return dict(value)  # type: ignore[arg-type]
	except Exception:  # noqa: BLE001 - best effort conversion
		return {}


def _load_local_env() -> None:
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
		print(f"Warning: failed to read {env_path.name}: {exc}", file=sys.stderr)


_load_local_env()


retry_config = types.HttpRetryOptions(
	attempts=5,
	exp_base=7,
	initial_delay=1,
	http_status_codes=[429, 500, 503, 504],
)


async def show_image_summary(
	image_base64: str,
	*,
	mime_type: str = "image/png",
	filename: Optional[str] = None,
	tool_context: ToolContext,
) -> dict:
	"""Return a short summary of the base64 payload instead of persisting it."""

	if not image_base64:
		return {
			"status": "error",
			"message": "image_base64 is required to inspect an image payload.",
		}

	try:
		image_bytes = _safe_b64decode(image_base64)
	except (binascii.Error, ValueError) as exc:
		return {
			"status": "error",
			"message": f"Failed to decode base64 image payload: {exc}",
		}

	if not _looks_like_image(image_bytes, mime_type):
		return {
			"status": "warning",
			"message": "Payload decoded, but header does not look like a standard image.",
			"bytes": len(image_bytes),
		}

	return {
		"status": "success",
		"message": "Image payload received.",
		"mime_type": mime_type,
		"bytes": len(image_bytes),
		"sample": image_base64[:60] + ("..." if len(image_base64) > 60 else ""),
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
	"- After receiving base64 image data from any tool, call show_image_summary to confirm "
	"what came back, then describe the result in Markdown.\n"
	"- If a tool returns URLs instead of base64 data, surface the links and offer to download "
	"them if needed.\n"
	"- If no configured server can satisfy the request, explain what setup is missing and "
	"offer guidance rather than fabricating results."
)


root_agent = LlmAgent(
	model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
	name="mcp_image_agent",
	instruction=instruction,
	tools=[*IMAGE_TOOLSETS, show_image_summary],
)

image_app = App(
	name="image_generation_app",
	root_agent=root_agent,
	resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()

image_runner = Runner(
	app=image_app,
	session_service=session_service,
)


async def _ensure_session(user_id: str, session_id: str) -> None:
	session = await session_service.get_session(
		app_name=image_runner.app_name,
		user_id=user_id,
		session_id=session_id,
	)
	if session is None:
		await session_service.create_session(
			app_name=image_runner.app_name,
			user_id=user_id,
			session_id=session_id,
		)


def _maybe_render_tool_result(event, verbose: bool) -> None:
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
			if "content" in response and isinstance(response["content"], list):
				text_parts = sum(
					1
					for entry in response["content"]
					if isinstance(entry, dict) and entry.get("type") == "text"
				)
				image_parts = sum(
					1
					for entry in response["content"]
					if isinstance(entry, dict) and entry.get("type") == "image"
				)
				print(
					f"{event.author} > Tool returned {text_parts} text parts and {image_parts} image parts."
				)
				for entry in response["content"]:
					if isinstance(entry, dict) and entry.get("type") == "image":
						summary = _summarize_inline_image(entry)
						if summary:
							print(f"{event.author} > {summary}")
				continue
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
			byte_count = response.get("bytes")
			mime_type = response.get("mime_type")
			sample = response.get("sample")
			if isinstance(byte_count, int):
				info = f"{byte_count} bytes"
				if mime_type:
					info += f" ({mime_type})"
				print(f"{event.author} > Image summary: {info}")
			if sample:
				print(f"{event.author} > Sample base64: {sample}")
		elif response is not None:
			print(f"{event.author} > Tool result: {response}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run the image generation agent from the console.",
	)
	parser.add_argument(
		"-m",
		"--message",
		help="Optional one-shot prompt to send to the agent. Leave blank for interactive mode.",
	)
	parser.add_argument(
		"--session",
		help=(
			"Optional session identifier. Reuse this when resuming an approval flow; "
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
		help="Show raw tool call details emitted by the agent.",
	)
	args = parser.parse_args()

	session_id = args.session or str(uuid.uuid4())
	bridge = RunnerBridge(image_runner)
	pending_confirmations: Dict[str, Dict[str, Any]] = {}

	def _collect_confirmation_requests(event: Any) -> None:
		content = getattr(event, "content", None)
		parts = getattr(content, "parts", None)
		if not parts:
			return
		for part in parts:
			function_call = getattr(part, "function_call", None)
			if function_call and function_call.name == "adk_request_confirmation":
				args_dict = _coerce_to_dict(getattr(function_call, "args", {}) or {})
				original_call = _coerce_to_dict(args_dict.get("originalFunctionCall"))
				tool_conf = _coerce_to_dict(args_dict.get("toolConfirmation"))

				payload = tool_conf.get("payload") if isinstance(tool_conf.get("payload"), dict) else None
				pending_confirmations[function_call.id] = {
					"hint": str(tool_conf.get("hint", "")).strip(),
					"original_tool": original_call.get("name"),
					"payload": payload,
					"original_args": original_call.get("args") if isinstance(original_call.get("args"), dict) else None,
				}

			function_response = getattr(part, "function_response", None)
			if function_response and function_response.name == "adk_request_confirmation":
				pending_confirmations.pop(function_response.id, None)

	def _send_content(content: types.Content) -> None:
		try:
			bridge.run(_ensure_session(args.user, session_id))
			for event in bridge.stream_events(
				user_id=args.user,
				session_id=session_id,
				content=content,
			):
				adk_runners.print_event(event, verbose=args.verbose)
				_maybe_render_tool_result(event, args.verbose)
				_collect_confirmation_requests(event)
		except Exception as exc:  # noqa: BLE001 - surface runtime issues for CLI users
			print(f"Error while running agent: {exc}", file=sys.stderr)

	def _dispatch_text(user_message: str) -> None:
		content = types.Content(role="user", parts=[types.Part(text=user_message)])
		_send_content(content)

	def _send_confirmation_response(call_id: str, *, confirmed: bool, payload: Optional[Dict[str, Any]]) -> None:
		response_payload: Dict[str, Any] = {"confirmed": confirmed}
		if payload is not None:
			response_payload["payload"] = payload
		pending_confirmations.pop(call_id, None)
		content = types.Content(
			role="user",
			parts=[
				types.Part(
					function_response=types.FunctionResponse(
						name="adk_request_confirmation",
						id=call_id,
						response=response_payload,
					)
				)
			],
		)
		_send_content(content)

	def _describe_request(request: Dict[str, Any]) -> str:
		payload = request.get("payload") or {}
		description_parts: list[str] = []
		if isinstance(payload, dict):
			num = payload.get("image_count") or payload.get("num_images") or payload.get("count")
			if isinstance(num, int):
				description_parts.append(f"{num} images")
			prompt = payload.get("prompt") or payload.get("prompts")
			if prompt:
				description_parts.append(str(prompt))
		if not description_parts and request.get("original_args"):
			description_parts.append(str(request["original_args"]))
		return " | ".join(description_parts)

	def _handle_pending_confirmations() -> None:
		while pending_confirmations:
			call_id, request = next(iter(pending_confirmations.items()))
			details = _describe_request(request)
			print()
			print("Confirmation required:")
			tool_name = request.get("original_tool")
			if tool_name:
				print(f"  Tool: {tool_name}")
			if details:
				print(f"  Details: {details}")
			hint = request.get("hint")
			if hint:
				print(f"  Hint: {hint}")
			payload = request.get("payload") if isinstance(request.get("payload"), dict) else None
			while True:
				try:
					choice = input("Approve this tool call? [y/n]: ").strip().lower()
				except EOFError:
					choice = ""
				if choice in {"y", "yes"}:
					_send_confirmation_response(call_id, confirmed=True, payload=payload)
					break
				if choice in {"n", "no"}:
					_send_confirmation_response(call_id, confirmed=False, payload=payload)
					break
				print("Please reply with 'y' to approve or 'n' to reject.")

	if args.message:
		try:
			_dispatch_text(args.message)
			_handle_pending_confirmations()
		finally:
			bridge.close()
		return

	print("Starting interactive session with image agent. Type 'exit' or 'quit' to leave.")
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
		bridge.close()


if __name__ == "__main__":
	main()
