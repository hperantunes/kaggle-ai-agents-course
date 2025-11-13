import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, List

from google.adk import runners as adk_runners
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import load_local_env
from utils.runner_bridge import RunnerBridge

load_local_env(__file__)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Handle potential incorrect datatype returned by the agent ----
def _normalize_papers(raw_papers: Any) -> List[str]:
    """Normalize tool input into a clean list of paper titles."""

    def _split_lines(text: str) -> List[str]:
        cleaned = []
        for line in text.splitlines():
            entry = line.strip(" -\t•")
            if entry:
                cleaned.append(entry)
        return cleaned

    normalized: List[str] = []

    if isinstance(raw_papers, str):
        candidate = raw_papers.strip()
        if not candidate:
            return []
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            normalized.extend(_split_lines(candidate))
        else:
            if isinstance(parsed, list):
                for item in parsed:
                    entry = str(item).strip()
                    if entry:
                        normalized.append(entry)
            else:
                normalized.extend(_split_lines(candidate))
        return normalized

    if not isinstance(raw_papers, Iterable):
        return normalized

    for item in raw_papers:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text:
                normalized.extend(_split_lines(text))
            else:
                normalized.append(text)
        else:
            if isinstance(parsed, list):
                for entry in parsed:
                    clean_entry = str(entry).strip()
                    if clean_entry:
                        normalized.append(clean_entry)
            else:
                normalized.extend(_split_lines(text))

    return normalized


def count_papers(papers: List[str]):
    """Count how many distinct paper titles were discovered.

    The agent occasionally sends a single string (e.g. bullet list or JSON) or
    a sequence of strings. We coerce the value into a list before counting to
    ensure the total reflects the actual number of papers.
    """

    normalized_papers = _normalize_papers(papers)
    return len(normalized_papers)


# Google Search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of research papers you found and not the summary.""",
    tools=[google_search]
)


# Root agent
research_agent_with_plugin = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using ONLY the 'google_search_agent'. 
    2) Whenever new papers are found, cleanly structure the results as a JSON array of paper titles and immediately call 'count_papers'.
    3) Never skip step 2. Your final response MUST include both the JSON list and the total number of papers returned by 'count_papers'.
    """,
    tools=[AgentTool(agent=google_search_agent), count_papers]
)

runner = InMemoryRunner(
    agent=research_agent_with_plugin,
    plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)


async def _ensure_session(*, user_id: str, session_id: str) -> None:
    session_service = runner.session_service
    session = await session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:
        await session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )


def _dispatch_user_message(
    *,
    bridge: RunnerBridge,
    user_id: str,
    session_id: str,
    message: str,
    verbose: bool,
) -> None:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    try:
        bridge.run(
            _ensure_session(
                user_id=user_id,
                session_id=session_id,
            )
        )
        for event in bridge.stream_events(
            user_id=user_id,
            session_id=session_id,
            content=content,
        ):
            adk_runners.print_event(event, verbose=verbose)
    except Exception as exc:  # noqa: BLE001 - surface runtime issues for CLI users
        print(f"Error while running agent: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the logging-in-production agent from the console.",
    )
    parser.add_argument(
        "-m",
        "--message",
        help="Optional one-shot prompt to send to the agent. Leave blank for interactive mode.",
    )
    parser.add_argument(
        "--session",
        help="Optional session identifier so you can resume a prior run.",
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
    bridge = RunnerBridge(runner)

    if args.message:
        try:
            _dispatch_user_message(
                bridge=bridge,
                user_id=args.user,
                session_id=session_id,
                message=args.message,
                verbose=args.verbose,
            )
        finally:
            bridge.close()
        return

    print("Starting interactive session with logging agent. Type 'exit' or 'quit' to leave.")
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

            _dispatch_user_message(
                bridge=bridge,
                user_id=args.user,
                session_id=session_id,
                message=user_input,
                verbose=args.verbose,
            )
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

