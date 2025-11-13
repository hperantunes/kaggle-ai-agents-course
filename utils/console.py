import argparse
import sys
import uuid

from google.adk import runners as adk_runners
from google.adk.runners import Runner
from google.genai import types

from .runner_bridge import RunnerBridge


async def _ensure_session(*, runner: Runner, user_id: str, session_id: str) -> None:
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
    runner: Runner,
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
                runner=runner,
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


def run_cli(
    *,
    runner: Runner,
    description: str,
    intro_message: str,
    prompt_label: str = "you> ",
    default_user: str = "cli_user",
) -> None:
    """Run a console interface for a runner."""

    parser = argparse.ArgumentParser(description=description)
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
        default=default_user,
        help=f"Identifier attached to events emitted by this CLI user (default: {default_user}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show raw tool call details emitted by the agent.",
    )
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    bridge = RunnerBridge(runner)

    try:
        if args.message:
            _dispatch_user_message(
                runner=runner,
                bridge=bridge,
                user_id=args.user,
                session_id=session_id,
                message=args.message,
                verbose=args.verbose,
            )
            return

        print(intro_message)
        print(f"Session ID: {session_id}")

        while True:
            try:
                user_input = input(prompt_label).strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            _dispatch_user_message(
                runner=runner,
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
