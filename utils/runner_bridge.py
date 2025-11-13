import asyncio
import queue
import sys
import threading
from typing import Any, Coroutine, Generator, Optional, AsyncGenerator, TYPE_CHECKING

from google.adk.runners import Runner
from google.genai import types


if TYPE_CHECKING:
    from google.adk.runners import RunConfig


class RunnerBridge:
    """Sync helper that lets synchronous code call an async Runner."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop_worker,
            name="adk-runner-loop",
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


class AutoSessionRunner(Runner):
    """Runner that creates sessions on first use when needed."""

    async def _ensure_session(
        self,
        *,
        user_id: str,
        session_id: str,
        allow_create: bool,
    ) -> None:
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            if not allow_create:
                message = self._format_session_not_found_message(session_id)
                raise ValueError(message)
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: Optional[str] = None,
        new_message: Optional[types.Content] = None,
        state_delta: Optional[dict[str, Any]] = None,
        run_config: Optional["RunConfig"] = None,
    ) -> AsyncGenerator[Any, None]:
        allow_create = new_message is not None or invocation_id is None
        await self._ensure_session(
            user_id=user_id,
            session_id=session_id,
            allow_create=allow_create,
        )
        async for event in super().run_async(
            user_id=user_id,
            session_id=session_id,
            invocation_id=invocation_id,
            new_message=new_message,
            state_delta=state_delta,
            run_config=run_config,
        ):
            yield event
