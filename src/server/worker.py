"""Background worker for processing the task queue."""

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from collections import deque
from pathlib import Path
from config import settings

from .models import TaskProgress
from .queue_manager import QueueManager


class Worker:
    """Background worker that processes tasks from the queue."""

    def __init__(self, queue_manager: QueueManager, generated_dir: Path, task_timeout: float | None = None):
        """Initialize the worker.

        Args:
            queue_manager: Queue manager instance
            generated_dir: Path to generated/ directory
            task_timeout: Optional overall deadline; no batch time limit by default.
        """
        self.queue_manager = queue_manager
        self.generated_dir = generated_dir
        self.task_timeout = task_timeout
        self._running = False
        self._current_process: asyncio.subprocess.Process | None = None
        self._worker_script = Path(__file__).parent / "worker_subprocess.py"

    async def run(self):
        """Main worker loop - processes tasks from the queue."""
        self._running = True

        while self._running:
            task = self.queue_manager.get_next_task()

            if task is None:
                # No task available, wait a bit
                await asyncio.sleep(0.5)
                continue

            # Execute the task
            await self._execute_task(task)

    def stop(self):
        """Stop the worker."""
        self._running = False
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception as e:
                logging.debug(f"Failed to terminate process during stop: {e}")

    async def _terminate_process(self, process, graceful_timeout: float = 2.0) -> None:
        """Reap an owned child with bounded graceful and force-kill waits."""
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            logging.warning("Worker process cleanup did not finish normally")

    async def kill_current(self) -> bool:
        """Kill the currently running task.

        Returns:
            True if a task was killed
        """
        process = self._current_process
        if process is None:
            return False

        task_id = None

        # Get current task info
        state = self.queue_manager.get_state()
        if state.current_task:
            task_id = state.current_task.id

        # Record user intent before EOF/exit can race the normal failure path.
        if task_id:
            self.queue_manager.cancel_task(task_id)

        try:
            await self._terminate_process(process)
        except Exception as e:
            # Process might already be dead
            logging.debug(f"Process cleanup during kill_current: {e}")

        if self._current_process is process:
            self._current_process = None
        return True

    async def _read_stderr(self, process, task_id: str, stderr_lines=None) -> list[str]:
        """Read stderr from process and stream to queue manager.

        Args:
            process: The subprocess to read stderr from
            task_id: Task ID for logging context

        Returns:
            Bounded tail of stderr (for final error reporting).
        """
        if stderr_lines is None:
            stderr_lines = deque(maxlen=20)

        def emit(line: bytes) -> None:
            text = line.decode(errors="replace").strip()[-1024:]
            if text:
                stderr_lines.append(text)
                # Emit as SSE event for real-time display
                self.queue_manager.emit_log(task_id, text)
                # Also log to server logs
                logging.info(f"[worker:{task_id[:8]}] {text}")

        # tqdm uses carriage returns; native errors may omit newlines entirely.
        # Bound both the unfinished line and the final diagnostic history.
        pending = b""
        try:
            while chunk := await process.stderr.read(4096):
                lines = (pending + chunk).replace(b"\r", b"\n").split(b"\n")
                pending = lines.pop()[-4096:]
                for line in lines:
                    emit(line)
        finally:
            emit(pending)
        return list(stderr_lines)

    async def _execute_task(self, task):
        """Own terminal status for optional deadlines and shutdown cancellation.

        Args:
            task: Task to execute
        """
        try:
            await asyncio.wait_for(
                self._run_task_inner(task),
                timeout=self.task_timeout,
            )
        except asyncio.TimeoutError:
            self.queue_manager.fail_task(
                task.id, f"Task exceeded timeout of {self.task_timeout:g}s"
            )
        except asyncio.CancelledError:
            self.queue_manager.cancel_task(task.id)
            raise

    async def _run_task_inner(self, task):
        """Inner execution loop for a single task. Called inside wait_for."""
        # Create temp file for task params
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
        ) as f:
            json.dump({
                "type": task.type.value,
                "params": task.params,
            }, f)
            task_file = f.name

        process = None
        stderr_task = None
        stderr_lines = deque(maxlen=20)
        try:
            # Spawn subprocess using the same Python interpreter
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self._worker_script),
                task_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.generated_dir.parent / "src"),
            )
            self._current_process = process

            # Update task with PID
            self.queue_manager.update_task_pid(task.id, process.pid)

            # Start stderr reader concurrently
            stderr_task = asyncio.create_task(self._read_stderr(process, task.id, stderr_lines))

            # Read output line by line
            result_data = None
            error_message = None

            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=settings.server.worker_timeout  # Liveness check, not a task deadline.
                    )
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    continue  # Process alive, keep waiting

                if not line:
                    break

                try:
                    data = json.loads(line.decode().strip())
                    msg_type = data.get("type")

                    if msg_type == "progress":
                        self.queue_manager.update_progress(
                            task.id,
                            TaskProgress(
                                stage=data.get("stage", ""),
                                current=data.get("current", 0),
                                total=data.get("total", 0),
                                message=data.get("message", ""),
                            ),
                        )
                    elif msg_type == "result":
                        if data.get("success"):
                            result_data = data.get("data")
                        else:
                            error_message = data.get("error", "Unknown error")
                    elif msg_type == "image_ready":
                        self.queue_manager.notify_image_ready(
                            data.get("run_id", ""),
                            data.get("path", ""),
                        )
                except json.JSONDecodeError:
                    logging.warning(f"Worker output not JSON: {line.decode().strip()[:100]}")

            # Wait for process to complete
            await process.wait()

            # Collect stderr (already streamed to log events, keep for error reporting)
            try:
                await asyncio.wait_for(stderr_task, timeout=2.0)
            except asyncio.TimeoutError:
                logging.warning("Worker stderr remained open after process exit")

            # Check exit code
            if process.returncode != 0:
                reported_error = error_message
                error_message = f"Process exited with code {process.returncode}"
                if process.returncode < 0:
                    try:
                        error_message += f" ({signal.Signals(-process.returncode).name})"
                    except ValueError:
                        pass
                if reported_error:
                    error_message += f"; {reported_error}"
                if stderr_lines:
                    error_message += "; stderr tail: " + "; ".join(stderr_lines)

            # Update task status
            if error_message:
                self.queue_manager.fail_task(task.id, error_message)
            else:
                self.queue_manager.complete_task(task.id, result_data)

        except asyncio.CancelledError:
            await self._terminate_process(process, graceful_timeout=5.0)
            raise

        except Exception as e:
            await self._terminate_process(process)
            self.queue_manager.fail_task(task.id, str(e))

        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            # Clean up temp file
            try:
                os.unlink(task_file)
            except Exception as e:
                logging.debug(f"Failed to clean up task file {task_file}: {e}")

            if self._current_process is process:
                self._current_process = None
