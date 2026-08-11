"""Background worker for processing the task queue."""

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path

from .models import TaskProgress
from .queue_manager import QueueManager


class Worker:
    """Background worker that processes tasks from the queue."""

    def __init__(self, queue_manager: QueueManager, generated_dir: Path, task_timeout: float = 1800.0):
        """Initialize the worker.

        Args:
            queue_manager: Queue manager instance
            generated_dir: Path to generated/ directory
            task_timeout: Max seconds allowed per task (default 30 minutes)
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

    async def kill_current(self) -> bool:
        """Kill the currently running task.

        Returns:
            True if a task was killed
        """
        if self._current_process is None:
            return False

        pid = self._current_process.pid
        task_id = None

        # Get current task info
        state = self.queue_manager.get_state()
        if state.current_task:
            task_id = state.current_task.id

        try:
            # Try graceful termination first
            self._current_process.terminate()

            # Wait briefly for process to terminate
            try:
                await asyncio.wait_for(self._current_process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if still running
                self._current_process.kill()
                await self._current_process.wait()

        except Exception as e:
            # Process might already be dead
            logging.debug(f"Process cleanup during kill_current: {e}")

        # Mark task as cancelled
        if task_id:
            self.queue_manager.cancel_task(task_id)

        self._current_process = None
        return True

    async def _read_stderr(self, process, task_id: str) -> list[str]:
        """Read stderr from process and stream to queue manager.

        Args:
            process: The subprocess to read stderr from
            task_id: Task ID for logging context

        Returns:
            List of all stderr lines (for final error reporting)
        """
        stderr_lines = []
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            text = line.decode().strip()
            if text:
                stderr_lines.append(text)
                # Emit as SSE event for real-time display
                self.queue_manager.emit_log(task_id, text)
                # Also log to server logs
                logging.info(f"[worker:{task_id[:8]}] {text}")
        return stderr_lines

    async def _execute_task(self, task):
        """Execute a single task with an overall timeout.

        Args:
            task: Task to execute
        """
        try:
            await asyncio.wait_for(
                self._run_task_inner(task),
                timeout=self.task_timeout,
            )
        except asyncio.TimeoutError:
            if self._current_process:
                try:
                    self._current_process.terminate()
                    await asyncio.wait_for(self._current_process.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    # Force kill if graceful termination did not complete
                    try:
                        self._current_process.kill()
                        await asyncio.wait_for(
                            self._current_process.wait(), timeout=2.0
                        )
                    except Exception:
                        pass  # best-effort cleanup; task already failed
            self.queue_manager.fail_task(
                task.id, f"Task exceeded timeout of {self.task_timeout:.0f}s"
            )

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

        try:
            # Spawn subprocess using the same Python interpreter
            self._current_process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self._worker_script),
                task_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.generated_dir.parent / "src"),
            )

            # Update task with PID
            self.queue_manager.update_task_pid(task.id, self._current_process.pid)

            # Start stderr reader concurrently
            stderr_task = asyncio.create_task(self._read_stderr(self._current_process, task.id))

            # Read output line by line
            result_data = None
            error_message = None

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._current_process.stdout.readline(),
                        timeout=300.0  # 5 minutes between output lines (image gen can be slow)
                    )
                except asyncio.TimeoutError:
                    if self._current_process.returncode is not None:
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
            await self._current_process.wait()

            # Collect stderr (already streamed to log events, keep for error reporting)
            stderr_lines = await stderr_task

            # Check exit code
            if self._current_process.returncode != 0 and error_message is None:
                error_message = "; ".join(stderr_lines[:3]) if stderr_lines else f"Process exited with code {self._current_process.returncode}"

            # Update task status
            if error_message:
                self.queue_manager.fail_task(task.id, error_message)
            else:
                self.queue_manager.complete_task(task.id, result_data)

        except asyncio.CancelledError:
            # Worker is being stopped
            if self._current_process:
                self._current_process.terminate()
                await self._current_process.wait()
            self.queue_manager.cancel_task(task.id)
            raise

        except Exception as e:
            self.queue_manager.fail_task(task.id, str(e))

        finally:
            # Clean up temp file
            try:
                os.unlink(task_file)
            except Exception as e:
                logging.debug(f"Failed to clean up task file {task_file}: {e}")

            self._current_process = None
