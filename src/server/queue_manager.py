"""Disk-based queue management for task persistence."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Callable

from .models import QueueState, Task, TaskStatus, TaskType, TaskProgress


class QueueManager:
    """Manages the task queue with disk persistence."""

    def __init__(self, queue_path: Path):
        """Initialize queue manager.
        
        Args:
            queue_path: Path to the queue.
        """
        self.queue_path = queue_path
        # Reentrant lock: _notify() runs listeners while the lock is held, and
        # a listener may legitimately re-enter the public API (e.g. remove
        # itself, or call get_state) from inside its own callback.
        self._lock = RLock()
        self._state = self._load_state()
        self._listeners: list[Callable[[str, dict], None]] = []

    def _load_state(self) -> QueueState:
        """Load queue state from disk."""
        if self.queue_path.exists():
            try:
                data = json.loads(self.queue_path.read_text())
                return QueueState.model_validate(data)
            except json.JSONDecodeError as e:
                logging.warning(f"Corrupted queue file, resetting: {e}")
            except Exception as e:
                logging.error(f"Failed to load queue state: {e}")
        return QueueState()

    def _save_state(self, state: QueueState) -> None:
        """Save queue state to disk atomically."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp file first, then atomic rename (POSIX rename is atomic)
        tmp_path = self.queue_path.with_suffix('.tmp')
        tmp_path.write_text(state.model_dump_json(indent=2))
        tmp_path.rename(self.queue_path)

    def _notify(self, event: str, data: dict) -> None:
        """Notify all listeners of an event.

        Called under self._lock. Iterates a snapshot to allow
        concurrent add/remove operations.
        """
        for listener in list(self._listeners):
            try:
                listener(event, data)
            except Exception as e:
                logging.exception(f"Error in queue listener for event {event}: {e}")

    def add_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Add an event listener (thread-safe)."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Remove an event listener (thread-safe)."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def add_task(
        self,
        task_type: TaskType,
        params: dict,
    ) -> Task:
        """Add a new task to the queue.

        Args:
            task_type: Type of task to create
            params: Parameters for the task

        Returns:
            The created task
        """
        with self._lock:
            state = self._state

            task = Task(
                id=str(uuid.uuid4()),
                type=task_type,
                status=TaskStatus.PENDING,
                params=params,
            )
            state.pending.append(task)
            self._save_state(state)

            self._notify("queue_updated", {
                "pending_count": len(state.pending),
                "current": state.current_task.model_dump(mode='json') if state.current_task else None,
            })

            return task

    def get_next_task(self) -> Task | None:
        """Get the next pending task and mark it as running.

        Returns:
            The next task, or None if queue is empty or a task is running
        """
        with self._lock:
            state = self._state

            # Don't start new task if one is running
            if state.current_task and state.current_task.status == TaskStatus.RUNNING:
                return None

            if not state.pending:
                return None

            task = state.pending.pop(0)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            state.current_task = task
            self._save_state(state)

            self._notify("task_started", task.model_dump(mode='json'))

            return task

    def update_progress(self, task_id: str, progress: TaskProgress) -> None:
        """Update progress for the current task.

        Args:
            task_id: ID of the task to update
            progress: Progress information
        """
        with self._lock:
            state = self._state

            if state.current_task and state.current_task.id == task_id:
                state.current_task.progress = progress
                self._save_state(state)

                self._notify("task_progress", {
                    "task_id": task_id,
                    "stage": progress.stage,
                    "current": progress.current,
                    "total": progress.total,
                    "message": progress.message,
                })

    def update_task_pid(self, task_id: str, pid: int) -> None:
        """Update the PID for a running task.

        Args:
            task_id: ID of the task
            pid: Process ID of the worker subprocess

        If ``task_id`` does not match the current task (or there is no current
        task), this method is a no-op — no state mutation and no notification
        event are emitted. This guarantees deterministic behavior when callers
        dispatch PID updates to arbitrary ids without prior lookup.
        """
        with self._lock:
            state = self._state

            if state.current_task and state.current_task.id == task_id:
                state.current_task.pid = pid
                self._save_state(state)

    def complete_task(self, task_id: str, result: dict | None = None) -> None:
        """Mark a task as completed.

        Args:
            task_id: ID of the task
            result: Optional result data
        """
        with self._lock:
            state = self._state

            if state.current_task and state.current_task.id == task_id:
                state.current_task.status = TaskStatus.COMPLETED
                state.current_task.completed_at = datetime.now()
                state.current_task.result = result
                state.current_task.pid = None

                # Move to completed list (keep last 50)
                state.completed.insert(0, state.current_task)
                state.completed = state.completed[:50]
                state.current_task = None
                self._save_state(state)

                self._notify("task_completed", {
                    "task_id": task_id,
                    "result": result,
                })

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: ID of the task
            error: Error message
        """
        with self._lock:
            state = self._state

            if state.current_task and state.current_task.id == task_id:
                state.current_task.status = TaskStatus.FAILED
                state.current_task.completed_at = datetime.now()
                state.current_task.error = error
                state.current_task.pid = None

                # Move to completed list
                state.completed.insert(0, state.current_task)
                state.completed = state.completed[:50]
                state.current_task = None
                self._save_state(state)

                self._notify("task_failed", {
                    "task_id": task_id,
                    "error": error,
                })

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was found and cancelled
        """
        with self._lock:
            state = self._state

            # Check if it's the current task
            if state.current_task and state.current_task.id == task_id:
                pid = state.current_task.pid
                state.current_task.status = TaskStatus.CANCELLED
                state.current_task.completed_at = datetime.now()
                state.current_task.pid = None

                state.completed.insert(0, state.current_task)
                state.completed = state.completed[:50]
                state.current_task = None
                self._save_state(state)

                self._notify("task_cancelled", {"task_id": task_id})
                return True

            # Check pending tasks
            for i, task in enumerate(state.pending):
                if task.id == task_id:
                    task.status = TaskStatus.CANCELLED
                    state.pending.pop(i)
                    state.completed.insert(0, task)
                    state.completed = state.completed[:50]
                    self._save_state(state)

                    self._notify("task_cancelled", {"task_id": task_id})
                    return True

            return False

    def clear_pending(self) -> int:
        """Clear all pending tasks.

        Returns:
            Number of tasks cleared
        """
        with self._lock:
            state = self._state
            count = len(state.pending)
            state.pending = []
            self._save_state(state)

            if count > 0:
                self._notify("queue_cleared", {"count": count})
                self._notify("queue_updated", {
                    "pending_count": len(state.pending),
                    "current": state.current_task.model_dump(mode='json') if state.current_task else None,
                })

            return count

    def get_state(self) -> QueueState:
        """Get the current queue state.

        Returns:
            Current queue state
        """
        with self._lock:
            return self._state

    def get_current_task_pid(self) -> int | None:
        """Get the PID of the currently running task.

        Returns:
            PID if a task is running, None otherwise
        """
        with self._lock:
            state = self._state
            if state.current_task and state.current_task.status == TaskStatus.RUNNING:
                return state.current_task.pid
            return None

    def notify_image_ready(self, run_id: str, path: str) -> None:
        """Notify listeners that an image is ready.

        Args:
            run_id: The run ID
            path: Path to the generated image
        """
        self._notify("image_ready", {
            "run_id": run_id,
            "path": path,
        })

    def get_task(self, task_id: str) -> Task | None:
        """Find and return any task by id (pending / current / completed).

        Args:
            task_id: ID of the task to look up

        Returns:
            The matching task, or None if not found.
        """
        with self._lock:
            state = self._state
            for t in state.pending:
                if t.id == task_id:
                    return t
            if state.current_task and state.current_task.id == task_id:
                return state.current_task
            for t in state.completed[:]:
                if t.id == task_id:
                    return t
            return None

    def emit_log(self, task_id: str, message: str) -> None:
        """Emit a log line from the worker subprocess.

        Args:
            task_id: ID of the task
            message: Log message to emit
        """
        self._notify("task_log", {
            "task_id": task_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
