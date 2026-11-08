"""Worker behavior through its public loop and real persisted queue state."""

import asyncio
import json
import signal

import pytest
import pytest_asyncio

from server.models import TaskStatus, TaskType
from server.queue_manager import QueueManager
from server.worker import Worker


class ChildProcess:
    """Controllable external process, without a subprocess or GPU runtime."""

    def __init__(self):
        self.pid = 12345
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.exited = asyncio.Event()
        self.started = asyncio.Event()
        self.ignore_terminate = False
        self.reap_delay_once = 0
        self.keep_stderr_open = False

    def finish(self, code=0):
        if self.returncode is None:
            self.returncode = code
            self.stdout.feed_eof()
            if not self.keep_stderr_open:
                self.stderr.feed_eof()
            self.exited.set()

    def terminate(self):
        if not self.ignore_terminate:
            self.finish(-signal.SIGTERM)

    def kill(self):
        self.finish(-signal.SIGKILL)

    async def wait(self):
        await self.exited.wait()
        delay, self.reap_delay_once = self.reap_delay_once, 0
        if delay:
            await asyncio.sleep(delay)
        return self.returncode

    def result(self, **data):
        self.stdout.feed_data((json.dumps({"type": "result", "success": True, "data": data}) + "\n").encode())
        self.finish()


@pytest_asyncio.fixture
async def queued_worker(monkeypatch, tmp_path):
    queue = QueueManager(tmp_path / "queue.json")
    child = ChildProcess()

    async def spawn(*args, **kwargs):
        child.started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return queue, Worker(queue, tmp_path / "generated"), child


async def run_until_terminal(queue, worker):
    """Stop the public loop after a real queue terminal notification."""
    events = []

    def observe(event, data):
        events.append((event, data))
        if event in {"task_completed", "task_failed", "task_cancelled"}:
            worker.stop()

    queue.add_listener(observe)
    try:
        await asyncio.wait_for(worker.run(), timeout=1)
    finally:
        queue.remove_listener(observe)
    return events


@pytest.mark.asyncio
async def test_explicit_deadline_is_reported_as_failure_not_user_cancellation(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    worker.task_timeout = 0.02
    events = await run_until_terminal(queue, worker)
    saved = queue.get_task(task.id)
    assert saved.status == TaskStatus.FAILED
    assert "timeout" in saved.error.lower()
    assert not any(event == "task_cancelled" for event, _ in events)
    assert child.returncode == -signal.SIGTERM


@pytest.mark.asyncio
async def test_default_worker_does_not_kill_long_healthy_batch(queued_worker, monkeypatch):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    real_wait_for = asyncio.wait_for

    async def accelerated_deadline(awaitable, timeout):
        # Accelerate only the former fixed 30-minute deadline, not liveness.
        return await real_wait_for(awaitable, 0.01 if timeout == 1800 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", accelerated_deadline)

    async def finish_batch():
        await child.started.wait()
        child.stdout.feed_data(b'{"type":"progress","stage":"generating_images","current":1,"total":2}\n')
        await asyncio.sleep(0.04)
        if child.returncode is None:
            child.result(image_count=2)

    producer = asyncio.create_task(finish_batch())
    try:
        await run_until_terminal(queue, worker)
        saved = queue.get_task(task.id)
        assert saved.status == TaskStatus.COMPLETED
        assert saved.result == {"image_count": 2}
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


@pytest.mark.asyncio
async def test_deadline_force_kills_child_that_ignores_termination(queued_worker, monkeypatch):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.ignore_terminate = True
    worker.task_timeout = 0.01
    real_wait_for = asyncio.wait_for

    async def accelerated_cleanup(awaitable, timeout):
        return await real_wait_for(awaitable, 0.01 if timeout in {2, 5} else timeout)

    monkeypatch.setattr(asyncio, "wait_for", accelerated_cleanup)
    await run_until_terminal(queue, worker)
    assert child.returncode == -signal.SIGKILL
    assert queue.get_task(task.id).status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_user_cancel_is_not_reported_as_process_failure(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.reap_delay_once = 0.02
    runner = asyncio.create_task(run_until_terminal(queue, worker))
    try:
        await asyncio.wait_for(child.started.wait(), timeout=1)
        assert await worker.kill_current()
        events = await runner
        assert queue.get_task(task.id).status == TaskStatus.CANCELLED
        assert not any(event in {"task_completed", "task_failed"} for event, _ in events)
    finally:
        if not runner.done():
            runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_crash_preserves_signal_and_recent_bounded_stderr(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.stderr.feed_data(b"startup diagnostics, not the crash\n")
    for index in range(80):
        child.stderr.feed_data(f"progress {index}: {'x' * 2048}\n".encode())
    child.stderr.feed_data(b"[METAL] Insufficient Memory\n")
    child.finish(-signal.SIGABRT)
    await run_until_terminal(queue, worker)
    saved = queue.get_task(task.id)
    assert saved.status == TaskStatus.FAILED
    assert "SIGABRT" in saved.error
    assert "code -6" in saved.error
    assert "[METAL] Insufficient Memory" in saved.error
    assert "startup diagnostics" not in saved.error
    assert len(saved.error) < 25_000


@pytest.mark.asyncio
async def test_child_error_keeps_exit_code_and_stderr_context(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.stdout.feed_data(b'{"type":"result","success":false,"error":"Image save failed"}\n')
    child.stderr.feed_data(b"OSError: disk full\n")
    child.finish(1)
    await run_until_terminal(queue, worker)
    error = queue.get_task(task.id).error
    assert "Image save failed" in error
    assert "code 1" in error
    assert "OSError: disk full" in error


@pytest.mark.asyncio
async def test_long_or_invalid_stderr_does_not_hide_native_crash(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.stderr.feed_data(b"x" * 100_000 + b"\n\xff fatal memory failure\n")
    child.finish(-signal.SIGKILL)
    await run_until_terminal(queue, worker)
    error = queue.get_task(task.id).error
    assert "SIGKILL" in error
    assert "fatal memory failure" in error
    assert len(error) < 25_000


@pytest.mark.asyncio
async def test_inherited_stderr_pipe_cannot_hang_exited_worker(queued_worker, monkeypatch):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.keep_stderr_open = True
    child.stderr.feed_data(b"fatal native error\n")
    child.finish(-signal.SIGABRT)
    real_wait_for = asyncio.wait_for

    async def accelerated_cleanup(awaitable, timeout):
        return await real_wait_for(awaitable, 0.01 if timeout in {2, 5} else timeout)

    monkeypatch.setattr(asyncio, "wait_for", accelerated_cleanup)
    await run_until_terminal(queue, worker)
    error = queue.get_task(task.id).error
    assert "SIGABRT" in error
    assert "fatal native error" in error


@pytest.mark.asyncio
async def test_protocol_failure_cannot_leave_image_process_running(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    exit_codes_at_failure = []
    queue.add_listener(lambda event, data: exit_codes_at_failure.append(child.returncode)
                       if event == "task_failed" else None)
    child.stdout.feed_data(b'{"type":"progress","current":"not a number"}\n')
    await run_until_terminal(queue, worker)
    assert queue.get_task(task.id).status == TaskStatus.FAILED
    assert exit_codes_at_failure == [-signal.SIGTERM]


@pytest.mark.asyncio
async def test_server_shutdown_cancels_task_and_reaps_child(queued_worker):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    runner = asyncio.create_task(worker.run())
    await asyncio.wait_for(child.started.wait(), timeout=1)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert queue.get_task(task.id).status == TaskStatus.CANCELLED
    assert child.returncode == -signal.SIGTERM


@pytest.mark.asyncio
async def test_manual_cancel_has_bounded_wait_even_after_force_kill(queued_worker, monkeypatch):
    queue, worker, child = queued_worker
    task = queue.add_task(TaskType.GENERATE_ALL_IMAGES, {})
    child.ignore_terminate = True
    child.reap_delay_once = 100
    real_wait_for = asyncio.wait_for

    async def accelerated_cleanup(awaitable, timeout):
        return await real_wait_for(awaitable, 0.01 if timeout in {2, 5} else timeout)

    monkeypatch.setattr(asyncio, "wait_for", accelerated_cleanup)
    runner = asyncio.create_task(worker.run())
    try:
        await real_wait_for(child.started.wait(), timeout=1)
        assert await real_wait_for(worker.kill_current(), timeout=0.2)
        assert queue.get_task(task.id).status == TaskStatus.CANCELLED
    finally:
        worker.stop()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
