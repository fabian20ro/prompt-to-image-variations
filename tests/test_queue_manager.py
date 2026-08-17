"""Tests for QueueManager."""

from server.models import TaskType, TaskStatus, TaskProgress
from server.queue_manager import QueueManager


class TestQueueManager:
    """Tests for QueueManager."""

    def test_load_empty_queue(self, queue_path):
        """Test loading an empty/nonexistent queue."""
        qm = QueueManager(queue_path)
        state = qm.get_state()

        assert state.version == 1
        assert state.current_task is None
        assert len(state.pending) == 0

    def test_add_task(self, queue_path):
        """Test adding a task to the queue."""
        qm = QueueManager(queue_path)

        task = qm.add_task(
            TaskType.GENERATE_PIPELINE,
            {"prompt": "test prompt", "count": 10},
        )

        assert task.type == TaskType.GENERATE_PIPELINE
        assert task.status == TaskStatus.PENDING
        assert task.params["prompt"] == "test prompt"
        assert task.params["count"] == 10

        # Verify state was saved
        state = qm.get_state()
        assert len(state.pending) == 1
        assert state.pending[0].id == task.id

    def test_get_next_task(self, queue_path):
        """Test getting the next task from the queue."""
        qm = QueueManager(queue_path)

        # Add a task
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        # Get the task
        next_task = qm.get_next_task()

        assert next_task is not None
        assert next_task.id == task.id
        assert next_task.status == TaskStatus.RUNNING

        # Verify queue state
        state = qm.get_state()
        assert len(state.pending) == 0
        assert state.current_task is not None
        assert state.current_task.id == task.id

    def test_get_next_task_when_running(self, queue_path):
        """Test that get_next_task returns None when a task is running."""
        qm = QueueManager(queue_path)

        # Add two tasks
        second_task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})
        first_task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})

        # Get first task (LIFO — last added goes in queue head)
        qm.get_next_task()

        # Should return None since a task is running
        next_task = qm.get_next_task()
        assert next_task is None

        state = qm.get_state()
        assert len(state.pending) == 1
        assert state.pending[0].status == TaskStatus.PENDING
        assert state.current_task.id == second_task.id
        assert state.current_task.status == TaskStatus.RUNNING

    def test_complete_task(self, queue_path):
        """Test completing a task."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        # Complete the task
        qm.complete_task(task.id, {"generated": 5})

        state = qm.get_state()
        assert state.current_task is None
        assert len(state.completed) == 1
        assert state.completed[0].status == TaskStatus.COMPLETED
        assert state.completed[0].result == {"generated": 5}

    def test_fail_task(self, queue_path):
        """Test failing a task."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        # Fail the task
        qm.fail_task(task.id, "Something went wrong")

        state = qm.get_state()
        assert state.current_task is None
        assert len(state.completed) == 1
        assert state.completed[0].status == TaskStatus.FAILED
        assert state.completed[0].error == "Something went wrong"

    def test_update_progress(self, queue_path):
        """Test updating task progress."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_PIPELINE, {"prompt": "test"})
        qm.get_next_task()

        # Update progress
        progress = TaskProgress(
            stage="generating_images",
            current=5,
            total=10,
            message="Working...",
        )
        qm.update_progress(task.id, progress)

        state = qm.get_state()
        assert state.current_task.progress.stage == "generating_images"
        assert state.current_task.progress.current == 5
        assert state.current_task.progress.total == 10

    def test_cancel_pending_task(self, queue_path):
        """Test cancelling a pending task."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        # Cancel the pending task
        result = qm.cancel_task(task.id)

        assert result is True
        state = qm.get_state()
        assert len(state.pending) == 0
        assert len(state.completed) == 1
        assert state.completed[0].status == TaskStatus.CANCELLED

    def test_cancel_running_task(self, queue_path):
        """Test cancelling a running task."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        # Cancel the running task
        result = qm.cancel_task(task.id)

        assert result is True
        state = qm.get_state()
        assert state.current_task is None
        assert len(state.completed) == 1
        assert state.completed[0].status == TaskStatus.CANCELLED

    def test_clear_pending(self, queue_path):
        """Test clearing all pending tasks."""
        qm = QueueManager(queue_path)

        # Add multiple tasks
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test3"})

        # Clear pending
        count = qm.clear_pending()

        assert count == 3
        state = qm.get_state()
        assert len(state.pending) == 0

    def test_clear_pending_emits_queue_cleared_and_queue_updated(self, queue_path):
        """Clearing pending tasks should emit both clear and updated queue events."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})

        # Ignore add_task notifications and inspect clear behavior only.
        events.clear()
        count = qm.clear_pending()

        assert count == 2
        assert len(events) == 2
        assert events[0][0] == "queue_cleared"
        assert events[0][1]["count"] == 2
        assert events[1][0] == "queue_updated"
        assert events[1][1]["pending_count"] == 0
        assert events[1][1]["current"] is None

    def test_persistence(self, queue_path):
        """Test that queue state is persisted to disk."""
        # Create queue and add task
        qm1 = QueueManager(queue_path)
        task = qm1.add_task(TaskType.GENERATE_PIPELINE, {"prompt": "test"})

        # Create new QueueManager instance
        qm2 = QueueManager(queue_path)
        state = qm2.get_state()

        assert len(state.pending) == 1
        assert state.pending[0].id == task.id
        assert state.pending[0].params["prompt"] == "test"

    def test_listener_notifications(self, queue_path):
        """Test that listeners are notified of events."""
        qm = QueueManager(queue_path)

        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)

        # Add a task
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        assert len(events) == 1
        assert events[0][0] == "queue_updated"

        # Get next task
        qm.get_next_task()

        assert len(events) == 2
        assert events[1][0] == "task_started"

        # Complete task
        qm.complete_task(task.id, {"result": "ok"})

        assert len(events) == 3
        assert events[2][0] == "task_completed"

    def test_listener_exception_does_not_block_other_listeners(self, queue_path):
        """A failing listener should not prevent other listeners from receiving events."""
        qm = QueueManager(queue_path)

        received_events = []

        def good_listener(event, data):
            received_events.append(("good", event))

        def bad_listener(event, data):
            raise RuntimeError("listener boom")

        qm.add_listener(bad_listener)
        qm.add_listener(good_listener)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        assert len([e for e in received_events if e[0] == "good"]) == 1

    def test_listener_add_remove_thread_safety(self, queue_path):
        """Test that listeners can be added/removed safely during notification."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append(event)

        qm.add_listener(listener)
        task = qm.add_task("generate_pipeline", {"prompt": "test"})
        assert len(events) == 1  # queue_updated

        qm.remove_listener(listener)
        qm.add_task("generate_pipeline", {"prompt": "test2"})
        assert len(events) == 1  # No new events after removal

    def test_update_task_pid(self, queue_path):
        """Test that update_task_pid stores the PID on the running task."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        # Ignore add/get notifications
        events.clear()

        qm.update_task_pid(task.id, 4242)

        state = qm.get_state()
        assert state.current_task.pid == 4242
        # update_task_pid does not emit a notification event
        assert len(events) == 0

    def test_update_task_pid_persists(self, queue_path):
        """Test that the updated PID survives across QueueManager instances."""
        qm1 = QueueManager(queue_path)
        task = qm1.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm1.get_next_task()
        qm1.update_task_pid(task.id, 9999)

        qm2 = QueueManager(queue_path)
        state = qm2.get_state()
        assert state.current_task.pid == 9999

    def test_update_task_pid_ignores_wrong_id(self, queue_path):
        """update_task_pid should not mutate state when the task id is wrong."""
        qm = QueueManager(queue_path)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        original_state = qm.get_state().current_task.pid
        qm.update_task_pid("nonexistent-id", 1234)

        state = qm.get_state()
        assert state.current_task.id == task.id
        assert state.current_task.pid == original_state

    def test_update_task_pid_no_current_task(self, queue_path):
        """update_task_pid should be a no-op when the queue is empty."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)

        # No current task — update_task_pid should not raise or mutate state.
        qm.update_task_pid("any-id", 1111)

        state = qm.get_state()
        assert state.current_task is None
        assert len(events) == 0


    def test_cancel_nonexistent_task_returns_false(self, queue_path):
        """cancel_task should return False for a non-existent task id without mutating state."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)

        result = qm.cancel_task("nonexistent-id")

        assert result is False
        # No mutation: no pending tasks affected, no completed list touched.
        state = qm.get_state()
        assert len(state.pending) == 0
        assert len(state.completed) == 0
        assert state.current_task is None
        # And no spurious event was emitted for a miss.
        assert len(events) == 0


    def test_complete_task_wrong_id_is_noop(self, queue_path):
        """complete_task should not mutate state when the task id doesn't match."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        # Ignore notifications from add/get.
        events.clear()

        original_completed_count = len(qm.get_state().completed)
        qm.complete_task("nonexistent-id", {"result": "ok"})

        state = qm.get_state()
        assert state.current_task is not None  # current task unchanged
        assert state.current_task.id == task.id
        assert state.current_task.status == TaskStatus.RUNNING
        assert len(state.completed) == original_completed_count  # no completed entry added
        assert len(events) == 0  # no spurious event emitted


class TestGetTask:
    """Tests for QueueManager.get_task lookup."""

    def test_get_task_pending(self, queue_path):
        """get_task should find pending tasks by id."""
        qm = QueueManager(queue_path)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        found = qm.get_task(task.id)

        assert found is not None
        assert found.id == task.id
        assert found.status == TaskStatus.PENDING

    def test_get_task_running(self, queue_path):
        """get_task should find running tasks by id."""
        qm = QueueManager(queue_path)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()  # moves to RUNNING

        found = qm.get_task(task.id)

        assert found is not None
        assert found.id == task.id
        assert found.status == TaskStatus.RUNNING

    def test_get_task_completed(self, queue_path):
        """get_task should find completed tasks by id."""
        qm = QueueManager(queue_path)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()
        qm.complete_task(task.id, {"result": "ok"})

        found = qm.get_task(task.id)

        assert found is not None
        assert found.id == task.id
        assert found.status == TaskStatus.COMPLETED

    def test_get_task_not_found(self, queue_path):
        """get_task should return None for non-existent ids."""
        qm = QueueManager(queue_path)

        found = qm.get_task("nonexistent-id")

        assert found is None


class TestGetCurrentTaskPid:
    """Tests for QueueManager.get_current_task_pid."""

    def test_get_current_task_pid_returns_pid(self, queue_path):
        """get_current_task_pid should return the PID when a task is running."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()
        qm.update_task_pid(task.id, 4242)

        # Ignore add/get notifications
        events.clear()

        pid = qm.get_current_task_pid()

        assert pid == 4242
        assert len(events) == 0

    def test_get_current_task_pid_no_running(self, queue_path):
        """get_current_task_pid should return None when no task is running."""
        qm = QueueManager(queue_path)

        pid = qm.get_current_task_pid()

        assert pid is None


class TestEmitLog:
    """Tests for QueueManager.emit_log."""

    def test_emit_log_emits_task_log_event(self, queue_path):
        """emit_log should emit a 'task_log' event with the correct payload shape."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)

        # Ignore add_task notifications.
        events.clear()

        qm.emit_log("task-xyz", "Step 3/5 complete")

        assert len(events) == 1
        event_type, payload = events[0]
        assert event_type == "task_log"
        assert payload["task_id"] == "task-xyz"
        assert payload["message"] == "Step 3/5 complete"
        assert isinstance(payload["timestamp"], str)

    def test_emit_log_no_spurious_events(self, queue_path):
        """emit_log should not mutate state or emit other events."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()
        events.clear()

        qm.emit_log(task.id, "log line")

        state = qm.get_state()
        assert len(events) == 1
        assert events[0][0] == "task_log"
        assert state.current_task is not None  # task unchanged


class TestNotifyImageReady:
    """Tests for QueueManager.notify_image_ready."""

    def test_notify_image_ready_emits_event(self, queue_path):
        """notify_image_ready should emit an 'image_ready' event with the correct payload."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)

        # Ignore add_task notifications.
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        events.clear()

        qm.notify_image_ready("run-abc", "/output/img.png")

        assert len(events) == 1
        event_type, payload = events[0]
        assert event_type == "image_ready"
        assert payload["run_id"] == "run-abc"
        assert payload["path"] == "/output/img.png"

    def test_notify_image_ready_does_not_mutate_state(self, queue_path):
        """notify_image_ready should not change any queue state."""
        qm = QueueManager(queue_path)
        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
        qm.get_next_task()

        original_pending_count = len(qm.get_state().pending)
        original_completed_count = len(qm.get_state().completed)

        qm.notify_image_ready("run-xyz", "/output/img2.png")

        state = qm.get_state()
        assert len(state.pending) == original_pending_count
        assert len(state.completed) == original_completed_count
        assert state.current_task is not None  # current task unchanged
