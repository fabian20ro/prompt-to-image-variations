"""Tests for Heartbeat context manager."""

import asyncio
import json
import threading
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.server.worker_subprocess import Heartbeat


@pytest.mark.asyncio
async def test_heartbeat_emits_progress():
    """Test that Heartbeat emits progress messages during its interval."""

    progress_events = []

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        progress_events.append((stage, current, total, message))

    # Patch emit_progress in the module where it is used
    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        with Heartbeat(message="test message", interval=0.1) as hb:
            # Allow enough time for at least one heartbeat
            await asyncio.sleep(0.25)

    assert len(progress_events) >= 1
    assert progress_events[0][0] == "heartbeat"
    assert progress_events[0][1] == 0
    assert progress_events[0][3] == "test message"

@pytest.mark.asyncio
async def test_heartbeat_stops_on_exit():
    """Test that Heartbeat stops when exiting the context manager."""

    heartbeats_count = 0

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        nonlocal heartbeats_count
        heartbeats_count += 1

    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        with Heartbeat(message="test message", interval=0.1) as hb:
            await asyncio.sleep(0.05)
            # Should have at least 0 or 1 heartbeats
            assert heartbeats_count < 5
            # Exit context manager
            pass

        # After exit, heartbeats should stop
        count_after_exit = heartbeats_count
        await asyncio.sleep(0.2)
        assert heartbeats_count == count_after_exit

@pytest.mark.asyncio
async def test_heartbeat_no_progress_on_immediate_exit():
    """Test that no progress is emitted if Heartbeat exits before first heartbeat interval."""

    progress_events = []

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        progress_events.append((stage, current, total, message))

    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        # Exit immediately - before first heartbeat interval elapses
        with Heartbeat(message="test", interval=1.0) as hb:
            pass  # Context manager exits immediately

    assert len(progress_events) == 0


@pytest.mark.asyncio
async def test_heartbeat_stops_on_exception():
    """Test that heartbeat thread stops cleanly when an exception occurs inside the with block."""

    heartbeats_count = 0

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        nonlocal heartbeats_count
        heartbeats_count += 1

    # Patch emit_progress so the counter tracks real emissions during the with-block.
    with patch("src.server.worker_subprocess.emit_progress", side_effect=mock_emit_progress):
        try:
            with Heartbeat(message="test", interval=0.1) as hb:
                await asyncio.sleep(0.25)  # Wait for at least one heartbeat
                assert heartbeats_count >= 1  # Verify heartbeat was emitted
                raise ValueError("simulated pipeline error")

        except ValueError:
            pass  # Context manager's __exit__ should have cleaned up the thread before exception propagates

    count_after_exit = heartbeats_count
    await asyncio.sleep(0.3)  # Wait to confirm no more heartbeats after exit
    assert heartbeats_count == count_after_exit  # Heartbeat thread should be stopped after context manager cleanup


@pytest.mark.asyncio
async def test_heartbeat_creates_daemon_thread():
    """Test that Heartbeat creates a daemon thread for process-safe cleanup."""

    with patch('src.server.worker_subprocess.emit_progress'):
        with Heartbeat(message="test", interval=1.0) as hb:
            # Verify thread is created and configured as daemon
            assert hb._thread is not None
            assert hb._thread.daemon is True, "Heartbeat should use daemon threads for safe cleanup"


@pytest.mark.asyncio
async def test_heartbeat_defaults():
    """Test that Heartbeat uses documented default parameter values."""

    hb = Heartbeat()
    assert hb.message == "Working...", "Default message must match constructor default"
    assert hb.interval == 30, "Default interval (seconds) must match constructor default"
    assert not hb._stop_event.is_set(), "_stop_event must start unset for default instance"

    # Verify emission works with a short interval using defaults applied to the object
    with patch('src.server.worker_subprocess.emit_progress') as mock_emit:
        heartbeat = Heartbeat()
        heartbeat.interval = 1
        with heartbeat:
            await asyncio.sleep(2.5)

    # Verify the thread started and emitted at least once with default values
    assert heartbeat._thread is not None
    assert any(call.args[0] == "heartbeat" for call in mock_emit.call_args_list), \
        "Default Heartbeat must emit 'heartbeat' stage progress"


@pytest.mark.asyncio
async def test_heartbeat_stop_event_lifecycle():
    """Test that the stop event transitions from unset to set exactly at __exit__."""

    with patch('src.server.worker_subprocess.emit_progress'):
        hb = Heartbeat(message="test", interval=1.0)
        assert not hb._stop_event.is_set(), "_stop_event must start unset before context entry"
        with hb:
            assert not hb._stop_event.is_set(), "Stop event should remain unset while active"
        # After __exit__, the stop event must be set, signalling thread join to proceed
        assert hb._stop_event.is_set(), "_stop_event must be set after __exit__"


@pytest.mark.asyncio
async def test_heartbeat_thread_stops_when_emit_raises():
    """Test that the heartbeat loop exits cleanly when emit_progress raises inside the thread.

    This ensures a transient failure in progress emission does not cause the
    Heartbeat thread to hang or block __exit__ indefinitely.
    """

    call_count = 0

    def failing_emit(stage: str, current: int = 0, total: int = 0, message: str = ""):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("emit_progress failure")

    with patch('src.server.worker_subprocess.emit_progress', side_effect=failing_emit):
        with Heartbeat(message="test", interval=0.05) as hb:
            await asyncio.sleep(0.3)

        # After __exit__, the stop event must be set and thread joined cleanly
        assert hb._stop_event.is_set(), "Stop event should be set after context manager exit"
        assert hb._thread is not None and not hb._thread.is_alive()
        # Thread should still exist (join happened) but no more heartbeats emitted
        assert call_count > 0, "At least one heartbeat was emitted before the error"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
