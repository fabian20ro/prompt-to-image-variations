"""Tests for API routes."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import models first (no circular imports)
from server.models import TaskStatus, Task, TaskType, QueueState


@pytest.fixture
def mock_queue_manager():
    """Create a mock queue manager."""
    mock_qm = MagicMock()
    mock_qm.get_state.return_value = QueueState()
    mock_qm.add_task.return_value = Task(
        id="test-task-123",
        type=TaskType.GENERATE_PIPELINE,
        status=TaskStatus.PENDING,
    )
    return mock_qm


@pytest.fixture
def mock_worker():
    """Create a mock worker."""
    mock_w = MagicMock()
    mock_w.kill_current = AsyncMock(return_value=True)
    return mock_w


@pytest.fixture
def client(temp_dir, mock_queue_manager, mock_worker):
    """Create test client with mocked dependencies."""
    # Create minimal directory structure
    prompts_dir = temp_dir / "prompts"
    saved_dir = temp_dir / "saved"
    prompts_dir.mkdir()
    saved_dir.mkdir()

    # Create fresh app to avoid circular import issues
    app = FastAPI()

    # Patch before importing routes
    with patch("server.app.get_queue_manager", return_value=mock_queue_manager):
        with patch("server.app.get_worker", return_value=mock_worker):
            # Import router inside the patch context
            import server.routes as routes_module
            from services.gallery_service import GalleryService

            # Patch the functions in routes module
            routes_module.get_queue_manager = lambda: mock_queue_manager
            routes_module.get_worker = lambda: mock_worker

            # Patch paths
            routes_module.paths = MagicMock()
            routes_module.paths.prompts_dir = prompts_dir
            routes_module.paths.saved_dir = saved_dir
            routes_module.paths.generated_dir = temp_dir

            # Clear the lru_cache and create test service
            routes_module.get_gallery_service.cache_clear()
            test_service = GalleryService(prompts_dir, saved_dir)

            # Use FastAPI dependency override
            app.dependency_overrides[routes_module.get_gallery_service] = lambda: test_service

            app.include_router(routes_module.router)

            with TestClient(app) as client:
                yield client

            # Clean up
            app.dependency_overrides.clear()
            routes_module.get_gallery_service.cache_clear()


class TestGenerateEndpoint:
    """Tests for /api/generate endpoint."""

    def test_generate_valid_request(self, client, mock_queue_manager):
        """Test generating with valid request."""
        response = client.post("/api/generate", json={
            "prompt": "a dragon flying",
            "count": 10,
        })

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "message" in data

    def test_generate_empty_prompt_rejected(self, client):
        """Test that empty prompt is rejected."""
        response = client.post("/api/generate", json={
            "prompt": "",
        })

        assert response.status_code == 422  # Validation error

    def test_generate_whitespace_prompt_rejected(self, client):
        """Test that whitespace-only prompt is rejected."""
        response = client.post("/api/generate", json={
            "prompt": "   ",
        })

        assert response.status_code == 400
        assert "required" in response.json()["detail"].lower()

    def test_generate_count_bounds(self, client):
        """Test count validation bounds."""
        # Too low
        response = client.post("/api/generate", json={
            "prompt": "test",
            "count": 0,
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/api/generate", json={
            "prompt": "test",
            "count": 20000,
        })
        assert response.status_code == 422


class TestGenerateFromGrammarEndpoint:
    """Tests for /api/generate-from-grammar endpoint."""

    def test_generate_from_grammar_valid_request(self, client):
        response = client.post("/api/generate-from-grammar", json={
            "grammar": '{"origin": ["hello"]}',
            "title": "Grammar import",
            "count": 5,
        })

        assert response.status_code == 200
        assert "task_id" in response.json()

    def test_generate_from_grammar_rejects_invalid_json(self, client):
        response = client.post("/api/generate-from-grammar", json={
            "grammar": '{"origin": [}',
        })

        assert response.status_code == 400
        assert "Invalid JSON grammar" in response.json()["detail"]


class TestGalleryEndpoint:
    """Tests for /gallery/{run_id} endpoint."""

    def test_gallery_not_found(self, client):
        """Test 404 for missing gallery."""
        response = client.get("/gallery/nonexistent_gallery")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_gallery_exists(self, client, temp_dir):
        """Test serving existing gallery."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "test",
            "image_generation": {"images_per_prompt": 1},
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["test"]}))
        (run_dir / "test_0.txt").write_text("Test prompt")

        response = client.get("/gallery/20240101_120000_abc123")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestGrammarUpdateEndpoint:
    """Tests for PUT /api/gallery/{run_id}/grammar."""

    def test_update_grammar_invalid_json(self, client, temp_dir):
        """Test that invalid JSON is rejected."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["test"]}))

        response = client.put("/api/gallery/20240101_120000_abc123/grammar", json={
            "grammar": "not valid json {",
        })

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_update_grammar_gallery_not_found(self, client):
        """Test 404 for updating grammar of missing gallery."""
        response = client.put("/api/gallery/nonexistent/grammar", json={
            "grammar": '{"origin": ["test"]}',
        })

        assert response.status_code == 404

    def test_update_grammar_success(self, client, temp_dir):
        """Test successful grammar update."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["old"]}))

        new_grammar = '{"origin": ["new"]}'
        response = client.put("/api/gallery/20240101_120000_abc123/grammar", json={
            "grammar": new_grammar,
        })

        assert response.status_code == 200
        assert response.json()["message"] == "Grammar updated"

        # Verify file was updated
        updated = (run_dir / "test_grammar.json").read_text()
        assert updated == new_grammar

    def test_update_grammar_appends_history(self, client, temp_dir):
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["old"]}))

        response = client.put("/api/gallery/20240101_120000_abc123/grammar", json={
            "grammar": '{"origin": ["new"]}',
        })

        assert response.status_code == 200
        history = json.loads((run_dir / "test_grammar_history.json").read_text())
        assert history[-1]["grammar"] == '{"origin": ["new"]}'
        assert history[-1]["action"] == "save"


class TestGrammarHistoryEndpoint:
    """Tests for GET /api/gallery/{run_id}/grammar/history."""

    def test_get_grammar_history_returns_current_revision(self, client, temp_dir):
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["current"]}))

        response = client.get("/api/gallery/20240101_120000_abc123/grammar/history")

        assert response.status_code == 200
        data = response.json()
        assert data["history"][0]["grammar"] == '{"origin": ["current"]}'


class TestLayoutEndpoint:
    """Tests for PUT /api/gallery/{run_id}/layout."""

    def test_update_layout_persists_gallery_layout(self, client, temp_dir):
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        response = client.put("/api/gallery/20240101_120000_abc123/layout", json={
            "images_per_prompt": 0,
            "max_prompts": 8,
        })

        assert response.status_code == 200
        metadata = json.loads((run_dir / "test.metaprompt.json").read_text())
        assert metadata["gallery_layout"] == {"images_per_prompt": 0, "max_prompts": 8}


class TestGenerateAllEndpoint:
    """Tests for POST /api/gallery/{run_id}/generate-all."""

    def test_generate_all_accepts_zero_images_per_prompt(self, client, temp_dir, mock_queue_manager):
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        response = client.post("/api/gallery/20240101_120000_abc123/generate-all", json={
            "images_per_prompt": 0,
            "resume": True,
        })

        assert response.status_code == 200
        queued_params = mock_queue_manager.add_task.call_args.args[1]
        assert queued_params["images_per_prompt"] == 0
        assert queued_params["resume"] is True


class TestRegenerateEndpoint:
    """Tests for POST /api/gallery/{run_id}/regenerate."""

    def test_regenerate_auto_saves_grammar_and_layout(self, client, temp_dir, mock_queue_manager):
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 10,
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["old"]}))

        response = client.post("/api/gallery/20240101_120000_abc123/regenerate", json={
            "grammar": '{"origin": ["new"]}',
            "images_per_prompt": 4,
            "max_prompts": 6,
        })

        assert response.status_code == 200
        assert (run_dir / "test_grammar.json").read_text() == '{"origin": ["new"]}'
        queued_params = mock_queue_manager.add_task.call_args.args[1]
        assert queued_params["grammar"] == '{"origin": ["new"]}'
        assert queued_params["images_per_prompt"] == 4
        assert queued_params["max_prompts"] == 6


class TestDeleteGalleryEndpoint:
    """Tests for DELETE /api/gallery/{run_id}."""

    def test_delete_gallery_not_found(self, client):
        """Test 404 for deleting missing gallery."""
        response = client.delete("/api/gallery/nonexistent")
        assert response.status_code == 404

    def test_delete_archive_protected(self, client, temp_dir):
        """Test that archives cannot be deleted (returns 400)."""
        # Create an archive (backup)
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
            },
        }))

        response = client.delete("/api/gallery/20240101_120000_abc123")
        assert response.status_code == 400
        assert "archived" in response.json()["detail"].lower()

    def test_delete_active_gallery_queued(self, client, temp_dir, mock_queue_manager):
        """Test that delete queues a task for active gallery."""
        # Create an active gallery (not a backup)
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        response = client.delete("/api/gallery/20240101_120000_abc123")
        assert response.status_code == 200
        assert "queued" in response.json()["message"].lower()


class TestSavedFileEndpoint:
    """Tests for /saved/{filename} endpoint."""

    def test_saved_file_not_found(self, client):
        """Test 404 for missing file."""
        response = client.get("/saved/nonexistent.png")
        assert response.status_code == 404

    def test_saved_file_non_png_rejected(self, client, temp_dir):
        """Test that non-PNG files are rejected."""
        saved_dir = temp_dir / "saved"
        (saved_dir / "test.txt").write_text("not an image")

        response = client.get("/saved/test.txt")
        assert response.status_code == 400
        assert "PNG" in response.json()["detail"]

    def test_saved_file_success(self, client, temp_dir):
        """Test serving a valid saved PNG."""
        saved_dir = temp_dir / "saved"
        (saved_dir / "test_20240101_0_0.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake png data")

        response = client.get("/saved/test_20240101_0_0.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"



class TestArchiveGalleryEndpoint:
    """Tests for POST /api/gallery/{run_id}/archive."""

    def test_archive_gallery_not_found(self, client):
        """Test 404 for archiving missing gallery."""
        response = client.post("/api/gallery/nonexistent/archive")
        assert response.status_code == 404

    def test_archive_backup_rejected(self, client, temp_dir):
        """Test that archiving a backup is rejected."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {"is_backup": True},
        }))

        response = client.post("/api/gallery/20240101_120000_abc123/archive")
        assert response.status_code == 400
        assert "backup" in response.json()["detail"].lower()


class TestArchiveFileEndpoint:
    """Tests for /archive/{run_id}/{filename:path} endpoint."""

    def test_archive_file_path_traversal_rejected(self, client, temp_dir):
        """Test that path traversal via symlink is rejected with 403.

        Symlinks are used because HTTP clients (requests) normalize '..' segments
        out of URLs before they reach the framework. A symlink inside run_dir
        pointing outside forces resolve() to escape the directory at filesystem level,
        triggering the security guard in routes.py.
        """
        saved_dir = temp_dir / "saved"
        run_id = "20240101_120000_abc123"
        run_dir = saved_dir / run_id
        run_dir.mkdir()

        # Target file outside the archive directory but under saved/
        escape_target = (temp_dir / "escape_file.txt")
        escape_target.write_text("should not be served")

        # Symlink inside the run dir that escapes to the target
        os.symlink(escape_target, run_dir / "link_to_escape")

        response = client.get(f"/archive/{run_id}/link_to_escape")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


class TestGalleryFileEndpoint:
    """Tests for /gallery/{run_id}/{filename:path} endpoint."""

    def test_gallery_file_path_traversal_rejected(self, client, temp_dir):
        """Test that path traversal via symlink is rejected with 403.

        Symlinks are used because HTTP clients (requests) normalize '..' segments
        out of URLs before they reach the framework. A symlink inside run_dir
        pointing outside forces resolve() to escape the directory at filesystem level,
        triggering the security guard in routes.py.
        """
        prompts_dir = temp_dir / "prompts"
        run_id = "20240101_120000_abc123"
        run_dir = prompts_dir / run_id
        run_dir.mkdir()

        # Target file outside the gallery directory but under prompts/
        escape_target = (temp_dir / "escape_file.txt")
        escape_target.write_text("should not be served")

        # Symlink inside the run dir that escapes to the target
        os.symlink(escape_target, run_dir / "link_to_escape")

        response = client.get(f"/gallery/{run_id}/link_to_escape")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


class TestQueueEndpoints:
    """Tests for queue management endpoints."""

    def test_clear_queue(self, client, mock_queue_manager):
        """Test clearing the queue."""
        mock_queue_manager.clear_pending.return_value = 5

        response = client.post("/api/queue/clear")
        assert response.status_code == 200
        assert "5" in response.json()["message"]

    def test_get_status_detailed(self, client, mock_queue_manager):
        """Test getting detailed queue status."""
        mock_queue_manager.get_state.return_value = QueueState(
            pending=[Task(id="1", type=TaskType.GENERATE_PIPELINE), Task(id="2", type=TaskType.GENERATE_PIPELINE)],
            current_task=Task(id="3", type=TaskType.GENERATE_PIPELINE),
            completed=[Task(id="4", type=TaskType.GENERATE_PIPELINE)],
        )
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["queue_length"] == 3
        assert data["pending_count"] == 2
        assert data["completed_count"] == 1


class TestKillWorkerEndpoint:
    """Tests for /api/worker/kill endpoint."""

    def test_kill_worker_success(self, client, mock_worker):
        """Test killing worker when task is running."""
        mock_worker.kill_current = AsyncMock(return_value=True)

        response = client.post("/api/worker/kill")
        assert response.status_code == 200
        assert "Killed" in response.json()["message"]

    def test_kill_worker_no_task(self, client, mock_worker):
        """Test killing worker when no task is running."""
        mock_worker.kill_current = AsyncMock(return_value=False)

        response = client.post("/api/worker/kill")
        assert response.status_code == 200
        assert "No task" in response.json()["message"]

class TestGrammarEndpoints:
    """Tests for /api/generate-from-grammar endpoint."""

    def test_generate_from_grammar_valid(self, client, mock_queue_manager):
        """Test generating from a valid JSON grammar."""
        grammar = '{"item": ["dragon", "unicorn"]}'
        response = client.post("/api/generate-from-grammar", json={
            "grammar": grammar,
            "count": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "Grammar gallery queued" in data["message"]

    def test_generate_from_grammar_invalid_json(self, client):
        """Test that invalid JSON grammar is rejected."""
        response = client.post("/api/generate-from-grammar", json={
            "grammar": "{ invalid json }",
        })
        assert response.status_code == 400
        assert "Invalid JSON grammar" in response.json()["detail"]

    def test_get_grammar_success(self, client, temp_dir):
        """Test retrieving existing grammar from gallery."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 5,
        }))
        grammar_content = '{"origin": ["a cat"], "cat": ["fluffy"]}'
        (run_dir / "test_grammar.json").write_text(grammar_content)

        response = client.get("/api/gallery/20240101_120000_abc123/grammar")
        assert response.status_code == 200
        data = response.json()
        assert "grammar" in data
        assert data["grammar"] == grammar_content

    def test_get_grammar_not_found(self, client, temp_dir):
        """Test 404 when gallery exists but no grammar file."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_def456"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 3,
        }))

        response = client.get("/api/gallery/20240101_120000_def456/grammar")
        assert response.status_code == 404
        assert "Grammar not found" in response.json()["detail"]


class TestRegenerateEndpointInvalidGrammar:
    """Tests for POST /api/gallery/{run_id}/regenerate error paths."""

    def test_regenerate_rejects_invalid_json_grammar(self, client, temp_dir):
        """Test that invalid JSON grammar is rejected with 400."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 5,
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["old"]}))

        response = client.post("/api/gallery/20240101_120000_abc123/regenerate", json={
            "grammar": "not valid json {",
        })

        assert response.status_code == 400
        assert "Invalid JSON grammar" in response.json()["detail"]


class TestGalleryLogsEndpoint:
    """Tests for GET /api/gallery/{run_id}/logs."""

    def test_logs_negative_tail_rejected(self, client, temp_dir):
        """Test that negative tail is rejected with 400."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))

        response = client.get("/api/gallery/20240101_120000_abc123/logs?tail=-5")
        assert response.status_code == 400
        assert "non-negative" in response.json()["detail"].lower()

    def test_logs_no_log_file_found(self, client, temp_dir):
        """Test that missing log file returns empty logs."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))

        response = client.get("/api/gallery/20240101_120000_abc123/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["logs"] == ""
        assert "No log file" in data["message"]

    def test_logs_with_tail(self, client, temp_dir):
        """Test returning a tail of lines from log."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))
        log_content = "line 1\nline 2\nline 3\nline 4\nline 5"
        (run_dir / "test_worker.log").write_text(log_content)

        response = client.get("/api/gallery/20240101_120000_abc123/logs?tail=2")
        assert response.status_code == 200
        data = response.json()
        assert "line 4" in data["logs"]
        assert "line 5" in data["logs"]
        assert "line 1" not in data["logs"]

    def test_logs_zero_tail_returns_all(self, client, temp_dir):
        """Test that tail=0 returns all lines."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))
        log_content = "first\nsecond\nthird\n"
        (run_dir / "test_worker.log").write_text(log_content)

        response = client.get("/api/gallery/20240101_120000_abc123/logs?tail=0")
        assert response.status_code == 200
        data = response.json()
        assert "first" in data["logs"]
        assert "third" in data["logs"]


class TestGrammarHistoryNotFoundEndpoint:
    """Tests for GET /api/gallery/{run_id}/grammar/history error paths."""

    def test_grammar_history_gallery_not_found(self, client):
        """Test 404 when requesting history for a missing gallery."""
        response = client.get("/api/gallery/nonexistent_abc/grammar/history")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestGalleryInfoEndpoint:
    """Tests for GET /api/gallery/{run_id} endpoint."""

    def test_gallery_info_returns_full_response(self, client, temp_dir):
        """Test full gallery info retrieval with all fields."""
        prompts_dir = temp_dir / "prompts"
        run_id = "20240101_120000_testrun"
        run_dir = prompts_dir / run_id
        run_dir.mkdir()

        # Create metadata file
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 5,
            "user_prompt": "a dragon flying over mountains",
        }))

        # Create gallery HTML file (required by _extract_run_info)
        (run_dir / "test_gallery.html").write_text("<html>gallery</html>")

        # Create grammar file
        (run_dir / "test_grammar.json").write_text(json.dumps({
            "origin": ["a {creature}"],
            "creature": ["dragon", "unicorn"]
        }))

        # Create prompt file
        (run_dir / "test_0.txt").write_text("a dragon flying over mountains")

        # Create an image file with proper naming convention
        img_path = run_dir / "test_0_0.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake png data")

        response = client.get(f"/api/gallery/{run_id}")
        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "info" in data
        assert "grammar" in data
        assert "prompts" in data
        assert "images" in data

        # Verify info fields
        info = data["info"]
        assert info["run_id"] == "20240101_120000_testrun"
        assert info["prefix"] == "test"
        assert info["user_prompt"] == "a dragon flying over mountains"
        assert info["prompt_count"] == 5
        assert info["image_count"] >= 1

    def test_gallery_info_returns_404_for_missing(self, client):
        """Test that missing gallery returns 404."""
        response = client.get("/api/gallery/nonexistent_run_id")
        assert response.status_code == 404

    def test_gallery_info_parses_image_metadata(self, client, temp_dir):
        """Test that image files are parsed with correct indices."""
        prompts_dir = temp_dir / "prompts"
        run_id = "20240101_120000_imgtest"
        run_dir = prompts_dir / run_id
        run_dir.mkdir()

        # Use prefix="img" and create matching metadata file
        (run_dir / "img.metaprompt.json").write_text(json.dumps({
            "prefix": "img",
            "count": 3,
        }))

        # Create gallery HTML file (required by _extract_run_info)
        (run_dir / "img_gallery.html").write_text("<html>gallery</html>")

        (run_dir / "img_grammar.json").write_text(json.dumps({"origin": ["test"]}))
        (run_dir / "img_0.txt").write_text("Prompt 1")
        (run_dir / "img_1.txt").write_text("Prompt 2")
        (run_dir / "img_2.txt").write_text("Prompt 3")

        # Create image files with specific naming convention: prefix_promptIdx_imageIdx.png
        (run_dir / "img_0_0.png").write_bytes(b"fake_png_data")
        (run_dir / "img_0_1.png").write_bytes(b"fake_png_data")
        (run_dir / "img_2_0.png").write_bytes(b"fake_png_data")

        response = client.get(f"/api/gallery/{run_id}")
        assert response.status_code == 200
        data = response.json()

        # Verify images are parsed correctly
        images = data["images"]
        assert len(images) == 3

        # Check first image from prompt 0
        img_0_0 = next(i for i in images if i["prompt_idx"] == 0 and i["image_idx"] == 0)
        assert img_0_0["filename"] == "img_0_0.png"
        assert img_0_0["prompt"] == "Prompt 1"

        # Check second image from prompt 2
        img_2_0 = next(i for i in images if i["prompt_idx"] == 2 and i["image_idx"] == 0)
        assert img_2_0["filename"] == "img_2_0.png"
        assert img_2_0["prompt"] == "Prompt 3"

    def test_gallery_info_handles_invalid_filename_format(self, client, temp_dir):
        """Test that files with invalid naming convention are skipped."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_badname"
        run_dir.mkdir()

        (run_dir / "bad.metaprompt.json").write_text(json.dumps({
            "prefix": "bad",
        }))

        # Required gallery HTML file (needed by _extract_run_info)
        (run_dir / "bad_gallery.html").write_text("<html>gallery</html>")

        # Files without proper naming convention should be skipped
        (run_dir / "random_file.png").write_bytes(b"fake_png_data")
        (run_dir / "no_dashes.png").write_bytes(b"fake_png_data")

        response = client.get("/api/gallery/20240101_120000_badname")
        assert response.status_code == 200
        data = response.json()

        # These files should be skipped due to invalid naming format
        images = data["images"]
        assert len(images) == 0
