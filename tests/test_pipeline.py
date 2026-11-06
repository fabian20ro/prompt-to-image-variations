"""Tests for pipeline executor - the core orchestration layer."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from conftest import create_run_files
from pipeline import (
    PipelineExecutor,
    PipelineResult,
)


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_result(self):
        result = PipelineResult(
            success=True,
            run_id="test_run",
            output_dir=Path("/tmp/test"),
            prompt_count=10,
            image_count=5,
        )
        assert result.success is True
        assert result.run_id == "test_run"
        assert result.error is None

    def test_failed_result(self):
        result = PipelineResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.run_id is None



class TestPipelineExecutorInit:
    """Tests for PipelineExecutor initialization."""

    def test_init_with_callbacks(self):
        progress_cb = MagicMock()
        image_cb = MagicMock()
        executor = PipelineExecutor(on_progress=progress_cb, on_image_ready=image_cb)
        assert executor.on_progress == progress_cb
        assert executor.on_image_ready == image_cb

    def test_init_without_callbacks(self):
        executor = PipelineExecutor()
        # Should use null callbacks
        executor.on_progress("test", 0, 0, "message")  # Should not raise
        executor.on_image_ready("run_id", "filename")  # Should not raise


class TestRunFullPipeline:
    """Tests for run_full_pipeline method."""

    @patch("pipeline.generate_grammar")
    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_run_full_pipeline_prompts_only(
        self, mock_index, mock_gallery, mock_tracery, mock_grammar, temp_dir
    ):
        """Test full pipeline without image generation."""
        mock_grammar.return_value = ('{"origin": ["test"]}', False, "raw response")
        mock_tracery.return_value = ["prompt 1", "prompt 2", "prompt 3"]
        mock_gallery.return_value = temp_dir / "test_gallery.html"

        # Mock paths
        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            progress_calls = []
            executor = PipelineExecutor(
                on_progress=lambda s, c, t, m: progress_calls.append((s, c, t, m))
            )

            result = executor.run_full_pipeline(
                prompt="test prompt",
                count=3,
                prefix="test",
                generate_images=False,
            )

        assert result.success is True
        assert result.prompt_count == 3
        assert result.image_count == 0
        assert result.output_dir is not None

        # Check progress was reported
        assert any(s == "generating_grammar" for s, _, _, _ in progress_calls)
        assert any(s == "expanding_prompts" for s, _, _, _ in progress_calls)

        # Check files were created
        assert result.output_dir.exists()
        assert (result.output_dir / "test.metaprompt.json").exists()

    @patch("pipeline.generate_grammar")
    def test_run_full_pipeline_grammar_failure(self, mock_grammar, temp_dir):
        """Test handling of grammar generation failure."""
        from tracery_runner import TraceryError
        mock_grammar.side_effect = Exception("LM Studio not available")

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"

            executor = PipelineExecutor()
            result = executor.run_full_pipeline(prompt="test", count=1)

        assert result.success is False
        assert "Grammar generation failed" in result.error

    @patch("pipeline.generate_grammar")
    @patch("pipeline.run_tracery")
    def test_run_full_pipeline_tracery_failure(self, mock_tracery, mock_grammar, temp_dir):
        """Test handling of Tracery expansion failure."""
        from tracery_runner import TraceryError
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.side_effect = TraceryError("Invalid grammar")

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"

            executor = PipelineExecutor()
            result = executor.run_full_pipeline(prompt="test", count=1)

        assert result.success is False
        assert "Tracery expansion failed" in result.error


class TestRunFromGrammar:
    """Tests for run_from_grammar method."""

    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_run_from_grammar_success(
        self, mock_index, mock_gallery, mock_tracery, temp_dir
    ):
        """Test running pipeline from existing grammar file."""
        grammar_dir = temp_dir / "grammars"
        grammar_dir.mkdir()
        grammar_file = grammar_dir / "test.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            result = executor.run_from_grammar(grammar_path=grammar_file, count=1)

        assert result.success is True
        assert result.prompt_count == 1
        assert result.output_dir is not None

    def test_run_from_grammar_metadata_match(self, temp_dir):
        """Test that run_from_grammar picks the metadata matching the grammar filename."""
        grammar_dir = temp_dir / "grammars"
        grammar_dir.mkdir()

        meta1 = grammar_dir / "match_me.metaprompt.json"
        meta1.write_text(json.dumps({"user_prompt": "correct"}))

        meta2 = grammar_dir / "wrong.metaprompt.json"
        meta2.write_text(json.dumps({"user_prompt": "wrong"}))

        grammar_file = grammar_dir / "match_me.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        from pipeline import PipelineExecutor
        executor = PipelineExecutor()

        with patch("pipeline.run_tracery") as mock_tracery:
            mock_tracery.return_value = ["prompt 1"]
            with patch("pipeline.create_gallery"):
                with patch("pipeline.generate_master_index"):
                    result = executor.run_from_grammar(grammar_path=grammar_file, count=1)
                    assert result.success is True

    def test_run_from_grammar_metadata_mismatch_fallback(self, temp_dir):
        """Test fallback to any metaprompt in the same directory if no direct match."""
        grammar_dir = temp_dir / "grammars_fallback"
        grammar_dir.mkdir()

        meta1 = grammar_dir / "some_other_file.metaprompt.json"
        meta1.write_text(json.dumps({"user_prompt": "fallback_user"}))

        grammar_file = grammar_dir / "target.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        from pipeline import PipelineExecutor
        executor = PipelineExecutor()

        with patch("pipeline.run_tracery") as mock_tracery:
            mock_tracery.return_value = ["prompt 1"]
            with patch("pipeline.create_gallery"):
                with patch("pipeline.generate_master_index"):
                    result = executor.run_from_grammar(grammar_path=grammar_file, count=1)
                    assert result.success is True
                    assert result.output_dir is not None
