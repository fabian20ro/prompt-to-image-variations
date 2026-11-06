"""Tests for cli.py - CLI entry point."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cli import main, clean_generated, cli_progress


class TestCleanGenerated:
    """Tests for the clean_generated function."""

    def test_clean_removes_files(self, temp_dir):
        """Test that clean removes generated files."""
        grammars = temp_dir / "grammars"
        prompts = temp_dir / "prompts"
        grammars.mkdir()
        prompts.mkdir()

        (grammars / "test.json").write_text("{}")
        (prompts / "run1").mkdir()
        (prompts / "run1" / "test.txt").write_text("test")

        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = grammars
            mock_paths.prompts_dir = prompts
            count = clean_generated()

        assert count == 2  # 1 file + 1 directory

    def test_clean_empty_dirs(self, temp_dir):
        """Test clean with empty directories."""
        grammars = temp_dir / "grammars"
        prompts = temp_dir / "prompts"
        grammars.mkdir()
        prompts.mkdir()

        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = grammars
            mock_paths.prompts_dir = prompts
            count = clean_generated()

        assert count == 0

    def test_clean_nonexistent_dirs(self, temp_dir):
        """Test clean when dirs don't exist."""
        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = temp_dir / "nonexistent1"
            mock_paths.prompts_dir = temp_dir / "nonexistent2"
            count = clean_generated()

        assert count == 0


class TestCliProgress:
    """Tests for the cli_progress callback."""

    def test_progress_with_message(self, capsys):
        """Test progress with a message."""
        cli_progress("stage", 1, 10, "Working...")
        captured = capsys.readouterr()
        assert "[1/10] Working..." in captured.out

    def test_progress_without_counts(self, capsys):
        """Test progress with just a message."""
        cli_progress("stage", 0, 0, "Starting...")
        captured = capsys.readouterr()
        assert "Starting..." in captured.out

    def test_progress_empty_message(self, capsys):
        """Test progress with empty message is silent."""
        cli_progress("stage", 1, 10, "")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestCliCleanCommand:
    """Tests for --clean flag."""

    def test_clean_flag(self):
        """Test --clean removes generated files."""
        runner = CliRunner()
        with patch("cli.clean_generated", return_value=5) as mock_clean:
            result = runner.invoke(main, ["--clean"])
            assert result.exit_code == 0
            assert "Cleaned 5 items" in result.output
            mock_clean.assert_called_once()


class TestCliValidation:
    """Tests for CLI argument validation."""

    def test_no_args_shows_error(self):
        """Test that no arguments shows error."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_from_grammar_and_from_prompts_conflict(self, temp_dir):
        """Test conflicting flags."""
        grammar_file = temp_dir / "test.json"
        grammar_file.write_text('{"origin": ["test"]}')
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            "--from-grammar", str(grammar_file),
            "--from-prompts", str(prompts_dir),
        ])
        assert result.exit_code != 0
        assert "Cannot use both" in result.output

    def test_from_prompts_requires_generate_images(self, temp_dir):
        """Test --from-prompts requires --generate-images."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, ["--from-prompts", str(prompts_dir)])
        assert result.exit_code != 0
        assert "--generate-images" in result.output

    def test_count_must_be_positive(self):
        """Test --count rejects zero/negative prompt counts before pipeline execution."""
        runner = CliRunner()
        result = runner.invoke(main, ["--prompt", "a cat", "--count", "0"])
        assert result.exit_code != 0
        assert "Invalid value for '-n' / '--count'" in result.output
        assert "0 is not in the range x>=1" in result.output

    def test_images_per_prompt_must_be_non_negative(self):
        """Test --images-per-prompt rejects negative values."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "--prompt", "a cat",
            "--generate-images",
            "--images-per-prompt", "-1",
        ])
        assert result.exit_code != 0
        assert "Invalid value for '--images-per-prompt'" in result.output
        assert "-1 is not in the range x>=0" in result.output

    def test_temperature_out_of_range(self):
        """Test --temperature rejects values outside [0, 2]."""
        runner = CliRunner()
        # Test too high
        result_high = runner.invoke(main, ["--prompt", "a cat", "--temperature", "2.5"])
        assert result_high.exit_code != 0
        assert "temperature must be between 0.0 and 2.0" in result_high.output

        # Test too low
        result_low = runner.invoke(main, ["--prompt", "a cat", "--temperature", "-0.1"])
        assert result_low.exit_code != 0
        assert "temperature must be between 0.0 and 2.0" in result_low.output

    def test_help_documents_prompt_only_layout(self):

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "0 = prompt-only layout" in result.output
        assert "Standalone: enhance existing images" in result.output
        assert "LM Studio API base URL (default:" in result.output
        assert "http://localhost:1234/v1" in result.output


class TestCliDryRun:
    """Tests for --dry-run behavior."""

    @patch("cli.generate_grammar")
    @patch("cli.PipelineExecutor")
    def test_dry_run_previews_grammar_without_pipeline(self, mock_executor_cls, mock_generate_grammar):
        """Test --dry-run prints grammar and skips the full pipeline."""
        mock_generate_grammar.return_value = ('{"origin": ["a cat"]}', False, None)

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "--dry-run"])

        assert result.exit_code == 0
        assert "--- Generated Grammar ---" in result.output
        assert '{"origin": ["a cat"]}' in result.output
        mock_generate_grammar.assert_called_once_with(
            user_prompt="a cat",
            base_url="http://localhost:1234/v1",
            use_cache=True,
            temperature=0.7,
            width=864,
            height=1152,
        )
        mock_executor_cls.assert_not_called()


class TestCliServe:
    """Tests for --serve flag."""

    def test_serve_starts_server(self):
        """Test that --serve starts the web server."""
        runner = CliRunner()

        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        mock_webbrowser = MagicMock()

        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self.target = target
                self.daemon = daemon

            def start(self):
                if self.target is not None:
                    self.target()

        with patch.dict("sys.modules", {
            "uvicorn": mock_uvicorn,
            "server.app": MagicMock(app=mock_app),
        }), patch("threading.Thread", ImmediateThread), patch("webbrowser.open", mock_webbrowser):
            result = runner.invoke(main, ["--serve", "--port", "9999"])

        assert "Starting web UI server" in result.output
        mock_webbrowser.assert_called_once_with("http://localhost:9999")
        mock_uvicorn.run.assert_called_once_with(mock_app, host="0.0.0.0", port=9999, log_level="info")


class TestCliFullPipeline:
    """Tests for full pipeline execution via CLI."""

    @patch("cli.PipelineExecutor")
    def test_prompt_generates_outputs(self, mock_executor_cls):
        """Test that -p runs the full pipeline."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 5
        mock_result.image_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "-n", "5"])

        assert result.exit_code == 0
        mock_executor.run_full_pipeline.assert_called_once()

    @patch("cli.PipelineExecutor")
    def test_prompt_with_dimensions(self, mock_executor_cls):
        """Test that --width and --height are passed correctly via CLI."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 5
        mock_result.image_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, [
            "-p", "a cat", 
            "-n", "5", 
            "--width", "1024", 
            "--height", "768"
        ])

        assert result.exit_code == 0
        mock_executor.run_full_pipeline.assert_called_once()
        args, kwargs = mock_executor.run_full_pipeline.call_args
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 768
