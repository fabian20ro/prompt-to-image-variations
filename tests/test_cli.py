"""Tests for cli.py - CLI entry point."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from config import settings
from cli import main, clean_generated, cli_progress, _status_echo


class TestStatusEcho:
    """Tests for the _status_echo helper (info messages go to stderr)."""

    def test_status_echo_to_stderr(self, capsys):
        """Test that info messages are written to stderr."""
        _status_echo("status message")
        captured = capsys.readouterr()
        assert "status message" in captured.err
        assert captured.out == ""


class TestCleanGenerated:

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

    def test_clean_empty_nothing_to_print(self):
        """Test --clean with no files prints a friendly message."""
        runner = CliRunner()
        with patch("cli.clean_generated", return_value=0) as mock_clean:
            result = runner.invoke(main, ["--clean"])
            assert result.exit_code == 0
            assert "Nothing to clean." in result.output
            assert "Cleaned 0 items" not in result.output
            mock_clean.assert_called_once()

    def test_clean_quiet_no_output(self):
        """Test --clean with --quiet suppresses all output."""
        runner = CliRunner()
        with patch("cli.clean_generated", return_value=3) as mock_clean:
            result = runner.invoke(main, ["--clean", "--quiet"])
            assert result.exit_code == 0
            assert "Cleaned" not in result.output
            assert "Nothing to clean." not in result.output


class TestCliValidation:
    """Tests for CLI argument validation."""

    def test_no_args_shows_error(self):
        """Test that no arguments shows error."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 1
        assert "Error: --prompt is required" in result.output

    @patch("cli.check_lm_studio", return_value=True)
    def test_version_check_success(self, mock_check):
        """Test --version-check prints success and exits 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version-check"])
        assert result.exit_code == 0
        assert "Successfully connected to LM Studio" in result.output
        mock_check.assert_called_once_with("http://localhost:1234/v1")

    @patch("cli.check_lm_studio", return_value=False)
    def test_version_check_failure(self, mock_check):
        """Test --version-check prints error and exits 1."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version-check"])
        assert result.exit_code == 1
        assert "Error: LM Studio is not reachable" in result.output

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

    def test_prompt_ignored_with_from_grammar(self, temp_dir):
        """Test that --prompt emits a warning (not an error) when used with --from-grammar."""
        grammar_file = temp_dir / "test.json"
        grammar_file.write_text('{"origin": ["a cat"]}')

        runner = CliRunner()
        result = runner.invoke(main, [
            "-p", "ignored prompt",
            "--from-grammar", str(grammar_file),
            "--dry-run",
        ])

        assert "Warning: --prompt is ignored when using --from-grammar" in result.output
        # Dry-run continues with the grammar file and exits 0
        assert result.exit_code == 0
        assert "--- Loaded Grammar ---" in result.output

    def test_prompt_ignored_with_from_prompts_quiet(self, temp_dir):
        """Test that --quiet suppresses the --prompt-ignored warning."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            "-p", "ignored prompt",
            "--from-prompts", str(prompts_dir),
            "--generate-images",
            "--quiet",
        ])

        assert "Warning" not in result.output
        assert "--prompt is ignored" not in result.output


class TestCliEnhanceImages:
    """Tests for --enhance-images flag."""

    @patch("image_enhancer.collect_images")
    def test_enhance_images_error(self, mock_collect):
        """Test --enhance-images with an invalid path raises a user-facing error."""
        from image_enhancer import collect_images

        mock_collect.side_effect = ValueError(
            "No images found matching pattern: /nonexistent/path"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["--enhance-images", "/nonexistent/path"])
        assert result.exit_code != 0
        assert (
            'Error: No images found matching pattern: /nonexistent/path' in result.output
        )

    def test_dry_run_from_grammar(self, temp_dir):
        """Test --dry-run with an existing grammar file."""
        grammar_file = temp_dir / "test.json"
        grammar_content = '{"origin": ["test_grammar"]}'
        grammar_file.write_text(grammar_content)

        runner = CliRunner()
        result = runner.invoke(main, ["--from-grammar", str(grammar_file), "--dry-run"])

        assert result.exit_code == 0
        assert f"Loading grammar from: {grammar_file}" in result.output
        assert "--- Loaded Grammar ---" in result.output
        assert grammar_content in result.output
        assert "--- End Grammar ---" in result.output

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
        )
        mock_executor_cls.assert_not_called()

    @patch("cli.generate_grammar")
    def test_dry_run_grammar_failure(self, mock_generate_grammar):
        """Test --dry-run surfaces grammar errors with helpful message and exits 1."""

        error_msg = "LM Studio connection refused"
        mock_generate_grammar.side_effect = ConnectionError(error_msg)

        runner = CliRunner()
        result = runner.invoke(
            main, ["-p", "a cat", "--dry-run"], catch_exceptions=False
        )

        assert result.exit_code == 1
        assert f"Error generating grammar: {error_msg}" in result.output
        assert "Make sure LM Studio is running at http://localhost:1234/v1" in result.output

    @patch("cli.generate_grammar")
    def test_dry_run_keyboard_interrupt(self, mock_generate_grammar):
        """Test --dry-run propagates Ctrl+C with exit 130 and 'Interrupted.' message."""

        mock_generate_grammar.side_effect = KeyboardInterrupt()

        runner = CliRunner()
        result = runner.invoke(
            main, ["-p", "a cat", "--dry-run"], catch_exceptions=False
        )

        assert result.exit_code == 130
        assert "Interrupted." in result.output
        # Should NOT show the LM Studio error message
        assert "Make sure LM Studio is running" not in result.output

    @patch("cli.generate_grammar")
    def test_dry_run_non_connectivity_error_no_lm_hint(self, mock_generate_grammar):
        """Test non-connectivity errors (e.g. JSON parse failures) do NOT show the LM Studio hint."""

        error_msg = "Grammar is not valid JSON"
        mock_generate_grammar.side_effect = ValueError(error_msg)

        runner = CliRunner()
        result = runner.invoke(
            main, ["-p", "a cat", "--dry-run"], catch_exceptions=False
        )

        assert result.exit_code == 1
        assert f"Error generating grammar: {error_msg}" in result.output
        # Should NOT show the LM Studio hint — this is not a connectivity issue
        assert "Make sure LM Studio is running" not in result.output

    @patch("cli.generate_grammar")
    def test_dry_run_timeout_error_shows_lm_hint(self, mock_generate_grammar):
        """Test TimeoutError surfaces with the LM Studio hint."""

        error_msg = "Connection timed out after 30s"
        mock_generate_grammar.side_effect = TimeoutError(error_msg)

        runner = CliRunner()
        result = runner.invoke(
            main, ["-p", "a cat", "--dry-run"], catch_exceptions=False
        )

        assert result.exit_code == 1
        assert f"Error generating grammar: {error_msg}" in result.output
        assert "Make sure LM Studio is running at http://localhost:1234/v1" in result.output

    @patch("cli.generate_grammar")
    def test_dry_run_cached_grammar_message(self, mock_generate_grammar):
        """Test --dry-run prints 'Using cached grammar' when grammar was served from cache."""
        mock_generate_grammar.return_value = ('{"origin": ["cached_cat"]}', True, None)

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "--dry-run"])

        assert result.exit_code == 0
        assert "Using cached grammar" in result.output
        assert '{"origin": ["cached_cat"]}' in result.output

    def test_enhance_images_success(self, tmp_path):
        """Test --enhance-images invokes enhancement for each image and prints summary."""
        from pathlib import Path

        fake_img1 = tmp_path / "test_0.png"
        fake_img2 = tmp_path / "test_1.png"
        fake_img1.touch()
        fake_img2.touch()

        class MockProgressBar:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self):
                for item in [fake_img1, fake_img2]: yield item
            def write(self, s): pass

        with patch("cli.collect_images", return_value=[fake_img1, fake_img2]), \
             patch("cli.enhance_image") as mock_enhance, \
             patch("click.progressbar", MockProgressBar):
            runner = CliRunner()
            result = runner.invoke(main, ["--enhance-images", "/some/path"])

            assert result.exit_code == 0
            assert "Collecting images from: /some/path" in result.output
            assert "Enhanced 2 images" in result.output
            assert mock_enhance.call_count == 2
            # Seed is None by default, so no seed increment expected
            for call in mock_enhance.call_args_list:
                kwargs = call.kwargs if hasattr(call, 'kwargs') else call[1]
                assert kwargs.get("seed") is None

    def test_enhance_images_seed_increments(self, tmp_path):
        """Test --enhance-images increments seed per image when provided."""
        from pathlib import Path

        fake_img = tmp_path / "test.png"
        fake_img.touch()

        class MockProgressBar:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self):
                for item in [fake_img] * 3: yield item
            def write(self, s): pass

        with patch("cli.collect_images", return_value=[fake_img] * 3), \
             patch("cli.enhance_image") as mock_enhance, \
             patch("click.progressbar", MockProgressBar):
            runner = CliRunner()
            result = runner.invoke(main, ["--enhance-images", "/some/path", "--seed", "42"])

            assert result.exit_code == 0
            # Verify seed increments across calls: 42, 43, 44
            seeds = [call.kwargs.get("seed") for call in mock_enhance.call_args_list]
            assert seeds == [42, 43, 44]


class TestStdinPrompt:
    """Tests for --prompt - (read prompt from stdin)."""

    def test_stdin_prompt_reads_from_stdin(self, monkeypatch):
        """Test that '--prompt -' reads the prompt from stdin."""
        runner = CliRunner()

        # Mock generate_grammar to avoid needing LM Studio
        with patch("cli.generate_grammar", return_value=('{\"origin":["test"]}', False, None)):
            result = runner.invoke(
                main, ["--prompt", "-", "--dry-run"], input="a cat from stdin\n"
            )

        assert result.exit_code == 0
        assert "a cat from stdin" in result.output
        assert "{origin: [\"test\"]}" not in result.output  # Just verifying it got the prompt text through

    def test_stdin_empty_prompt_errors(self, monkeypatch):
        """Test that empty stdin gives an error."""
        runner = CliRunner()
        result = runner.invoke(main, ["--prompt", "-"], input="")
        assert "empty prompt" in result.output.lower() or "Error" in result.output

    def test_stdin_with_full_pipeline(self, monkeypatch):
        """Test that stdin works with the full pipeline."""
        runner = CliRunner()

        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 1
        mock_result.image_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        monkeypatch.setattr("cli.PipelineExecutor", lambda **kw: mock_executor)

        result = runner.invoke(
            main, ["--prompt", "-"], input="a prompt from stdin\n"
        )

        assert result.exit_code == 0
        call_args = mock_executor.run_full_pipeline.call_args
        assert call_args.kwargs["prompt"] == "a prompt from stdin"


class TestCliDryRunValidation:
    """Tests for dry-run argument validation."""

    def test_dry_run_without_prompt_or_grammar_exits_1(self):
        """Test --dry-run without --prompt or --from-grammar hits the dry-run-specific error message."""
        runner = CliRunner()
        result = runner.invoke(main, ["--dry-run"])

        assert result.exit_code == 1
        # The specific _run_dry_run branch (line 509-511) prints its own error, not generic validation.
        assert "Error: --prompt is required for --dry-run" in result.output


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

    @patch("cli.PipelineExecutor")
    def test_pipeline_failure_exits_1(self, mock_executor_cls):
        """Test that a failed pipeline result exits with code 1 and prints the error."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Grammar generation timed out"

        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "-n", "5"])

        assert result.exit_code == 1
        assert "Error: Grammar generation timed out" in result.output
        # No summary should be printed on failure
        assert "Generated" not in result.output
        mock_executor.run_full_pipeline.assert_called_once()

    @patch("cli.PipelineExecutor")
    def test_from_grammar_calls_run_from_grammar(self, mock_executor_cls, tmp_path):
        """Test that --from-grammar invokes run_from_grammar with correct args."""
        import json as _json

        grammar_file = tmp_path / "test_grammar.json"
        grammar_file.write_text(_json.dumps({"origin": ["test"]}))

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/out")
        mock_result.prompt_count = 10
        mock_result.image_count = 5
        mock_result.skipped_count = 0
        mock_result.error = None

        mock_executor_cls.return_value.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value.run_from_grammar.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(
            main, ["--from-grammar", str(grammar_file), "-n", "10"]
        )

        assert result.exit_code == 0
        mock_executor_cls.assert_called_once_with(on_progress=cli_progress)
        call_kwargs = mock_executor_cls.return_value.run_from_grammar.call_args.kwargs
        assert call_kwargs["grammar_path"] == grammar_file
        assert call_kwargs["count"] == 10

    @patch("cli.PipelineExecutor")
    def test_from_prompts_calls_run_from_prompts(self, mock_executor_cls, tmp_path):
        """Test that --from-prompts + --generate-images invokes run_from_prompts."""
        prompts_dir = tmp_path

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/out")
        mock_result.prompt_count = 5
        mock_result.image_count = 5
        mock_result.skipped_count = 0
        mock_result.error = None

        mock_executor_cls.return_value.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value.run_from_prompts.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--from-prompts", str(prompts_dir),
                "--generate-images",
                "--images-per-prompt", "2",
                "--width", "1024",
                "--height", "768",
            ],
        )

        assert result.exit_code == 0
        mock_executor_cls.assert_called_once_with(on_progress=cli_progress)
        call_kwargs = mock_executor_cls.return_value.run_from_prompts.call_args.kwargs
        assert call_kwargs["prompts_dir"] == prompts_dir
        assert call_kwargs["images_per_prompt"] == 2
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 768

    @patch("cli.PipelineExecutor")
    def test_default_prefix_and_dimensions(self, mock_executor_cls):
        """"Test that --prompt without explicit prefix/width/height passes CLI defaults to the executor."""
        from config import settings

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 50
        mock_result.image_count = 0
        mock_result.skipped_count = 0
        mock_result.error = None

        mock_executor_cls.return_value.run_full_pipeline.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat"])

        assert result.exit_code == 0
        call_kwargs = mock_executor_cls.return_value.run_full_pipeline.call_args.kwargs
        # CLI default: prefix = prefix or "image"
        assert call_kwargs["prefix"] == "image"
        # Width/height fall back to config defaults (not from_env overrides)
        assert call_kwargs["width"] == settings.image_generation.default_width
        assert call_kwargs["height"] == settings.image_generation.default_height
        assert call_kwargs["seed"] is None
        assert call_kwargs["max_prompts"] is None

    @patch("cli.PipelineExecutor")
    def test_skipped_images_message_when_both_counts_positive(self, mock_executor_cls):
        """Test that CLI prints 'skipped N existing' when image_count and skipped_count are both > 0."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 10
        mock_result.image_count = 8
        mock_result.skipped_count = 2
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "-n", "10", "--generate-images"])

        assert result.exit_code == 0
        assert "Generated 8 images, skipped 2 existing" in result.output

    @patch("cli.PipelineExecutor")
    def test_no_skipped_message_when_all_images_new(self, mock_executor_cls):
        """Test that CLI does NOT print 'skipped N existing' when all images are freshly generated."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 10
        mock_result.image_count = 10
        mock_result.skipped_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "-n", "10", "--generate-images"])

        assert result.exit_code == 0
        assert "skipped" not in result.output
        assert "Generated 10 images" in result.output

    @patch("cli.PipelineExecutor")
    def test_json_flag_outputs_valid_summary(self, mock_executor_cls):
        """Test that --json prints a valid JSON summary with expected keys.

        The CLI emits status echoes (e.g. "Generating grammar for: ...") to stdout
        before the JSON summary; extract JSON from the first brace onward.
        """
        import json as _json

        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 50
        mock_result.image_count = 10
        mock_result.skipped_count = 2
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "--json"])

        assert result.exit_code == 0
        # Extract JSON from the first '{' character (status messages precede it)
        json_start = result.output.find("{")
        assert json_start >= 0, f"No JSON found in output:\n{result.output}"

        summary = _json.loads(result.output[json_start:])
        assert summary["prompt_count"] == 50
        assert summary["image_count"] == 10
        assert summary["skipped_count"] == 2
        assert summary["success"] is True
        assert "output_dir" in summary

    @patch("cli.PipelineExecutor")
    def test_json_output_nulls_output_dir(self, mock_executor_cls):
        """Test that --json serializes null output_dir as JSON null (no crash)."""
        import json as _json

        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = None  # Unset output dir
        mock_result.prompt_count = 30
        mock_result.image_count = 5
        mock_result.skipped_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "--json"])

        assert result.exit_code == 0
        json_start = result.output.find("{")
        summary = _json.loads(result.output[json_start:])
        assert summary["output_dir"] is None


class TestDumpConfigFlag:
    """Tests for --dump-config flag."""

    def test_dump_config_exits_0(self):
        """Test that --dump-config exits cleanly with code 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["--dump-config"])
        assert result.exit_code == 0
        assert "ERNIE-Image-Turbo CLI Settings" in result.output

    def test_dump_config_shows_all_sections(self):
        """Test that --dump-config prints all four config sections."""
        runner = CliRunner()
        result = runner.invoke(main, ["--dump-config"])
        assert "[LM Studio]" in result.output
        assert "[Image Generation]" in result.output
        assert "[Server]" in result.output
        assert "[Enhancement]" in result.output

    def test_dump_config_shows_field_names(self):
        """Test that --dump-config prints field names for each section."""
        runner = CliRunner()
        result = runner.invoke(main, ["--dump-config"])
        # LM Studio fields
        assert "base_url" in result.output
        assert "model" in result.output
        assert "timeout" in result.output
        # Image Generation fields
        assert "default_width" in result.output
        assert "default_height" in result.output
        assert "seed" in result.output

    def test_dump_config_with_other_flags_no_pipeline(self, capsys):
        """Test that --dump-config does not run the pipeline even with other flags."""
        runner = CliRunner()
        # Pass several other flags alongside --dump-config
        result = runner.invoke(main, [
            "--dump-config", "-p", "a cat", "-n", "5"
        ])
        assert result.exit_code == 0
        assert "ERNIE-Image-Turbo CLI Settings" in result.output
        # No pipeline output should appear
        assert "Generating grammar for:" not in result.output

    def test_dump_config_help_mentions_flag(self):
        """Test that --help documents the --dump-config flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--dump-config" in result.output
        assert "effective settings" in result.output or "settings" in result.output
