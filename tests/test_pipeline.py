"""Tests for pipeline executor - the core orchestration layer."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from conftest import create_run_files
from pipeline import (
    PipelineExecutor,
    PipelineResult,
    PipelineConfig,
    ImageGenerationConfig,
    EnhancementConfig,
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


class TestImageGenerationConfig:
    """Tests for ImageGenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ImageGenerationConfig()
        assert config.enabled is False
        assert config.images_per_prompt == 1
        assert config.width == 864
        assert config.height == 1152
        assert config.seed is None
        assert config.tiled_vae is False
        assert config.resume is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ImageGenerationConfig(
            enabled=True,
            width=1024,
            height=1024,
            seed=42,
        )
        assert config.enabled is True
        assert config.width == 1024
        assert config.seed == 42

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ImageGenerationConfig(
            enabled=True,
            width=1024,
            height=768,
        )
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["width"] == 1024
        assert result["height"] == 768
        # Optional fields should not be present if None
        assert "seed" not in result or result.get("seed") is None
        assert "max_prompts" not in result  # None by default

    def test_to_dict_excludes_none_optionals(self):
        """Test that None optional values are excluded from dict."""
        config = ImageGenerationConfig()
        result = config.to_dict()
        assert "seed" not in result  # None by default
        assert "max_prompts" not in result  # None by default


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnhancementConfig()
        assert config.enabled is False
        assert config.softness == 0.5
        assert config.batch_after is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EnhancementConfig(
            enabled=True,
            softness=0.3,
            batch_after=True,
        )
        assert config.enabled is True
        assert config.softness == 0.3
        assert config.batch_after is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EnhancementConfig(enabled=True, softness=0.7)
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["softness"] == 0.7
        assert result["batch_after"] is False


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.prompt == ""
        assert config.count == 50
        assert config.prefix == "image"
        assert config.temperature == 0.7
        assert config.no_cache is False
        assert isinstance(config.image, ImageGenerationConfig)
        assert isinstance(config.enhancement, EnhancementConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            prompt="a dragon flying",
            count=100,
            prefix="dragon",
            image=ImageGenerationConfig(enabled=True, width=1024),
            enhancement=EnhancementConfig(enabled=True),
        )
        assert config.prompt == "a dragon flying"
        assert config.count == 100
        assert config.image.enabled is True
        assert config.image.width == 1024
        assert config.enhancement.enabled is True

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
        """Test full pipeline without image generation.

        Verifies progress callback sequence ordering, metadata structure content,
        and prompt file naming conventions to catch regressions in flow logic
        rather than just the success flag.
        """
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

        # Verify progress callback sequence ordering: grammar -> expansion start -> expansion end
        stages = [s for s, _, _, _ in progress_calls]
        grammar_idx = stages.index("generating_grammar")
        expand_start = stages.index("expanding_prompts")
        expand_end = stages.index("expanding_prompts", expand_start + 1)
        assert expand_start < expand_end  # expansion starts and ends at distinct calls
        assert stages[grammar_idx] == "generating_grammar"

        # Verify progress callback message content for grammar stage
        grammar_messages = [m for s, _, _, m in progress_calls if s == "generating_grammar"]
        assert len(grammar_messages) >= 2  # start + end messages
        assert any("test prompt" in msg for msg in grammar_messages)

        # Verify expansion counts are correct (count=3)
        expand_start_call = next(c for c in progress_calls if c[0] == "expanding_prompts" and c[1] == 0)
        assert expand_start_call[2] == 3  # total should match count

        # Verify prompt files were created with correct naming convention (numbered prompts)
        expected_prompt_files = [result.output_dir / f"test_{i}.txt" for i in range(3)]
        for expected_file in expected_prompt_files:
            assert expected_file.exists(), f"Expected file {expected_file.name} not found"

        # Verify metadata file content structure
        import json as _json
        meta_file = result.output_dir / "test.metaprompt.json"
        with open(meta_file) as f:
            meta_content = _json.load(f)
        assert meta_content["prefix"] == "test"
        assert meta_content["count"] == 3
        assert meta_content["user_prompt"] == "test prompt"
        assert "created_at" in meta_content

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

        # A lexically earlier fallback makes the direct-match contract
        # deterministic instead of depending on filesystem iteration order.
        meta2 = grammar_dir / "aaa_wrong.metaprompt.json"
        meta2.write_text(json.dumps({"user_prompt": "wrong"}))

        meta1 = grammar_dir / "match_me.metaprompt.json"
        meta1.write_text(json.dumps({"user_prompt": "correct"}))

        grammar_file = grammar_dir / "match_me.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        from pipeline import PipelineExecutor
        executor = PipelineExecutor()

        captured_kwargs = {}

        def capture_gallery(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.run_tracery") as mock_tracery:
            mock_tracery.return_value = ["prompt 1"]
            with patch("pipeline.create_gallery", side_effect=capture_gallery):
                with patch("pipeline.generate_master_index"):
                    result = executor.run_from_grammar(grammar_path=grammar_file, count=1)

        assert result.success is True
        assert captured_kwargs["user_prompt"] == "correct"
        # Verify raw_response_file resolves to None when no .txt raw response exists
        assert captured_kwargs.get("raw_response_file") is None

class TestRunFromGrammarText:
    """Tests for run_from_grammar_text method - metadata shape characterization.

    These tests verify structural contracts of the metadata written by
    run_from_grammar_text — fields that downstream consumers depend on.
    Losing any of these silently breaks gallery rendering or index generation.
    """

    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_source_field_always_present(self, mock_index, mock_gallery, mock_tracery, temp_dir):
        """run_from_grammar_text must write a 'source' key to metadata.

        Downstream consumers (e.g., gallery rendering) check the source field
        to decide how to interpret the grammar file. Losing it silently forces
        fallback defaults and loses provenance — users see 'unknown' instead of
        knowing which pipeline step produced their prompts. The default value
        is "from_grammar_text"; a custom source should override it when provided.
        """
        mock_tracery.return_value = ["prompt 1", "prompt 2"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            result = executor.run_from_grammar_text(
                grammar='{"origin": ["test"]}',
                count=2,
                prefix="meta",
                generate_images=False,
            )

        assert result.success is True
        meta_file = result.output_dir / "meta.metaprompt.json"
        with open(meta_file) as f:
            meta_content = json.load(f)
        assert "source" in meta_content
        # Default source when none specified
        assert meta_content["source"] == "from_grammar_text"

    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_image_generation_always_written_to_metadata(self, mock_index, mock_gallery, mock_tracery, temp_dir):
        """image_generation block must be present in metadata even when disabled.

        Gallery rendering reads image_generation settings (width, height) from
        the metadata file to lay out cards. If this key is missing when
        generate_images=False, cards default to incorrect dimensions — breaking
        gallery layout silently. The contract: always write the block; let
        consumers check 'enabled' instead of checking existence.
        """
        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            result = executor.run_from_grammar_text(
                grammar='{"origin": ["test"]}',
                count=1,
                prefix="img_meta",
                generate_images=False,  # explicitly disabled
            )

        assert result.success is True
        meta_file = result.output_dir / "img_meta.metaprompt.json"
        with open(meta_file) as f:
            meta_content = json.load(f)
        assert "image_generation" in meta_content
        img_gen = meta_content["image_generation"]
        # Must contain core settings consumers rely on
        assert "enabled" in img_gen and img_gen["enabled"] is False
        assert "width" in img_gen and "height" in img_gen

    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_custom_source_overrides_default(self, mock_index, mock_gallery, mock_tracery, temp_dir):
        """Custom source parameter must override the default 'from_grammar_text'."""
        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            result = executor.run_from_grammar_text(
                grammar='{"origin": ["test"]}',
                count=1,
                prefix="src",
                source="custom_import",
                generate_images=False,
            )

        assert result.success is True
        meta_file = result.output_dir / "src.metaprompt.json"
        with open(meta_file) as f:
            meta_content = json.load(f)
        assert meta_content["source"] == "custom_import"


class TestRunFromGrammar:
    """Tests for run_from_grammar method."""

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


class TestRunFullPipelineWithImages:
    """Tests for run_full_pipeline when generate_images=True."""

    @patch("pipeline.generate_master_index")
    @patch("pipeline.create_gallery")
    @patch("pipeline.run_tracery")
    @patch("pipeline.generate_grammar")
    def test_generate_images_invokes_internal_generator(
        self, mock_grammar, mock_tracery, mock_gallery, mock_index, temp_dir
    ):
        """Test that generate_images=True triggers _generate_images with correct parameters.

        Verifies the public API correctly routes image generation requests
        through the internal pipeline method without requiring mflux hardware.
        Catches regressions in parameter passing between run_full_pipeline
        and _generate_images.
        """
        mock_grammar.return_value = ('{"origin": ["test"]}', False, "raw response")
        mock_tracery.return_value = ["prompt 1", "prompt 2"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            # Mock _generate_images to capture its call arguments without invoking mflux
            captured_kwargs = {}

            def fake_generate_images(**kwargs):
                captured_kwargs.update(kwargs)
                return PipelineResult(
                    success=True, run_id="test", output_dir=temp_dir,
                    image_count=2, skipped_count=0,
                )

            executor = PipelineExecutor()
            with patch.object(PipelineExecutor, "_generate_images", side_effect=fake_generate_images):
                result = executor.run_full_pipeline(
                    prompt="a dragon flying",
                    count=2,
                    prefix="dragon",
                    generate_images=True,
                    images_per_prompt=1,
                    width=1024,
                    height=768,
                    seed=42,
                )

        assert result.success is True
        assert result.image_count == 2
        assert captured_kwargs["width"] == 1024
        assert captured_kwargs["height"] == 768
        assert captured_kwargs["seed"] == 42
        assert captured_kwargs["images_per_prompt"] == 1
        assert captured_kwargs["tiled_vae"] is True
        assert captured_kwargs["enhance"] is False
        # Verify prompt files were created with numbered naming convention: {prefix}_{i}.txt
        expected_files = [result.output_dir / f"dragon_{i}.txt" for i in range(2)]
        for ef in expected_files:
            assert ef.exists(), f"Expected prompt file {ef.name} not found"

    @patch("pipeline.generate_master_index")
    @patch("pipeline.create_gallery")
    @patch("pipeline.run_tracery")
    @patch("pipeline.generate_grammar")
    def test_max_prompts_write_all_prompt_files_but_limit_gallery(
        self, mock_grammar, mock_tracery, mock_gallery, mock_index, temp_dir
    ):
        """Test that max_prompts limits gallery rendering but writes all prompt files.

        Verifies the 'save all / render limited' contract: when max_prompts is set,
        ALL generated prompts are written to disk (for later regeneration), while
        only the first N prompts are passed to create_gallery for rendering.
        This catches regressions in the save-vs-render split that would otherwise
        silently lose prompt data or over-render.
        """
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.return_value = ["prompt 1", "prompt 2", "prompt 3"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        captured_kwargs = {}
        captured_prompts = []

        def fake_create_gallery(*args, **kwargs):
            # args: (output_dir, prefix, prompts_to_render, images_per_prompt)
            captured_prompts.extend(args[2])  # third positional arg is the prompt list
            captured_kwargs.update(kwargs)
            return temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            with patch("pipeline.create_gallery", side_effect=fake_create_gallery):
                result = executor.run_full_pipeline(
                    prompt="test", count=3, prefix="test",
                    generate_images=False, max_prompts=2,
                )

        assert result.success is True
        assert result.prompt_count == 3
        assert result.image_count == 0

        # All 3 prompt files should be written to disk (save all)
        for i in range(3):
            expected_file = result.output_dir / f"test_{i}.txt"
            assert expected_file.exists(), f"Expected {expected_file.name} not found"

        # Gallery should only receive first 2 prompts (render limited)
        assert len(captured_prompts) == 2
        assert captured_prompts == ["prompt 1", "prompt 2"]

    @patch("pipeline.generate_master_index")
    @patch("pipeline.create_gallery")
    @patch("pipeline.run_tracery")
    @patch("pipeline.generate_grammar")
    def test_generate_images_passes_enhancement_params(
        self, mock_grammar, mock_tracery, mock_gallery, mock_index, temp_dir
    ):
        """Test that enhancement flags flow through to _generate_images."""
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        captured_kwargs = {}

        def fake_generate_images(**kwargs):
            captured_kwargs.update(kwargs)
            return PipelineResult(success=True, run_id="test", output_dir=temp_dir, image_count=1, skipped_count=0)

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            with patch.object(PipelineExecutor, "_generate_images", side_effect=fake_generate_images):
                result = executor.run_full_pipeline(
                    prompt="test", count=1, generate_images=True,
                    enhance=True, enhance_softness=0.75, enhance_after=True,
                )

        assert result.success is True
        assert captured_kwargs["enhance"] is True
        assert captured_kwargs["enhance_softness"] == 0.75
        assert captured_kwargs["enhance_after"] is True

    @patch("pipeline.generate_master_index")
    @patch("pipeline.create_gallery")
    @patch("pipeline.run_tracery")
    @patch("pipeline.generate_grammar")
    def test_generate_images_respect_max_prompts_limit(
        self, mock_grammar, mock_tracery, mock_gallery, mock_index, temp_dir
    ):
        """Test that max_prompts correctly limits the prompt list passed to _generate_images."""
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.return_value = [f"prompt {i}" for i in range(5)]  # 5 prompts generated
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        captured_kwargs = {}

        def fake_generate_images(**kwargs):
            captured_kwargs.update(kwargs)
            return PipelineResult(success=True, run_id="test", output_dir=temp_dir, image_count=2, skipped_count=0)

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            with patch.object(PipelineExecutor, "_generate_images", side_effect=fake_generate_images):
                result = executor.run_full_pipeline(
                    prompt="test", count=5, generate_images=True, max_prompts=2,
                )

        assert result.success is True
        assert len(captured_kwargs["prompts"]) == 2  # Only first 2 prompts passed to image gen

    @patch("pipeline.generate_master_index")
    @patch("pipeline.create_gallery")
    @patch("pipeline.run_tracery")
    @patch("pipeline.generate_grammar")
    def test_generate_images_passes_resume_param(
        self, mock_grammar, mock_tracery, mock_gallery, mock_index, temp_dir
    ):
        """Test that resume flag flows through to _generate_images."""
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        captured_kwargs = {}

        def fake_generate_images(**kwargs):
            captured_kwargs.update(kwargs)
            return PipelineResult(success=True, run_id="test", output_dir=temp_dir, image_count=1, skipped_count=0)

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            with patch.object(PipelineExecutor, "_generate_images", side_effect=fake_generate_images):
                result = executor.run_full_pipeline(
                    prompt="test", count=1, generate_images=True, resume=True,
                )

        assert result.success is True
        assert captured_kwargs["resume"] is True

    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_run_full_pipeline_passes_gallery_and_index_arguments(
        self, mock_index, mock_gallery, temp_dir
    ):
        """Test that gallery and master index receive correct arguments from pipeline.

        Verifies create_gallery receives the full prompt list (or max_prompts slice),
        grammar string, raw response reference, and user display title; also verifies
        generate_master_index is called with the resolved generated directory path.
        Catches regressions where gallery content or index location drifts from pipeline state.
        """
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.generate_grammar") as mock_grammar, \
             patch("pipeline.run_tracery") as mock_tracery:
            mock_grammar.return_value = ('{"origin": ["test"]}', False, "raw response")
            mock_tracery.return_value = [f"prompt {i}" for i in range(3)]

            with patch("pipeline.paths") as mock_paths:
                mock_paths.prompts_dir = temp_dir / "prompts"
                mock_paths.prompts_dir.mkdir(parents=True)
                mock_paths.generated_dir = temp_dir / "generated"

                executor = PipelineExecutor()
                result = executor.run_full_pipeline(
                    prompt="a dragon flying",
                    count=3,
                    prefix="dragon",
                    generate_images=False,
                    max_prompts=2,
                )

        assert result.success is True
        # Gallery should receive only the first max_prompts prompts (not all 3)
        gallery_call = mock_gallery.call_args
        positional = gallery_call[0]
        assert len(positional) >= 4
        assert positional[1] == "dragon"  # prefix
        assert list(positional[2]) == [f"prompt {i}" for i in range(2)]  # prompts slice
        assert positional[3] == 1  # images_per_prompt default

        keyword = gallery_call[1]
        assert keyword["grammar"] == '{"origin": ["test"]}'
        assert keyword["raw_response_file"] == "dragon_raw_response.txt"
        assert keyword["user_prompt"] == "a dragon flying"

        # Master index should be called with resolved generated_dir
        mock_index.assert_called_once_with(temp_dir / "generated")
