"""Tests for centralized configuration."""

import os
from pathlib import Path

import pytest
from unittest.mock import patch

from config import Settings, LMStudioConfig, ImageGenerationConfig, ServerConfig, paths


class TestConfig:
    """Tests for centralized configuration."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = Settings()

        assert settings.lm_studio.base_url == "http://localhost:1234/v1"
        assert settings.image_generation.default_width == 864
        assert settings.image_generation.default_height == 1152
        assert settings.image_generation.seed == 0
        assert settings.lm_studio.model == "google/gemma-4-26b-a4b-qat"
        assert settings.image_generation.model_path.name == "ernie-image-turbo-4bit"
        assert settings.server.sse_queue_size == 100
        assert settings.enhancement.default_softness == 0.5

    def test_settings_from_env(self):
        """Test that settings can be loaded from environment variables."""
        env_vars = {
            "PROMPT_GEN_LM_STUDIO_URL": "http://test:5000/v1",
            "PROMPT_GEN_DEFAULT_WIDTH": "1024",
            "PROMPT_GEN_DEFAULT_HEIGHT": "768",
            "PROMPT_GEN_SSE_QUEUE_SIZE": "200",
            "PROMPT_GEN_IMAGE_SEED": "42",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.lm_studio.base_url == "http://test:5000/v1"
            assert settings.image_generation.default_width == 1024
            assert settings.image_generation.default_height == 768
            assert settings.server.sse_queue_size == 200
            assert settings.image_generation.seed == 42

    def test_cross_platform_default_model_path(self):
        """Test that model path uses cross-platform fallback on non-macOS."""
        from config import _default_model_path, ImageGenerationConfig

        default = _default_model_path()
        assert isinstance(default, Path)
        # On any platform this should resolve to a sensible location
        assert "mflux" in str(default)
        assert "ernie-image-turbo-4bit" in str(default)

    def test_env_override_uses_default_function(self):
        """Test that env override still works with cross-platform default."""
        from config import _default_model_path, ImageGenerationConfig

        custom_path = "/custom/model/path"
        with patch.dict(os.environ, {"PROMPT_GEN_ERNIE_MODEL_PATH": custom_path}):
            settings = Settings.from_env()
            assert str(settings.image_generation.model_path) == custom_path

    def test_immutable_config(self):
        """Test that config dataclasses are immutable."""
        config = LMStudioConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.base_url = "http://changed"

    def test_invalid_env_vars(self):
        """Test that invalid environment variables fall back to defaults."""
        env_vars = {
            "PROMPT_GEN_DEFAULT_WIDTH": "not-an-integer",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.image_generation.default_width == 864

    def test_invalid_server_timeouts(self):
        """Test that invalid server timeouts raise ValueError for numeric edge cases."""
        # Numeric: zero and negative values hit __post_init__ validation
        with pytest.raises(ValueError, match="sse_timeout must be positive"):
            ServerConfig(sse_timeout=0)
        with pytest.raises(ValueError, match="sse_timeout must be positive"):
            ServerConfig(sse_timeout=-1)
        with pytest.raises(ValueError, match="worker_timeout must be positive"):
            ServerConfig(worker_timeout=0)
        with pytest.raises(ValueError, match="worker_timeout must be positive"):
            ServerConfig(worker_timeout=-1)

    def test_non_numeric_env_float_falls_back_to_default(self):
        """Test that non-numeric float env vars (e.g. 'abc') fall back to defaults via _get_env_float ValueError handler.

        Distinct from the numeric edge cases above: this exercises the _get_env_float
        ValueError → logger.warning → default return path, ensuring Settings.from_env()
        never raises for malformed float strings on float-valued env keys.
        """
        from config import _get_env_float, logger, ServerConfig

        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "abc"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout)
            assert result == ServerConfig.sse_timeout
            mock_warn.assert_called_once()

        # Through Settings.from_env(): non-numeric float env values fall back silently
        with patch.dict(os.environ, {"PROMPT_GEN_WORKER_TIMEOUT": "not-a-number"}):
            settings = Settings.from_env()
            assert settings.server.worker_timeout == ServerConfig.worker_timeout

    def test_invalid_lm_studio_timeouts(self):
        """Test that invalid LM Studio timeouts raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LMStudioConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            LMStudioConfig(timeout=-1)

    def test_invalid_image_dimensions(self):
        """Test that invalid image dimensions raise ValueError."""
        with pytest.raises(ValueError, match="default_width must be positive"):
            ImageGenerationConfig(default_width=0)
        with pytest.raises(ValueError, match="default_height must be positive"):
            ImageGenerationConfig(default_height=-1)
    def test_invalid_enhancement_config(self):
        """Test that invalid enhancement settings raise ValueError."""
        from config import EnhancementConfig

        with pytest.raises(ValueError, match="default_softness must be between 0 and 1"):
            EnhancementConfig(default_softness=-0.5)
        with pytest.raises(ValueError, match="default_softness must be between 0 and 1"):
            EnhancementConfig(default_softness=2.0)
        with pytest.raises(ValueError, match="default_scale must be at least 1"):
            EnhancementConfig(default_scale=0)
        with pytest.raises(ValueError, match="default_scale must be at least 1"):
            EnhancementConfig(default_scale=-3)

    def test_nan_inf_env_vars_fall_back(self):
        """Test that NaN and Infinity env values fall back to defaults."""
        for bad_value in ("NaN", "nan", "inf", "-inf", "Infinity"):
            with patch.dict(os.environ, {"PROMPT_GEN_DEFAULT_WIDTH": bad_value}):
                settings = Settings.from_env()
                assert settings.image_generation.default_width == 864

    def test_float_inf_env_var_falls_back_to_default(self):
        """Test that _get_env_float rejects Infinity values for float env vars.

        Covers all three explicit branches in production: positive inf, negative -inf,
        and the capitalized "Infinity" string — each routed through Settings.from_env()
        to exercise the full config-load path.
        """
        from config import Settings, ServerConfig, _get_env_float
        for bad_value in ("inf", "-inf", "Infinity"):
            with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": bad_value}):
                settings = Settings.from_env()
                assert settings.server.sse_timeout == ServerConfig.sse_timeout

    def test_get_env_str_ignores_empty_string(self):
        """Test that _get_env_str returns default when env var is empty."""
        from config import _get_env_str, LMStudioConfig
        with patch.dict(os.environ, {"PROMPT_GEN_LM_STUDIO_URL": ""}):
            result = _get_env_str("PROMPT_GEN_LM_STUDIO_URL", LMStudioConfig.base_url)
            assert result == LMStudioConfig.base_url

    def test_get_env_float_rejects_nan(self):
        """Test that _get_env_float returns default and logs warning on NaN."""
        from config import _get_env_float, logger, ServerConfig
        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "nan"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout)
            assert result == ServerConfig.sse_timeout
            mock_warn.assert_called_once()

    def test_get_env_float_accepts_valid_positive_float(self):
        """Test that _get_env_float returns valid positive float unchanged with no warning."""
        from config import _get_env_float, logger
        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "42.5"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout)
            assert result == 42.5
            mock_warn.assert_not_called()

    def test_get_env_float_accepts_positive_zero(self):
        """Test that _get_env_float returns zero (positive boundary) unchanged."""
        from config import _get_env_float, logger
        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "0.0"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout)
            assert result == 0.0
            mock_warn.assert_not_called()

    def test_get_env_float_accepts_valid_negative_when_allowed(self):
        """Test that _get_env_float returns a negative float unchanged when negative_allowed=True, with no warning."""
        from config import _get_env_float, logger
        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "-42.5"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", 10.0)
            assert result == -42.5
            mock_warn.assert_not_called()

    def test_negative_float_softness_env_var_falls_back(self):
        """Test that negative float values via env var fall back to defaults."""
        from config import Settings, EnhancementConfig
        with patch.dict(os.environ, {"PROMPT_GEN_ENHANCE_SOFTNESS": "-0.5"}):
            settings = Settings.from_env()
            assert settings.enhancement.default_softness == EnhancementConfig.default_softness

    def test_negative_float_timeout_env_var_falls_back(self):
        """Test that negative float timeout values via env var fall back to defaults."""
        from config import Settings, ServerConfig
        with patch.dict(os.environ, {"PROMPT_GEN_WORKER_TIMEOUT": "-10"}):
            settings = Settings.from_env()
            assert settings.server.worker_timeout == ServerConfig.worker_timeout

    def test_negative_float_env_var_logs_warning_when_not_allowed(self):
        """Test that _get_env_float logs a warning and returns the default when negative_allowed=False.

        Distinct from fallback-only tests: this asserts both the logging contract
        (logger.warning called exactly once) AND the fallback value, making any
        regression in the negative-validation branch of _get_env_float immediately
        detectable — not just silent.
        """
        from config import Settings, ServerConfig, _get_env_float, logger

        default = ServerConfig.sse_timeout  # 5.0

        with patch.dict(os.environ, {"PROMPT_GEN_SSE_TIMEOUT": "-10"}), \
             patch.object(logger, "warning") as mock_warn:
            result = _get_env_float("PROMPT_GEN_SSE_TIMEOUT", default, negative_allowed=False)
            assert result == default
            assert mock_warn.call_count == 1

        with patch.dict(os.environ, {"PROMPT_GEN_WORKER_TIMEOUT": "-10"}):
            settings = Settings.from_env()
            assert settings.server.worker_timeout == ServerConfig.worker_timeout

    def test_negative_scale_env_var_raises_via_enhancement_config(self):
        """Test that negative scale via env var raises ValueError through EnhancementConfig.

        _get_env_int silently accepts negatives/zeros, but EnhancementConfig.__post_init__
        validates the value. Unlike float keys (which fall back to defaults), int scale
        values pass validation and then fail at dataclass construction — no graceful fallback.
        """
        from config import Settings
        with patch.dict(os.environ, {"PROMPT_GEN_ENHANCE_SCALE": "-1"}):
            with pytest.raises(ValueError, match="default_scale must be at least 1"):
                Settings.from_env()

    def test_negative_width_via_enhancement_config(self):
        """Test that negative width via env var raises ValueError through ImageGenerationConfig.

        _get_env_int silently accepts negatives/zeros, but ImageGenerationConfig.__post_init__
        validates the value. Unlike float keys (which fall back to defaults), int width
        values pass validation and then fail at dataclass construction — no graceful fallback.
        """
        from config import Settings
        with patch.dict(os.environ, {"PROMPT_GEN_DEFAULT_WIDTH": "-100"}):
            with pytest.raises(ValueError, match="default_width must be positive"):
                Settings.from_env()

    def test_negative_height_via_enhancement_config(self):
        """Test that negative height via env var raises ValueError through ImageGenerationConfig.

        _get_env_int silently accepts negatives/zeros, but ImageGenerationConfig.__post_init__
        validates the value. Unlike float keys (which fall back to defaults), int height
        values pass validation and then fail at dataclass construction — no graceful fallback.
        """
        from config import Settings
        with patch.dict(os.environ, {"PROMPT_GEN_DEFAULT_HEIGHT": "-200"}):
            with pytest.raises(ValueError, match="default_height must be positive"):
                Settings.from_env()

    def test_zero_width_via_enhancement_config(self):
        """Test that zero width via env var raises ValueError through ImageGenerationConfig.

        _get_env_int accepts zero as a valid int, but ImageGenerationConfig.__post_init__
        validates the value (must be > 0). Zero passes validation at the env layer and
        then fails at dataclass construction — no graceful fallback like floats get.
        """
        from config import Settings
        with patch.dict(os.environ, {"PROMPT_GEN_DEFAULT_WIDTH": "0"}):
            with pytest.raises(ValueError, match="default_width must be positive"):
                Settings.from_env()

    def test_non_numeric_seed_falls_back(self):
        """Test that non-numeric seed values fall back to default 0."""
        from config import Settings
        with patch.dict(os.environ, {"PROMPT_GEN_IMAGE_SEED": "abc"}):
            settings = Settings.from_env()
            assert settings.image_generation.seed == 0

    def test_nan_timeout_raises(self):
        """Test that NaN timeout values raise ValueError instead of silently passing."""
        import math
        with pytest.raises(ValueError, match="timeout must be positive"):
            LMStudioConfig(timeout=float("nan"))

    def test_nan_server_timeouts_raise(self):
        """Test that NaN server timeouts raise ValueError instead of silently passing."""
        import math
        with pytest.raises(ValueError, match="sse_timeout must be positive"):
            ServerConfig(sse_timeout=float("nan"))
        with pytest.raises(ValueError, match="worker_timeout must be positive"):
            ServerConfig(worker_timeout=float("nan"))

    def test_path_properties(self):
        """Verify path properties are Path objects and correct."""
        assert isinstance(paths.root_dir, Path)
        assert isinstance(paths.generated_dir, Path)
        assert isinstance(paths.grammars_dir, Path)
        assert isinstance(paths.prompts_dir, Path)
        assert isinstance(paths.saved_dir, Path)
        assert isinstance(paths.queue_path, Path)
        assert isinstance(paths.templates_dir, Path)

        # Verify path relationships
        assert paths.generated_dir == paths.root_dir / "generated"
        assert paths.grammars_dir == paths.generated_dir / "grammars"
        assert paths.prompts_dir == paths.generated_dir / "prompts"
        assert paths.saved_dir == paths.generated_dir / "saved"
        assert paths.queue_path == paths.generated_dir / "queue.json"
        assert paths.templates_dir == paths.root_dir / "templates"

    def test_path_config_resolves_from_file_not_cwd(self, tmp_path):
        """Test that PathConfig resolves relative to __file__, not cwd.

        The singleton `paths` is defined at module level in src/config.py.
        Changing the working directory must NOT affect where root_dir points.
        """
        import config as cfg_module
        original_root = str(cfg_module.paths.root_dir)
        expected_root = Path(__file__).parent.parent

        assert str(expected_root) == original_root, (
            "root_dir should resolve to src/config.py's parent directory"
        )

    def test_settings_from_env_empty_string_falls_back(self):
        """Test that empty-string env vars fall back to defaults for string and float keys.

        _get_env_str and _get_env_float both treat empty/whitespace strings as unset.
        This differs from missing (None) only in which branch triggers, but the result is identical.
        """
        import os
        from unittest.mock import patch
        from config import Settings, LMStudioConfig

        env_vars = {
            "PROMPT_GEN_LM_STUDIO_URL": "",
            "PROMPT_GEN_SSE_TIMEOUT": "   ",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.lm_studio.base_url == LMStudioConfig.base_url
            assert settings.server.sse_timeout == ServerConfig.sse_timeout


class TestEnvVarDocs:
    """Tests for ENV_VAR_DOCS documentation dictionary and format_env_docs helper.

    These ensure operators can discover all available env vars without reading source code —
    the core user-visible value of this feature. The completeness check catches any new env var
    added to Settings.from_env() that hasn't been documented yet.
    """

    def test_all_env_vars_documented(self):
        """Test that every env var used in Settings.from_env() has a doc entry."""
        from config import ENV_VAR_DOCS, Settings
        expected_keys = {
            "PROMPT_GEN_LM_STUDIO_URL",
            "PROMPT_GEN_LM_STUDIO_MODEL",
            "PROMPT_GEN_LM_STUDIO_TIMEOUT",
            "PROMPT_GEN_DEFAULT_WIDTH",
            "PROMPT_GEN_DEFAULT_HEIGHT",
            "PROMPT_GEN_IMAGE_SEED",
            "PROMPT_GEN_ERNIE_MODEL_PATH",
            "PROMPT_GEN_SSE_QUEUE_SIZE",
            "PROMPT_GEN_SSE_TIMEOUT",
            "PROMPT_GEN_WORKER_TIMEOUT",
            "PROMPT_GEN_ENHANCE_SOFTNESS",
            "PROMPT_GEN_ENHANCE_SCALE",
        }
        assert ENV_VAR_DOCS.keys() == expected_keys

    def test_format_env_docs_returns_comment_block(self):
        """Test that format_env_docs produces a comment-block string with all env vars."""
        from config import format_env_docs, ENV_VAR_DOCS

        output = format_env_docs()
        # Starts with header and blank line
        assert output.startswith("# Environment Variables\n\n")
        # Each key has description and default lines
        for key in ENV_VAR_DOCS:
            assert f"# {key}  (" in output
            assert "#   Description:" in output
            assert "#   Default:" in output

    def test_format_env_docs_sorted_keys(self):
        """Test that format_env_docs outputs keys in sorted order."""
        from config import format_env_docs, ENV_VAR_DOCS

        output = format_env_docs()
        # Extract key lines and verify sorted
        key_lines = [line for line in output.split("\n") if line.startswith("# PROMPT_GEN_")]
        assert key_lines == sorted(key_lines)

    def test_format_env_docs_custom_dict(self):
        """Test that format_env_docs accepts a custom dict."""
        from config import format_env_docs, ENV_VAR_DOCS

        partial = {k: v for k, v in ENV_VAR_DOCS.items() if "LM_STUDIO" in k}
        output = format_env_docs(partial)
        # Should only contain LM_STUDIO keys
        assert "PROMPT_GEN_LM_STUDIO_URL" in output
        assert "PROMPT_GEN_DEFAULT_WIDTH" not in output

    def test_default_values_match_actual_defaults(self):
        """Test that ENV_VAR_DOCS default values match actual config defaults."""
        from config import ENV_VAR_DOCS, Settings, LMStudioConfig, ImageGenerationConfig, ServerConfig, EnhancementConfig
        s = Settings()
        assert ENV_VAR_DOCS["PROMPT_GEN_LM_STUDIO_URL"]["default"] == s.lm_studio.base_url
        assert ENV_VAR_DOCS["PROMPT_GEN_LM_STUDIO_MODEL"]["default"] == s.lm_studio.model
        assert ENV_VAR_DOCS["PROMPT_GEN_DEFAULT_WIDTH"]["default"] == str(s.image_generation.default_width)
        assert ENV_VAR_DOCS["PROMPT_GEN_SSE_QUEUE_SIZE"]["default"] == str(s.server.sse_queue_size)
        assert ENV_VAR_DOCS["PROMPT_GEN_ENHANCE_SCALE"]["default"] == str(s.enhancement.default_scale)

    def test_format_env_docs_type_annotations(self):
        """Test that type annotations are included for each env var."""
        from config import format_env_docs, ENV_VAR_DOCS
        output = format_env_docs()
        # int types should show (int), float should show (float), str should show (str)
        assert "(int)" in output
        assert "(float)" in output
        assert "(str)" in output

    def test_format_env_docs_empty_dict(self):
        """Test that format_env_docs handles an empty dict gracefully."""
        from config import format_env_docs
        output = format_env_docs({})
        assert output.startswith("# Environment Variables\n\n")

    def test_path_config_env_docs_file_property(self):
        """Test that PathConfig.env_docs_file returns the expected generated path."""
        from pathlib import Path
        from config import paths

        env_doc_path = paths.env_docs_file
        assert isinstance(env_doc_path, Path)
        assert str(env_doc_path).endswith(".env.example")
        # Should be under generated_dir, not root or src
        assert ".env.example" in str(env_doc_path)
        assert "generated" in str(env_doc_path)

    def test_generate_env_example_writes_documented_file(self):
        """Test that generate_env_example writes a properly formatted .env.example file.

        Verifies the new convenience function: it should call format_env_docs(),
        write to paths.env_docs_file, create parent dirs if needed, and return
        the path of the written file. The produced content must match what
        format_env_docs() returns for the same input.
        """
        from config import generate_env_example, ENV_VAR_DOCS, format_env_docs

        result = generate_env_example()

        assert result.exists()
        assert str(result).endswith(".env.example")

        content = result.read_text()
        expected = format_env_docs(ENV_VAR_DOCS)
        assert content == expected

        # Verify the file is a proper comment block with all env vars documented
        assert "# Environment Variables\n" in content
        for key in ENV_VAR_DOCS:
            assert f"# {key}" in content

    def test_generate_env_example_accepts_custom_subset(self):
        """Test that generate_env_example writes only the subset of env vars passed as argument.

        Mirrors test_format_env_docs_custom_dict for format_env_docs(): when a partial
        dict is passed, only those keys should appear in the output file. This exercises
        the delegation path through format_env_docs with non-default input.
        """
        from config import generate_env_example, ENV_VAR_DOCS, format_env_docs

        partial = {k: v for k, v in ENV_VAR_DOCS.items() if "LM_STUDIO" in k}
        result = generate_env_example(partial)

        assert result.exists()

        content = result.read_text()
        expected = format_env_docs(partial)
        assert content == expected

        # Verify only the subset keys are present
        for key in partial:
            assert f"# {key}" in content
        # And non-LM_STUDIO keys should NOT be present
        assert "PROMPT_GEN_DEFAULT_WIDTH" not in content
