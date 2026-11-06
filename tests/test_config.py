"""Tests for centralized configuration."""

import os
from pathlib import Path

import pytest
from unittest.mock import patch

from config import Settings, LMStudioConfig, paths


class TestConfig:
    """Tests for centralized configuration."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = Settings()

        assert settings.lm_studio.base_url == "http://localhost:1234/v1"
        assert settings.image_generation.default_width == 864
        assert settings.image_generation.default_height == 1152
        assert settings.lm_studio.model == "qwen3.8-27b-splash"
        assert settings.image_generation.model_path.name == "flux3-klein-q4"
        assert settings.server.sse_queue_size == 100
        assert settings.enhancement.default_softness == 0.5

    def test_settings_from_env(self):
        """Test that settings can be loaded from environment variables."""
        env_vars = {
            "PROMPT_GEN_LM_STUDIO_URL": "http://test:5000/v1",
            "PROMPT_GEN_DEFAULT_WIDTH": "1024",
            "PROMPT_GEN_DEFAULT_HEIGHT": "768",
            "PROMPT_GEN_SSE_QUEUE_SIZE": "200",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.lm_studio.base_url == "http://test:5000/v1"
            assert settings.image_generation.default_width == 1024
            assert settings.image_generation.default_height == 768
            assert settings.server.sse_queue_size == 200

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
        # Verify paths are Path objects
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
