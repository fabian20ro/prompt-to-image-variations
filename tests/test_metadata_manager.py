"""Tests for metadata_manager.py - centralized metadata operations."""

import json
from pathlib import Path

import pytest

from metadata_manager import (
    MetadataManager,
    MetadataError,
    MetadataNotFoundError,
    RunMetadata,
    resolve_gallery_layout,
    load_metadata,
    save_metadata,
    get_metadata_prefix,
)


class TestRunMetadata:
    """Tests for RunMetadata dataclass."""

    def test_from_dict_basic(self):
        """Test creating RunMetadata from a basic dictionary."""
        data = {
            "prefix": "test",
            "count": 10,
            "user_prompt": "test prompt",
            "model": "ernie-image-turbo",
        }
        metadata = RunMetadata.from_dict(data)

        assert metadata.prefix == "test"
        assert metadata.count == 10
        assert metadata.user_prompt == "test prompt"
        assert metadata.model == "ernie-image-turbo"

    def test_from_dict_with_defaults(self):
        """Test that defaults are used for missing fields."""
        data = {}
        metadata = RunMetadata.from_dict(data)

        assert metadata.prefix == "image"
        assert metadata.count == 0
        assert metadata.model == "unknown"

    def test_run_metadata_invalid_count(self):
        """Test that negative count raises ValueError."""
        with pytest.raises(ValueError, match="count must be non-negative"):
            RunMetadata(count=-1)

        data = {
            "prefix": "test",
            "image_generation": {
                "enabled": True,
                "width": 1024,
                "height": 1024,
            },
        }
        metadata = RunMetadata.from_dict(data)

        assert metadata.image_generation["enabled"] is True
        assert metadata.image_generation["width"] == 1024

    def test_to_dict_roundtrip(self):
        """Test that to_dict produces valid JSON for from_dict."""
        original = {
            "prefix": "test",
            "count": 5,
            "user_prompt": "test",
            "model": "ernie-image-turbo",
            "created_at": "2024-01-01",
            "grammar_cached": True,
            "image_generation": {"enabled": True},
            "extra_field": "preserved",
        }
        metadata = RunMetadata.from_dict(original)
        result = metadata.to_dict()

        assert result["prefix"] == original["prefix"]
        assert result["count"] == original["count"]
        assert result["extra_field"] == "preserved"

    def test_get_method(self):
        """Test the get method for backwards compatibility."""
        data = {"prefix": "test", "custom_key": "custom_value"}
        metadata = RunMetadata.from_dict(data)

        assert metadata.get("custom_key") == "custom_value"
        assert metadata.get("nonexistent", "default") == "default"


class TestMetadataManager:
    """Tests for MetadataManager class."""

    def test_find_metadata_file_exists(self, temp_dir):
        """Test finding existing metadata file."""
        meta_file = temp_dir / "test.metaprompt.json"
        meta_file.write_text('{"prefix": "test"}')

        result = MetadataManager.find_metadata_file(temp_dir)
        assert result == meta_file

    def test_find_metadata_file_not_exists(self, temp_dir):
        """Test when no metadata file exists."""
        result = MetadataManager.find_metadata_file(temp_dir)
        assert result is None

    def test_load_success(self, temp_dir):
        """Test loading metadata successfully."""
        data = {"prefix": "test", "count": 5, "user_prompt": "hello"}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(data))

        metadata = MetadataManager.load(temp_dir)

        assert isinstance(metadata, RunMetadata)
        assert metadata.prefix == "test"
        assert metadata.count == 5

    def test_load_not_found(self, temp_dir):
        """Test loading when no metadata file exists."""
        with pytest.raises(MetadataNotFoundError, match="No metadata file found"):
            MetadataManager.load(temp_dir)

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON metadata."""
        (temp_dir / "test.metaprompt.json").write_text("{invalid json")

        with pytest.raises(MetadataError, match="Invalid JSON"):
            MetadataManager.load(temp_dir)

    def test_load_raw(self, temp_dir):
        """Test loading raw dictionary."""
        data = {"prefix": "test", "custom": "value"}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(data))

        result = MetadataManager.load_raw(temp_dir)

        assert isinstance(result, dict)
        assert result["prefix"] == "test"
        assert result["custom"] == "value"

    def test_save_with_dict(self, temp_dir):
        """Test saving metadata from dictionary."""
        data = {"prefix": "myprefix", "count": 10}

        result = MetadataManager.save(temp_dir, data)

        assert result.exists()
        assert result.name == "myprefix.metaprompt.json"
        saved = json.loads(result.read_text())
        assert saved["count"] == 10

    def test_save_with_run_metadata(self, temp_dir):
        """Test saving metadata from RunMetadata object."""
        metadata = RunMetadata(prefix="custom", count=20, user_prompt="test")

        result = MetadataManager.save(temp_dir, metadata)

        assert result.name == "custom.metaprompt.json"
        saved = json.loads(result.read_text())
        assert saved["count"] == 20

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates directory if needed."""
        nested = temp_dir / "nested" / "dir"
        data = {"prefix": "test", "count": 5}

        result = MetadataManager.save(nested, data)

        assert result.exists()
        assert nested.exists()

    def test_update(self, temp_dir):
        """Test updating specific fields."""
        initial = {"prefix": "test", "count": 5, "user_prompt": "original"}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        result = MetadataManager.update(temp_dir, count=10, user_prompt="updated")

        assert result.count == 10
        assert result.user_prompt == "updated"

        # Verify file was updated
        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert saved["count"] == 10
        assert saved["user_prompt"] == "updated"

    def test_update_not_found(self, temp_dir):
        """Test updating when no metadata exists."""
        with pytest.raises(MetadataNotFoundError):
            MetadataManager.update(temp_dir, count=10)

    def test_update_deletion(self, temp_dir):
        """Test deleting a key via update by passing None."""
        initial = {"prefix": "test", "count": 5, "user_prompt": "original"}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        # In our implementation, passing None to update deletes the key from the raw dict
        result = MetadataManager.update(temp_dir, user_prompt=None)

        # The object will have default value (empty string), but the file should not have the key
        assert result.user_prompt == ""
        # Check that it's actually gone from the JSON file
        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert "user_prompt" not in saved

        (temp_dir / "test.metaprompt.json").write_text('{"prefix": "test"}')
        assert MetadataManager.exists(temp_dir) is True

    def test_get_prefix(self, temp_dir):
        """Test getting prefix from metadata."""
        (temp_dir / "myprefix.metaprompt.json").write_text('{"prefix": "myprefix"}')

        result = MetadataManager.get_prefix(temp_dir)
        assert result == "myprefix"

    def test_get_prefix_default(self, temp_dir):
        """Test getting default prefix when no metadata."""
        result = MetadataManager.get_prefix(temp_dir, default="custom")
        assert result == "custom"

    def test_get_image_settings(self, temp_dir):
        """Test getting image generation settings."""
        data = {
            "prefix": "test",
            "image_generation": {
                "enabled": True,
                "width": 1024,
                "model": "ernie-image-turbo",
            },
        }
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(data))

        settings = MetadataManager.get_image_settings(temp_dir)

        assert settings["enabled"] is True
        assert settings["width"] == 1024

    def test_get_image_settings_empty(self, temp_dir):
        """Test getting image settings when not present."""
        (temp_dir / "test.metaprompt.json").write_text('{"prefix": "test"}')

        settings = MetadataManager.get_image_settings(temp_dir)
        assert settings == {}

    def test_resolve_gallery_layout_preserves_explicit_zero_images_per_prompt(self):
        """Explicit zero-image layouts should survive normalization."""
        metadata = {
            "gallery_layout": {
                "images_per_prompt": 0,
                "max_prompts": 3,
            },
            "image_generation": {
                "images_per_prompt": 4,
                "max_prompts": 7,
            },
        }

        layout = resolve_gallery_layout(metadata, prompt_count=2)

        assert layout["images_per_prompt"] == 0
        assert layout["max_prompts"] == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_metadata_success(self, temp_dir):
        """Test load_metadata convenience function."""
        data = {"prefix": "test", "count": 5}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(data))

        result = load_metadata(temp_dir)
        assert result["prefix"] == "test"

    def test_load_metadata_not_found(self, temp_dir):
        """Test load_metadata returns empty dict when not found."""
        result = load_metadata(temp_dir)
        assert result == {}

    def test_save_metadata_success(self, temp_dir):
        """Test save_metadata convenience function."""
        data = {"prefix": "test", "count": 5}

        result = save_metadata(temp_dir, data, "test")

        assert result is not None
        assert result.exists()

    def test_get_metadata_prefix(self, temp_dir):
        """Test get_metadata_prefix convenience function."""
        (temp_dir / "myprefix.metaprompt.json").write_text('{"prefix": "myprefix"}')

        result = get_metadata_prefix(temp_dir)
        assert result == "myprefix"

    def test_get_metadata_prefix_default(self, temp_dir):
        """Test get_metadata_prefix with default."""
        result = get_metadata_prefix(temp_dir, "default_prefix")
        assert result == "default_prefix"
