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
        assert metadata.model == "ernie-image-turbo"

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

    def test_find_legacy_metadata_pattern(self, temp_dir):
        """Test find_metadata_file returns legacy _metadata.json files for backward compatibility.

        The search loop in find_metadata_file supports the old *_metadata.json naming
        convention alongside new *.metaprompt.json files. This test ensures that legacy
        metadata directories (pre-refactor) are still discovered correctly.
        """
        meta_file = temp_dir / "test_metadata.json"
        meta_file.write_text('{"prefix": "legacy"}')

        result = MetadataManager.find_metadata_file(temp_dir)
        assert result == meta_file

    def test_find_metaprompt_before_legacy(self, temp_dir):
        """Test that new .metaprompt.json takes precedence over legacy _metadata.json.

        When both naming patterns coexist in the same directory, the modern file wins —
        matching the iteration order in find_metadata_file's glob loop.
        """
        legacy = temp_dir / "test_legacy_metadata.json"
        legacy.write_text('{"prefix": "legacy"}')

        modern = temp_dir / "test.metaprompt.json"
        modern.write_text('{"prefix": "modern"}')

        result = MetadataManager.find_metadata_file(temp_dir)
        assert result == modern

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

    def test_save_atomic_write(self, temp_dir):
        """Test that save uses atomic write (tempfile + os.replace)."""
        from unittest.mock import patch as mock_patch
        data = {"prefix": "test", "count": 5}

        with mock_patch("metadata_manager.os.replace") as replace_mock:
            result = MetadataManager.save(temp_dir, data)
            replace_mock.assert_called_once()

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

    def test_update_deletion_roundtrip(self, temp_dir):
        """Test that deleted keys are absent after reload via load_raw."""
        initial = {"prefix": "test", "count": 5, "user_prompt": "original", "model": "ernie"}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        MetadataManager.update(temp_dir, user_prompt=None)

        # Roundtrip through load_raw — the deleted key must not reappear
        raw = MetadataManager.load_raw(temp_dir)
        assert "user_prompt" not in raw
        assert raw["count"] == 5
        assert raw["prefix"] == "test"

    def test_update_merges_dict_fields(self, temp_dir):
        """Test that dict-valued updates merge into existing dicts."""
        initial = {
            "prefix": "test",
            "image_generation": {
                "enabled": True,
                "width": 1024,
                "height": 768,
            },
        }
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        result = MetadataManager.update(
            temp_dir, image_generation={"width": 512, "format": "png"}
        )

        assert result.image_generation["enabled"] is True
        assert result.image_generation["width"] == 512
        assert result.image_generation["height"] == 768
        assert result.image_generation["format"] == "png"

    def test_update_overwrites_non_dict_value(self, temp_dir):
        """Test that non-dict updates overwrite scalar values directly."""
        initial = {"prefix": "test", "count": 5}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        result = MetadataManager.update(temp_dir, count=42)

        assert result.count == 42
        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert saved["count"] == 42

    def test_update_overwrites_scalar_with_dict(self, temp_dir):
        """Test that a dict value overwrites an existing scalar field on disk.

        When the existing key holds a non-dict (e.g. int) and the update
        supplies a dict, the merge branch is skipped and the dict replaces
        the value wholesale — matching line 268 in metadata_manager.py's
        isinstance guard logic. Note: this also exposes that RunMetadata
        rejects non-int counts via __post_init__, so we assert only on-disk.
        """
        initial = {"prefix": "test", "count": 5}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        with pytest.raises(TypeError):
            MetadataManager.update(temp_dir, count={"nested": "value"})

        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert saved["count"] == {"nested": "value"}
        # Confirm the field was replaced entirely — not merged into an int
        assert isinstance(saved["count"], dict)

    def test_update_overwrites_dict_with_scalar(self, temp_dir):
        """Test that a scalar value overwrites an existing dict field on disk.

        When the existing key holds a dict and the update supplies a
        non-dict (e.g. string), the merge branch is skipped and the
        scalar replaces the dict wholesale — matching line 267 in
        metadata_manager.py's isinstance guard logic.
        """
        initial = {
            "prefix": "test",
            "image_generation": {"enabled": True, "width": 1024},
        }
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        result = MetadataManager.update(temp_dir, image_generation="removed")

        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert saved["image_generation"] == "removed"
        # Confirm the dict was replaced entirely — not merged with a string
        assert isinstance(saved["image_generation"], str)

    def test_update_adds_missing_key(self, temp_dir):
        """Test that update adds a key absent from existing metadata.

        When the update supplies a value for a key not present in the
        original JSON (e.g. 'regenerated_at' on first generation), the
        merge branch is skipped and the new key is inserted — matching
        line 268 in metadata_manager.py's isinstance guard logic.
        """
        initial = {"prefix": "test", "count": 5}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps(initial))

        result = MetadataManager.update(temp_dir, source="pipeline")

        assert result.source == "pipeline"
        saved = json.loads((temp_dir / "test.metaprompt.json").read_text())
        assert saved["source"] == "pipeline"
        # Other fields must be preserved — not lost by the update path
        assert saved["prefix"] == "test"
        assert saved["count"] == 5

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

    def test_get_gallery_layout_missing_metadata(self, temp_dir):
        """Test get_gallery_layout returns deterministic defaults when metadata is absent.

        When no metadata file exists the load path raises MetadataNotFoundError;
        the fallback must resolve to concrete values — not None or empty — so callers
        can safely use them for layout decisions without further guard checks.
        """
        layout = MetadataManager.get_gallery_layout(temp_dir, prompt_count=3)

        assert isinstance(layout, dict), "fallback must return a dict"
        assert layout["images_per_prompt"] == 1
        assert layout["max_prompts"] is None

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

    def test_resolve_gallery_layout_with_run_metadata(self):
        """Test fallback resolution when called with a RunMetadata object."""
        metadata = RunMetadata(
            prefix="test",
            count=5,
            image_generation={"images_per_prompt": 3, "max_prompts": 10},
        )

        layout = resolve_gallery_layout(metadata, prompt_count=4)

        assert layout["images_per_prompt"] == 3
        assert layout["max_prompts"] == 4


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
