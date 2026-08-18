"""Tests for GalleryService."""

import json
from pathlib import Path

import pytest

from services.gallery_service import (
    GalleryService,
    GalleryNotFoundError,
    MetadataNotFoundError,
)
from conftest import create_run_files


class TestGalleryService:
    """Tests for GalleryService."""

    def test_get_run_directory_success(self, temp_dir):
        """Test getting a valid run directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        result = service.get_run_directory("20240101_120000_abc123")
        assert result == run_dir

    def test_get_run_directory_not_found(self, temp_dir):
        """Test getting a non-existent run directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        prompts_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        with pytest.raises(GalleryNotFoundError, match="Gallery not found"):
            service.get_run_directory("nonexistent")

    def test_get_run_directory_archive(self, temp_dir):
        """Test getting an archive directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()
        archive_dir = saved_dir / "archive_20240101"
        archive_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        result = service.get_run_directory("archive_20240101", is_archive=True)
        assert result == archive_dir

    def test_get_run_directory_archive_not_found(self, temp_dir):
        """Test getting a non-existent archive directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        with pytest.raises(GalleryNotFoundError, match="Archive not found"):
            service.get_run_directory("nonexistent", is_archive=True)

    def test_load_metadata_success(self, temp_dir):
        """Test loading metadata."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        service = GalleryService(temp_dir, temp_dir)
        metadata = service.load_metadata(temp_dir)

        assert metadata["prefix"] == "test"
        assert metadata["user_prompt"] == "a dragon"

    def test_load_metadata_not_found(self, temp_dir):
        """Test loading metadata when not found."""
        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(MetadataNotFoundError, match="No metadata file found"):
            service.load_metadata(temp_dir)

    def test_load_metadata_malformed_json(self, temp_dir):
        """Test loading metadata with malformed JSON."""
        (temp_dir / "test.metaprompt.json").write_text("not valid json")

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(json.JSONDecodeError, match="Expecting value"):
            service.load_metadata(temp_dir)

    def test_get_prefix(self, temp_dir):
        """Test getting prefix from metadata."""
        (temp_dir / "cat.metaprompt.json").write_text(json.dumps({"prefix": "cat"}))

        service = GalleryService(temp_dir, temp_dir)
        assert service.get_prefix(temp_dir) == "cat"

    def test_get_prefix_default(self, temp_dir):
        """Test getting default prefix when no metadata."""
        service = GalleryService(temp_dir, temp_dir)
        assert service.get_prefix(temp_dir) == "image"

    def test_get_prefix_metadata_exists_no_prefix_key(self, temp_dir):
        """Test get_prefix returns 'image' when metadata exists but lacks 'prefix' key."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "user_prompt": "a cat",
        }))

        service = GalleryService(temp_dir, temp_dir)
        assert service.get_prefix(temp_dir) == "image"

    def test_load_grammar_success(self, temp_dir):
        """Test loading grammar."""
        grammar = {"origin": ["#subject#"], "subject": ["cat"]}
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_grammar.json").write_text(json.dumps(grammar))

        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(temp_dir, "test")

        assert '"origin"' in result
        assert '"subject"' in result

    def test_load_grammar_not_found(self, temp_dir):
        """Test loading grammar when file doesn't exist."""
        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(temp_dir, "test")
        assert result is None

    def test_load_grammar_auto_detects_missing_prefix_returns_none(self, temp_dir):
        """Test load_grammar returns None when auto-detecting prefix but no grammar file exists."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "cat.metaprompt.json").write_text(json.dumps({"prefix": "cat"}))

        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(run_dir)  # no explicit prefix — auto-detects from metadata

        assert result is None

    def test_load_grammar_auto_detects_existing_grammar_returns_content(self, temp_dir):
        """Test load_grammar returns grammar content when auto-detecting prefix and file exists."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "cat.metaprompt.json").write_text(json.dumps({"prefix": "cat"}))
        (run_dir / "cat_grammar.json").write_text(
            json.dumps({"origin": ["#subject#"], "subject": ["dragon"]})
        )

        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(run_dir)  # auto-detects prefix from metadata

        assert result is not None
        assert '"origin"' in result
        assert '"subject"' in result

    def test_get_grammar_file_explicit_prefix(self, temp_dir):
        """Test get_grammar_file with explicit prefix returns correct path."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()

        service = GalleryService(temp_dir, temp_dir)
        result = service.get_grammar_file(run_dir, prefix="test")

        assert result == run_dir / "test_grammar.json"

    def test_get_grammar_file_auto_detects_prefix(self, temp_dir):
        """Test get_grammar_file auto-detects prefix from metadata when None."""
        (temp_dir / "cat.metaprompt.json").write_text(json.dumps({"prefix": "cat"}))

        service = GalleryService(temp_dir, temp_dir)
        result = service.get_grammar_file(temp_dir)

        assert result == temp_dir / "cat_grammar.json"

    def test_load_prompts(self, temp_dir):
        """Test loading prompts."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0.txt").write_text("First prompt")
        (temp_dir / "test_1.txt").write_text("Second prompt")
        (temp_dir / "test_2.txt").write_text("Third prompt")
        # Should NOT include these (more than 1 underscore):
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_not_a_prompt.txt").write_text("other")

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(temp_dir)

        assert len(prompts) == 3
        assert prompts[0] == "First prompt"
        assert prompts[1] == "Second prompt"
        assert prompts[2] == "Third prompt"

    def test_load_prompts_with_missing_files(self, temp_dir):
        """Test loading prompts when directory is empty."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(temp_dir)

        assert prompts == []

    def test_validate_file_access_valid(self, temp_dir):
        """Test validating file access for valid path."""
        file_path = temp_dir / "subdir" / "file.txt"

        service = GalleryService(temp_dir, temp_dir)
        result = service.validate_file_access(file_path, temp_dir)

        assert result is True

    def test_validate_file_access_invalid_path_traversal(self, temp_dir):
        """Test validating file access blocks path traversal."""
        base_dir = temp_dir / "base"
        base_dir.mkdir()
        outside_file = temp_dir / "outside" / "file.txt"

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(ValueError, match="Access denied"):
            service.validate_file_access(outside_file, base_dir)

    def test_is_backup_run_false(self, temp_dir):
        """Test is_backup_run returns False for normal runs."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is False

    def test_is_backup_run_true(self, temp_dir):
        """Test is_backup_run returns True for backup runs."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
            },
        }))

        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is True

    def test_is_backup_run_no_metadata(self, temp_dir):
        """Test is_backup_run returns False when no metadata."""
        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is False

    def test_count_images(self, temp_dir):
        """Test counting images."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_0_1.png").write_bytes(b"fake")
        (temp_dir / "test_1_0.png").write_bytes(b"fake")
        # Should NOT count these:
        (temp_dir / "test_0.txt").write_text("prompt")
        (temp_dir / "other.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        count = service.count_images(temp_dir)

        assert count == 3

    def test_list_images(self, temp_dir):
        """Test listing images."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_1_0.png").write_bytes(b"fake")
        (temp_dir / "test_0_1.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        images = service.list_images(temp_dir)

        assert len(images) == 3
        # Check sorted order
        assert images[0].name == "test_0_0.png"
        assert images[1].name == "test_0_1.png"
        assert images[2].name == "test_1_0.png"

    def test_count_images_multi_segment(self, temp_dir):
        """Test counting images with multi-segment prefix (files have two segments after prefix)."""
        run_dir = temp_dir / "run_dir"
        run_dir.mkdir(parents=True)
        # Pattern is "{prefix}_*_*.png" so files need TWO underscore-separated segments after prefix
        (run_dir / "test_01_run_a.png").touch()
        (run_dir / "test_01_run_b.png").touch()
        (run_dir / "test_01_extra.jpg").touch()  # not .png — excluded
        (run_dir / "not_an_image.txt").touch()

        service = GalleryService(temp_dir, temp_dir)
        assert service.count_images(run_dir, prefix="test_01") == 2

    def test_list_images_multi_segment(self, temp_dir):
        """Test listing images with multi-segment prefix (files have two segments after prefix)."""
        run_dir = temp_dir / "run_dir"
        run_dir.mkdir(parents=True)
        # Pattern is "{prefix}_*_*.png" so files need TWO underscore-separated segments after prefix
        (run_dir / "test_01_run_a.png").touch()
        (run_dir / "test_01_run_b.PNG").touch()  # Case-sensitive glob won't match .PNG
        (run_dir / "test_01_extra.jpg").touch()  # not .png — excluded
        (run_dir / "not_an_image.txt").touch()

        service = GalleryService(temp_dir, temp_dir)
        images = service.list_images(run_dir, prefix="test_01")
        assert len(images) == 1  # only the lowercase .png matches
        assert any(i.name == "test_01_run_a.png" for i in images)

    def test_count_images_with_timestamps(self, temp_dir):
        """Test counting images."""
        run_dir = temp_dir / "run_dir"
        run_dir.mkdir(parents=True)
        (run_dir / "test_01_20240101_120000.png").touch()
        (run_dir / "test_01_20240101_120001.png").touch()
        (run_dir / "test_01_20240101_120002.jpg").touch()
        (run_dir / "not_an_image.txt").touch()

        service = GalleryService(temp_dir, temp_dir)
        assert service.count_images(run_dir, prefix="test_01") == 2

    def test_get_metadata_file_alternate_pattern(self, temp_dir):
        """Test get_metadata_file returns *_metadata.json when metaprompt is absent."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        service = GalleryService(temp_dir, temp_dir)
        meta_file = service.get_metadata_file(run_dir)

        assert meta_file.name == "test_metadata.json"

    def test_count_images_zero_matches(self, temp_dir):
        """Test count_images returns 0 when no matching images exist."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "other_0_0.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        assert service.count_images(run_dir, prefix="test") == 0

    def test_list_images_empty_directory(self, temp_dir):
        """Test list_images returns empty list when no images match."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()

        service = GalleryService(temp_dir, temp_dir)
        assert service.list_images(run_dir, prefix="test") == []

    def test_load_prompts_filters_multi_segment_stems(self, temp_dir):
        """Test load_prompts excludes files with multiple underscores in stem."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test_0.txt").write_text("valid")       # 1 underscore — kept
        (run_dir / "test_a_b.txt").write_text("invalid")   # 2 underscores — filtered
        (run_dir / "test_c_d_e.txt").write_text("filtered") # 3 underscores — filtered

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(run_dir, prefix="test")

        assert prompts == ["valid"]

    def test_get_metadata_file_prefers_metaprompt(self, temp_dir):
        """Test get_metadata_file prefers *.metaprompt.json over *_metadata.json."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (run_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test_alt"}))

        service = GalleryService(temp_dir, temp_dir)
        meta_file = service.get_metadata_file(run_dir)

        assert meta_file.name == "test.metaprompt.json"

    def test_get_metadata_file_fallback_to_metadata(self, temp_dir):
        """Test get_metadata_file falls back to *_metadata.json when no metaprompt exists."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        service = GalleryService(temp_dir, temp_dir)
        meta_file = service.get_metadata_file(run_dir)

        assert meta_file.name == "test_metadata.json"

    def test_get_metadata_file_no_matching_patterns_in_existing_dir(self, temp_dir):
        """Test get_metadata_file raises MetadataNotFoundError when no matching patterns exist."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        # Only non-matching files — neither *.metaprompt.json nor *_metadata.json
        (run_dir / "some_other_file.txt").write_text("data")

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(MetadataNotFoundError, match=f"No metadata file found in {run_dir}"):
            service.get_metadata_file(run_dir)

    def test_get_metadata_file_both_patterns_match(self, temp_dir):
        """Test get_metadata_file returns a metaprompt file when both patterns have files."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "a.metaprompt.json").write_text(json.dumps({"prefix": "mp"}))
        (run_dir / "b.metaprompt.json").write_text(json.dumps({"prefix": "mp2"}))
        (run_dir / "c_metadata.json").write_text(json.dumps({"prefix": "md"}))

        service = GalleryService(temp_dir, temp_dir)
        meta_file = service.get_metadata_file(run_dir)

        assert meta_file.name.endswith(".metaprompt.json")

    def test_load_metadata_empty_file(self, temp_dir):
        """Test loading metadata from an empty file raises JSONDecodeError."""
        (temp_dir / "test.metaprompt.json").write_text("")

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(json.JSONDecodeError):
            service.load_metadata(temp_dir)

    def test_load_metadata_empty_object(self, temp_dir):
        """Test loading metadata from a valid empty JSON object returns empty dict."""
        (temp_dir / "test.metaprompt.json").write_text("{}")

        service = GalleryService(temp_dir, temp_dir)
        metadata = service.load_metadata(temp_dir)

        assert isinstance(metadata, dict)
        assert metadata == {}
        # No prefix key should fall back to default when consumed downstream,
        # but load_metadata itself returns the raw parsed dict.
        assert "prefix" not in metadata

    def test_load_prompts_nonexistent_directory(self, temp_dir):
        """Test load_prompts returns empty list for missing directory."""
        nonexistent = temp_dir / "does_not_exist"

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(nonexistent)
        assert prompts == []

    def test_validate_file_access_absolute_path_outside_base(self, temp_dir):
        """Test validate_file_access raises ValueError for absolute paths outside base."""
        real_root = Path("/tmp")

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(ValueError, match="Access denied"):
            service.validate_file_access(real_root.resolve(), temp_dir)

    def test_validate_file_access_symlink_targeting_inside_base(self, temp_dir):
        """Test validate_file_access allows symlinks whose resolved target is inside base dir."""
        import os

        # Create a real file inside temp_dir
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        target_file = subdir / "real.txt"
        target_file.touch()

        # Create a symlink inside temp_dir pointing to the same target
        link_path = temp_dir / "link_to_subdir"
        os.symlink(subdir, link_path)
        resolved_link = link_path.resolve(strict=True)

        service = GalleryService(temp_dir, temp_dir)
        result = service.validate_file_access(resolved_link, temp_dir)

        assert result is True

    def test_validate_file_access_at_base_boundary(self, temp_dir):
        """Test validate_file_access accepts paths exactly at the base directory root."""
        file_path = temp_dir / "file.txt"
        file_path.touch()

        service = GalleryService(temp_dir, temp_dir)
        result = service.validate_file_access(file_path, temp_dir)

        assert result is True

    def test_validate_file_access_nested_deep_traversal(self, temp_dir):
        """Test validate_file_access rejects paths that escape via ../ in deep nesting."""
        inside_dir = temp_dir / "deep" / "nested"
        inside_dir.mkdir(parents=True)
        safe_file = inside_dir / "safe.txt"
        safe_file.touch()

        outside_path = (temp_dir / ".." / "etc" / "passwd").resolve()

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(ValueError, match="Access denied"):
            service.validate_file_access(outside_path, temp_dir)


class TestRunSummary:
    """Tests for get_run_summary."""

    def test_get_run_summary(self, temp_dir):
        """Test getting a run summary with all fields populated."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (run_dir / "test_0.txt").write_text("prompt 1")
        (run_dir / "test_1.txt").write_text("prompt 2")
        (run_dir / "test_0_0.png").write_bytes(b"fake")
        (run_dir / "test_0_1.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        summary = service.get_run_summary("run")

        assert summary["prefix"] == "test"
        assert summary["prompt_count"] == 2
        assert summary["image_count"] == 2
        assert summary["is_backup"] is False

    def test_get_run_summary_missing_directory(self, temp_dir):
        """Test get_run_summary raises GalleryNotFoundError for missing run."""
        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(GalleryNotFoundError):
            service.get_run_summary("nonexistent")

    def test_get_run_summary_default_prefix(self, temp_dir):
        """Test get_run_summary uses default prefix when no metadata."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "image_0.txt").write_text("prompt")
        (run_dir / "image_0_0.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        summary = service.get_run_summary("run")

        assert summary["prefix"] == "image"
        assert summary["prompt_count"] == 1
        assert summary["image_count"] == 1

    def test_get_run_summary_is_backup(self, temp_dir):
        """Test get_run_summary reports backup status."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {"is_backup": True},
        }))

        service = GalleryService(temp_dir, temp_dir)
        summary = service.get_run_summary("run")

        assert summary["is_backup"] is True

    def test_get_run_summary_empty_existing_directory(self, temp_dir):
        """Test get_run_summary with an existing but empty run directory (no metadata)."""
        run_dir = temp_dir / "run"
        run_dir.mkdir()

        service = GalleryService(temp_dir, temp_dir)
        summary = service.get_run_summary("run")

        assert summary["prefix"] == "image"
        assert summary["prompt_count"] == 0
        assert summary["image_count"] == 0
        assert summary["is_backup"] is False
