"""Tests for utility functions."""

import json

import pytest
from PIL import Image

from utils import (
    PNG_TEXT_MAX_BYTES,
    load_run_metadata,
    get_prefix_from_metadata,
    count_images_in_run,
    get_prompts_from_run,
    _get_prompt_text,
    _truncate_for_png,
    backup_run,
    is_backup_run,
    scan_flat_archives,
    get_flat_archive_metadata,
    run_has_images,
    delete_run,
    format_run_timestamp,
)


class TestUtils:
    """Tests for utility functions."""

    def test_format_run_timestamp_valid(self):
        """Test formatting a valid timestamp."""
        result = format_run_timestamp("20240115_143022")
        assert result == "2024-01-15 14:30:22"

    def test_format_run_timestamp_invalid_length(self):
        """Test that invalid length returns original."""
        result = format_run_timestamp("20240115")
        assert result == "20240115"

    def test_format_run_timestamp_missing_separator(self):
        """Test that missing separator returns original."""
        result = format_run_timestamp("20240115-143022")
        assert result == "20240115-143022"

    def test_format_run_timestamp_empty(self):
        """Test that empty string returns empty."""
        result = format_run_timestamp("")
        assert result == ""

    def test_load_run_metadata(self, temp_dir):
        """Test loading metadata from a run directory."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 10,
            "user_prompt": "a dragon",
        }))

        metadata = load_run_metadata(temp_dir)
        assert metadata["prefix"] == "test"
        assert metadata["count"] == 10

    def test_load_run_metadata_not_found(self, temp_dir):
        """Test that ValueError is raised when no metadata found."""
        with pytest.raises(ValueError, match="No metadata file found"):
            load_run_metadata(temp_dir)

    def test_load_run_metadata_malformed_json(self, temp_dir):
        """Test loading metadata with malformed JSON."""
        (temp_dir / "test.metaprompt.json").write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_run_metadata(temp_dir)

    def test_get_prefix_from_metadata(self, temp_dir):
        """Test getting prefix from metadata."""
        (temp_dir / "cat.metaprompt.json").write_text(json.dumps({
            "prefix": "cat",
        }))

        prefix = get_prefix_from_metadata(temp_dir)
        assert prefix == "cat"

    def test_get_prefix_from_metadata_default(self, temp_dir):
        """Test that default prefix is returned when metadata missing."""
        prefix = get_prefix_from_metadata(temp_dir)
        assert prefix == "image"

    def test_count_images_in_run(self, temp_dir):
        """Test counting images in a run directory."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_text("fake image")
        (temp_dir / "test_0_1.png").write_text("fake image")
        (temp_dir / "test_1_0.png").write_text("fake image")

        count = count_images_in_run(temp_dir)
        assert count == 3

    def test_get_prompts_from_run_filtering(self, temp_dir):
        """Test that non-prompt files are filtered out."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0.txt").write_text("First prompt")
        (temp_dir / "test_1.txt").write_text("Second prompt")
        (temp_dir / "test_2.txt").write_text("Third prompt")
        (temp_dir / "test_metadata.json").write_text("{}")
        (temp_dir / "test_0.raw.txt").write_text("raw")
        (temp_dir / "test_1_image.png").write_text("image")
        # Glob-match but regex-rejected: non-numeric middle segment should not pass.
        (temp_dir / "test_abc.txt").write_text("not a prompt")

        prompts = get_prompts_from_run(temp_dir)
        assert prompts == ["First prompt", "Second prompt", "Third prompt"]
        assert len(prompts) == 3
        assert prompts[0] == "First prompt"
        assert prompts[2] == "Third prompt"

    def test_get_prompt_text_found(self, temp_dir):
        """Test _get_prompt_text returns file content when the prompt file exists."""
        run_dir = temp_dir / "prompts" / "run123"
        run_dir.mkdir(parents=True)
        (run_dir / "image_5.txt").write_text("My dragon prompt")

        result = _get_prompt_text(run_dir, "image", 5)
        assert result == "My dragon prompt"

    def test_get_prompt_text_missing(self, temp_dir):
        """Test _get_prompt_text returns empty string when the prompt file is absent."""
        run_dir = temp_dir / "prompts" / "run123"
        run_dir.mkdir(parents=True)
        # No image_9.txt created

        result = _get_prompt_text(run_dir, "image", 9)
        assert result == ""

    def test_get_prompts_from_run_empty_directory(self, temp_dir):
        """Test get_prompts_from_run returns [] when no prompt files exist."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        prompts = get_prompts_from_run(temp_dir)
        assert prompts == []
        assert len(prompts) == 0

    def test_backup_run(self, temp_dir):
        """Test backing up a run directory to flat files with EXIF metadata."""
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create test files
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
            "model": "test-model",
        }))
        (run_dir / "test_0.txt").write_text("Prompt 0")

        # Create a real PNG file for Pillow to process
        img = Image.new('RGB', (10, 10), color='red')
        img.save(run_dir / "test_0_0.png")

        # Create backup
        saved_files = backup_run(run_dir, saved_dir, reason="pre_regenerate")

        # Check that files were saved
        assert len(saved_files) == 1
        assert saved_files[0].exists()

        # Check flat file naming pattern: prefix_timestamp_promptIdx_imgIdx.png
        filename = saved_files[0].name
        assert filename.startswith("test_")
        assert filename.endswith("_0_0.png")

        # Check that metadata is embedded in PNG
        metadata = get_flat_archive_metadata(saved_files[0])
        assert metadata.get("user_prompt") == "a dragon"
        assert metadata.get("model") == "test-model"
        assert metadata.get("backup_reason") == "pre_regenerate"
        assert metadata.get("prompt") == "Prompt 0"

        # Check is_backup_run still works for original run_dir
        assert is_backup_run(run_dir) is False

    def test_backup_run_not_found(self, temp_dir):
        """Test that backup fails for non-existent directory."""
        non_existent = temp_dir / "does_not_exist"
        saved_dir = temp_dir / "saved"

        with pytest.raises(ValueError, match="Run directory not found"):
            backup_run(non_existent, saved_dir)

    def test_backup_run_no_images(self, temp_dir):
        """Test backup with no images returns empty list."""
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create metadata but no images
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        saved_files = backup_run(run_dir, saved_dir, reason="manual_archive")
        assert saved_files == []

    def test_run_has_images(self, temp_dir):
        """Test checking if a run has images."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        # No images yet
        assert run_has_images(temp_dir) is False

        # Add an image
        (temp_dir / "test_0_0.png").write_bytes(b"fake image")
        assert run_has_images(temp_dir) is True

    def test_delete_run(self, temp_dir):
        """Test deleting a run directory."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        # Create test files
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))
        (run_dir / "test_0.txt").write_text("Prompt 0")
        (run_dir / "test_0_0.png").write_bytes(b"fake image")

        assert run_dir.exists()

        # Delete the run
        delete_run(run_dir, prompts_dir)

        assert not run_dir.exists()

    def test_delete_run_not_found(self, temp_dir):
        """Test that delete fails for non-existent directory."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        non_existent = prompts_dir / "does_not_exist"

        with pytest.raises(ValueError, match="Run directory not found"):
            delete_run(non_existent, prompts_dir)

    def test_delete_run_outside_prompts_dir(self, temp_dir):
        """Test that delete fails for directories outside prompts_dir."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        outside_dir = temp_dir / "outside"
        outside_dir.mkdir()
        (outside_dir / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        with pytest.raises(ValueError, match="not inside prompts directory"):
            delete_run(outside_dir, prompts_dir)

    def test_delete_run_archive_protected(self, temp_dir):
        """Test that delete fails for archived galleries."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        # Create backup metadata (marks this as an archive)
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
                "backup_reason": "manual_archive",
            },
        }))

        with pytest.raises(ValueError, match="Cannot delete archived galleries"):
            delete_run(run_dir, prompts_dir)

    def test_delete_run_path_traversal(self, temp_dir):
        """Test that delete fails for path traversal attempts."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        # Attempt path traversal
        traversal_path = prompts_dir / ".." / "outside"
        traversal_path.mkdir(parents=True)
        (traversal_path / "test.metaprompt.json").write_text(json.dumps({"prefix": "test"}))

        with pytest.raises(ValueError, match="not inside prompts directory"):
            delete_run(traversal_path, prompts_dir)

    def test_get_flat_archive_metadata_corrupted_png(self, temp_dir):
        """Test get_flat_archive_metadata with corrupted PNG returns empty dict."""
        corrupted_file = temp_dir / "corrupted.png"
        corrupted_file.write_bytes(b"not a valid png file")

        metadata = get_flat_archive_metadata(corrupted_file)
        assert metadata == {}

    def test_get_flat_archive_metadata_missing_file(self, temp_dir):
        """Test get_flat_archive_metadata with missing file returns empty dict."""
        missing_file = temp_dir / "missing.png"

        metadata = get_flat_archive_metadata(missing_file)
        assert metadata == {}

    def test_is_backup_run_true(self, temp_dir):
        """Test is_backup_run returns True when backup_info.is_backup is set."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {"is_backup": True},
        }))
        assert is_backup_run(temp_dir) is True

    def test_is_backup_run_false(self, temp_dir):
        """Test is_backup_run returns False when backup_info.is_backup is absent."""
        (temp_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
        }))
        assert is_backup_run(temp_dir) is False

    def test_is_backup_run_no_metadata(self, temp_dir):
        """Test is_backup_run returns False when no metadata file exists."""
        assert is_backup_run(temp_dir) is False

    def test_scan_flat_archives_empty(self, temp_dir):
        """Test scan_flat_archives returns empty list for non-existent directory."""
        result = scan_flat_archives(temp_dir / "nonexistent")
        assert result == []

    def test_scan_flat_archives_groups_by_prefix_and_timestamp(self, temp_dir):
        """Test scan_flat_archives groups flat PNG files by prefix+timestamp."""
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create flat archive PNGs with the expected naming pattern
        img = Image.new('RGB', (10, 10), color='blue')
        img.save(saved_dir / "cat_20240115_143022_0_0.png")
        img.save(saved_dir / "cat_20240115_143022_0_1.png")
        img.save(saved_dir / "dog_20240115_143022_0_0.png")

        result = scan_flat_archives(saved_dir)
        assert len(result) == 2

        # Find the cat archive
        cat_archive = next(a for a in result if a["prefix"] == "cat")
        assert cat_archive["timestamp"] == "20240115_143022"
        assert cat_archive["image_count"] == 2

        # Find the dog archive
        dog_archive = next(a for a in result if a["prefix"] == "dog")
        assert dog_archive["image_count"] == 1

    def test_scan_flat_archives_ignores_non_matching_files(self, temp_dir):
        """Test scan_flat_archives ignores files that don't match the flat pattern."""
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create a file that doesn't match the flat archive pattern
        (saved_dir / "random_file.png").write_bytes(b"fake")

        result = scan_flat_archives(saved_dir)
        assert len(result) == 0

    def test_scan_flat_archives_preserves_all_timestamps(self, temp_dir):
        """Test scan_flat_archives returns every distinct archive present on disk.

        The flat archive scanner groups PNGs by (prefix, timestamp) and yields
        one dict per group. This test verifies all three archives are returned
        regardless of creation order, confirming no archive is dropped silently.
        """
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create three PNGs with different timestamps to verify grouping completeness
        img = Image.new('RGB', (10, 10), color='green')
        img.save(saved_dir / "gal_20240301_100000_0_0.png")
        img.save(saved_dir / "gal_20240115_143022_0_0.png")
        img.save(saved_dir / "gal_20240220_081500_0_0.png")

        result = scan_flat_archives(saved_dir)
        timestamps = sorted(a["timestamp"] for a in result)
        assert timestamps == ["20240115_143022", "20240220_081500", "20240301_100000"]

    def test_scan_flat_archives_first_image_is_lexicographically_smallest(self, temp_dir):
        """Test scan_flat_archives sets first_image to the lex-smallest PNG name per group.

        The flat archive scanner tracks `first_image` by comparing filenames
        lexicographically within each (prefix, timestamp) group — this is used as
        a thumbnail path for gallery views, so correctness matters. This test
        verifies that given multiple images in one group, first_image points to the
        filename that sorts lowest alphabetically, regardless of creation order.
        """
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create flat archive PNGs where lex-smallest is NOT the numerically smallest index.
        # Files: cat_20240115_143022_9_0.png (lex smallest) and cat_20240115_143022_10_1.png
        img = Image.new('RGB', (10, 10), color='blue')
        img.save(saved_dir / "cat_20240115_143022_9_0.png")
        img.save(saved_dir / "cat_20240115_143022_10_1.png")

        result = scan_flat_archives(saved_dir)
        assert len(result) == 1
        archive = result[0]
        assert archive["image_count"] == 2
        # '9' sorts before '10' lexicographically because '9' > '1' char-wise... actually
        # lex: "cat_...9_0.png" vs "cat_...10_1.png": at position of 9 vs 1, '9'>'1', so
        # "cat_20240115_143022_10_1.png" is lex-smallest.
        assert archive["first_image"].name == "cat_20240115_143022_10_1.png"

    def test_backup_run_embeds_image_generation_settings(self, temp_dir):
        """Test backup_run embeds image generation settings in PNG text chunks."""
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create test files with image_generation settings in metadata
        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
            "model": "test-model",
            "created_at": "2024-01-01T12:00:00Z",
            "image_generation": {
                "width": 512,
                "height": 768,
                "steps": 30,
            },
        }))
        (run_dir / "test_0.txt").write_text("Prompt 0")

        # Create a real PNG file for Pillow to process
        img = Image.new('RGB', (10, 10), color='red')
        img.save(run_dir / "test_0_0.png")

        # Create backup
        saved_files = backup_run(run_dir, saved_dir, reason="pre_regenerate")

        assert len(saved_files) == 1

        # Check that image generation settings are embedded in PNG text chunks
        metadata = get_flat_archive_metadata(saved_files[0])
        assert metadata.get("width") == "512"
        assert metadata.get("height") == "768"
        assert metadata.get("steps") == "30"
        assert metadata.get("created_at") == "2024-01-01T12:00:00Z"

    def test_scan_flat_archives_images_list_covers_all_prompt_indices(self, temp_dir):
        """Test scan_flat_archives populates images list for every prompt index.

        A flat archive may contain PNGs from multiple prompts (different promptIdx)
        sharing the same prefix+timestamp. This test verifies that all such images
        are collected in `images`, not just the first or last one encountered.
        """
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        img = Image.new('RGB', (10, 10), color='red')
        # Same prefix+timestamp but different promptIdx values (3, 7) and imgIdx values (0, 2)
        img.save(saved_dir / "gal_20240510_180000_3_0.png")
        img.save(saved_dir / "gal_20240510_180000_7_2.png")
        img.save(saved_dir / "gal_20240510_180000_3_1.png")

        result = scan_flat_archives(saved_dir)
        assert len(result) == 1
        archive = result[0]
        assert archive["image_count"] == 3
        names = {p.name for p in archive["images"]}
        expected_names = {
            "gal_20240510_180000_3_0.png",
            "gal_20240510_180000_7_2.png",
            "gal_20240510_180000_3_1.png",
        }
        assert names == expected_names

    def test_scan_flat_archives_first_image_cross_prompt_idx(self, temp_dir):
        """Test scan_flat_archives first_image is lex-smallest across all prompt indices.

        When multiple images from different promptIdx values share the same archive
        timestamp, first_image must point to the overall lexicographically smallest
        filename — not just within one promptIdx's files. This catches regressions where
        implementation might sort per-prompt-index instead of full-filename comparison.
        """
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        img = Image.new('RGB', (10, 10), color='purple')
        # Same prefix+timestamp but different promptIdx values: 5_0, 12_0, 3_9
        # Lex sort of full filenames: "gal_20240510_180000_12_0.png" < "gal_20240510_180000_3_9.png" < "gal_20240510_180000_5_0.png"
        # So first_image must be the one with promptIdx 12 (not 3, which is numerically smallest)
        img.save(saved_dir / "gal_20240510_180000_5_0.png")
        img.save(saved_dir / "gal_20240510_180000_12_0.png")
        img.save(saved_dir / "gal_20240510_180000_3_9.png")

        result = scan_flat_archives(saved_dir)
        assert len(result) == 1
        archive = result[0]
        assert archive["image_count"] == 3
        # Lex-smallest full filename across all files (not just per-promptIdx smallest)
        assert archive["first_image"].name == "gal_20240510_180000_12_0.png"

    def test_truncate_for_png_under_limit(self):
        """Test _truncate_for_png returns value unchanged when under limit."""
        short_text = "a simple prompt" * 10
        assert len(short_text.encode()) < 8000
        result = _truncate_for_png(short_text, "prompt")
        assert result == short_text

    def test_truncate_for_png_over_limit(self):
        """Test _truncate_for_png truncates values exceeding the byte limit."""
        long_text = "x" * 10000  # well over 8KB in UTF-8
        result = _truncate_for_png(long_text, "prompt")
        assert len(result.encode()) <= PNG_TEXT_MAX_BYTES
        # Verify actual truncation occurred (not just that the result is within limits)
        assert result != long_text

    def test_truncate_for_png_preserves_unicode(self):
        """Test truncation preserves UTF-8 characters at the boundary."""
        # Use multibyte chars to exercise utf-8 byte counting
        emoji = "\U0001F680" * 3000  # ~12KB in UTF-8 (4 bytes each)
        result = _truncate_for_png(emoji, "prompt")
        assert len(result.encode("utf-8")) <= PNG_TEXT_MAX_BYTES

    def test_truncate_for_png_passes_through_empty(self):
        """Test empty string passes through unchanged."""
        assert _truncate_for_png("", "key") == ""

    def test_backup_run_large_prompt_does_not_crash(self, temp_dir):
        """Test backup_run with a prompt exceeding PNG text limits.

        A large prompt (>8KB) previously crashed Pillow's add_text via OOM or
        compression failure. This test verifies the new truncation path makes
        the behavior deterministic: the archive is created with truncated prompt
        text, not lost entirely.
        """
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create metadata with a large prompt (>8KB)
        huge_prompt = "dragon " * 2500  # ~17.5KB in UTF-8

        (run_dir / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
            "model": "test-model",
        }))
        (run_dir / "test_0.txt").write_text(huge_prompt)

        img = Image.new('RGB', (10, 10), color='red')
        img.save(run_dir / "test_0_0.png")

        # Previously this would crash; now it must succeed with truncated text.
        saved_files = backup_run(run_dir, saved_dir, reason="pre_regenerate")

        assert len(saved_files) == 1
        metadata = get_flat_archive_metadata(saved_files[0])
        # The prompt was truncated but is still non-empty and starts correctly
        assert metadata.get("prompt", "").startswith("dragon ")
