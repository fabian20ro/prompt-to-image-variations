"""Tests for interactive gallery index generation."""

import json
import struct
import zlib
from unittest.mock import patch

from gallery_index import generate_master_index


class TestGalleryIndexInteractive:
    """Tests for interactive gallery index generation."""

    def test_index_interactive_mode(self, temp_dir):
        """Test that interactive index includes generation form."""
        (temp_dir / "prompts").mkdir()

        # Generate interactive index
        index_path = generate_master_index(temp_dir, interactive=True)

        assert index_path.exists()
        content = index_path.read_text()

        # Check for interactive elements
        assert "generate-form" in content
        assert "generate-from-grammar-form" in content
        assert "Images/Prompt (0 = prompt-only layout)" in content
        assert "queue-status" in content
        assert "btn-kill" in content
        assert "btn-clear" in content
        assert "/api/generate" in content
        assert "EventSource" in content

    def test_index_prefers_display_title_for_run_cards(self, temp_dir):
        active_run = temp_dir / "prompts" / "20240101_120000_abc123"
        active_run.mkdir(parents=True)
        (active_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "hidden raw prompt",
            "display_title": "Grammar import",
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        assert "Grammar import" in content
        assert "hidden raw prompt" not in content

    def test_index_non_interactive_mode(self, temp_dir):
        """Test that non-interactive index doesn't include form."""
        (temp_dir / "prompts").mkdir()

        # Generate non-interactive index
        index_path = generate_master_index(temp_dir, interactive=False)

        content = index_path.read_text()

        # Check that form is NOT present
        assert "generate-form" not in content
        assert "queue-status" not in content

    def test_index_interactive_uses_toasts_modal_and_no_blocking_dialogs(self, temp_dir):
        """Interactive index should use shared toast/modal helpers instead of alert/confirm."""
        (temp_dir / "prompts").mkdir()

        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        assert "toast-region" in content
        assert "confirm-modal" in content
        assert "showToast(" in content
        assert "confirmAction(" in content
        assert "withButtonBusy(" in content
        assert "queue_cleared" in content

        # Guard against regressions to blocking browser dialogs.
        import re
        assert re.search(r'\balert\s*\(', content) is None, (
            "Generated index must not contain alert() calls"
        )
        assert re.search(r'\bconfirm\s*\(', content) is None, (
            "Generated index must not contain confirm() calls"
        )

    def test_index_with_archived_runs(self, temp_dir):
        """Test that index shows archived runs separately."""
        # Create active run
        active_run = temp_dir / "prompts" / "20240101_120000_abc123"
        active_run.mkdir(parents=True)
        (active_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "active prompt",
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        # Create archived run
        archived_run = temp_dir / "saved" / "20240101_100000_def456_20240101_130000"
        archived_run.mkdir(parents=True)
        (archived_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "archived prompt",
            "backup_info": {
                "is_backup": True,
                "backup_reason": "pre_regenerate",
                "source_run_id": "20240101_100000_def456",
            },
        }))
        (archived_run / "test_gallery.html").write_text("<html></html>")

        # Generate index
        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        # Check for both runs
        assert "active prompt" in content
        assert "archived prompt" in content
        assert "Saved Archives" in content
        assert "archive-badge" in content
        assert "Pre-Regen" in content  # Badge for pre_regenerate

    def test_index_flat_archive_prefers_display_title(self, temp_dir):
        """Flat archive cards should prefer display_title over raw user_prompt."""
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create a minimal valid PNG matching the naming pattern
        import struct
        png_path = saved_dir / "image_20240101_120000_0_0.png"
        # Write a minimal 1x1 red PNG with embedded text metadata
        png_data = bytearray()
        # PNG signature
        png_data.extend(b'\x89PNG\r\n\x1a\n')

        # IHDR chunk (minimal)
        ihdr_payload = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)  # width, height, bitdepth=8, colortype=2(RGB), compression, filter, interlace
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_payload))
        png_data.extend(struct.pack('>I', len(ihdr_payload)))
        png_data.extend(b'IHDR')
        png_data.extend(ihdr_payload)
        png_data.extend(ihdr_crc)

        # IDAT chunk (minimal compressed data for 1x1 red pixel)
        raw_data = b'\x00\xff\x00\x00'  # filter byte + RGB
        compressed = zlib.compress(raw_data)
        idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
        png_data.extend(struct.pack('>I', len(compressed)))
        png_data.extend(b'IDAT')
        png_data.extend(compressed)
        png_data.extend(idat_crc)

        # IEND chunk
        iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
        png_data.extend(struct.pack('>I', 0))
        png_data.extend(b'IEND')
        png_data.extend(iend_crc)

        with open(png_path, 'wb') as f:
            f.write(bytes(png_data))

        # Mock get_flat_archive_metadata to return controlled data
        with patch('gallery_index.get_flat_archive_metadata', return_value={
            "display_title": "Custom Gallery Title",
            "user_prompt": "raw hidden prompt text",
            "model": "ERNIE v3.5",
        }):
            index_path = generate_master_index(temp_dir, interactive=True)
            content = index_path.read_text()

            # display_title should be preferred over raw user_prompt
            assert "Custom Gallery Title" in content
            assert "raw hidden prompt text" not in content

    def test_index_handles_flat_archive_metadata_errors_gracefully(self, temp_dir):
        """Flat archive metadata errors must not crash the index generator."""
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        # Create a minimal valid PNG matching the naming pattern
        import struct
        png_path = saved_dir / "image_20240101_120000_0_0.png"
        png_data = bytearray()
        png_data.extend(b'\x89PNG\r\n\x1a\n')

        ihdr_payload = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_payload))
        png_data.extend(struct.pack('>I', len(ihdr_payload)))
        png_data.extend(b'IHDR')
        png_data.extend(ihdr_payload)
        png_data.extend(ihdr_crc)

        raw_data = b'\x00\xff\x00\x00'
        compressed = zlib.compress(raw_data)
        idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
        png_data.extend(struct.pack('>I', len(compressed)))
        png_data.extend(b'IDAT')
        png_data.extend(compressed)
        png_data.extend(idat_crc)

        iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
        png_data.extend(struct.pack('>I', 0))
        png_data.extend(b'IEND')
        png_data.extend(iend_crc)

        with open(png_path, 'wb') as f:
            f.write(bytes(png_data))

        # Mock get_flat_archive_metadata to raise an exception (simulating corrupted metadata)
        with patch('gallery_index.get_flat_archive_metadata', side_effect=RuntimeError("corrupted")):
            index_path = generate_master_index(temp_dir, interactive=True)
            content = index_path.read_text()

            # The index should still be generated with sensible defaults
            assert index_path.exists()
            assert "Archived images" in content  # default user_prompt fallback
            assert ">N/A<" in content or "N/A" in content  # default model fallback appears rendered

    def test_index_resolves_nested_image_generation_model_fallback(self, temp_dir):
        """_extract_run_info should fall back to image_generation.model when top-level model is absent."""
        active_run = temp_dir / "prompts" / "20240101_120000_xyz789"
        active_run.mkdir(parents=True)
        (active_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "fallback test prompt",
            "image_generation": {"model": "Flux-dev-v2"},
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        assert "Flux-dev-v2" in content

    def test_build_generation_form_returns_complete_markup(self):
        from gallery_index import _build_generation_form
        html = _build_generation_form()
        assert "<form id=\"generate-form\">" in html
        assert 'name="prompt"' in html
        assert 'name="prefix"' in html
        assert 'id="count"' in html
        assert 'id="temperature"' in html

    def test_build_queue_status_bar_contains_queue_element(self):
        from gallery_index import _build_queue_status_bar
        html = _build_queue_status_bar()
        assert "queue-status" in html
        assert "<div" in html and "</div>" in html

    def test_build_log_panel_returns_collapsible_markup(self):
        from gallery_index import _build_log_panel
        html = _build_log_panel()
        # Requires both the specific element ID and collapsible structure — catches
        # regressions where the component is replaced or its ID changes.
        assert 'id="log-panel"' in html
        assert "<details" in html

    def test_build_notifications_contains_toast_and_confirm(self):
        from gallery_index import _build_notifications
        html = _build_notifications()
        assert "toast-region" in html or "showToast" in html
        assert "confirm-modal" in html or "confirmAction" in html

    def test_extract_run_info_returns_none_when_no_metadata(self, temp_dir):
        """_extract_run_info should return None when no metadata file exists."""
        from gallery_index import _extract_run_info
        empty_run = temp_dir / "prompts" / "20240101_120000_empty"
        empty_run.mkdir(parents=True)

        result = _extract_run_info(empty_run, is_archive=False)
        assert result is None

    def test_extract_run_info_returns_none_for_corrupt_metadata(self, temp_dir):
        """_extract_run_info should return None for corrupt JSON metadata."""
        from gallery_index import _extract_run_info
        bad_run = temp_dir / "prompts" / "20240101_120000_bad"
        bad_run.mkdir(parents=True)
        (bad_run / "test.metaprompt.json").write_text("{not valid json")

        result = _extract_run_info(bad_run, is_archive=False)
        assert result is None

    def test_extract_run_info_returns_none_when_no_gallery(self, temp_dir):
        """_extract_run_info should return None if gallery file missing."""
        from gallery_index import _extract_run_info
        no_gallery = temp_dir / "prompts" / "20240101_120000_nogal"
        no_gallery.mkdir(parents=True)
        (no_gallery / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
        }))

        result = _extract_run_info(no_gallery, is_archive=False)
        assert result is None

    def test_build_card_html_truncates_long_prompt(self):
        """_build_card_html should truncate prompts longer than 100 chars."""
        from gallery_index import _build_card_html
        long_prompt = "x" * 150
        run = {
            "user_prompt": long_prompt,
            "display_time": "2024-01-01 12:00",
            "image_count": 5,
            "prompt_count": 3,
            "model": "test-model",
            "dir_name": "20240101_120000_xxx",
            "gallery_path": "prompts/20240101_120000_xxx/test_gallery.html",
            "thumbnail_file": None,
            "thumbnail": None,
        }
        html = _build_card_html(run, interactive=False)
        assert "...".encode() in html.encode("utf-8") or '...' in html
        # The full long prompt should NOT appear as visible text (only in title attribute)
        # Check that truncated prompt appears: first 100 chars + "..." = 103 chars
        expected_visible = f'{"x" * 100}...'
        assert expected_visible.encode() in html.encode("utf-8") or expected_visible in html

    def test_build_flat_archive_card_no_thumbnail_fallback(self):
        """_build_flat_archive_card_html should show no-thumbnail when no first_image."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "simple prompt",
            "display_time": "2024-01-01 12:00",
            "image_count": 3,
            "model": "test-model",
            "first_image": None,
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert "No images" in html

    def test_build_index_html_empty_state_shows_message(self):
        """_build_index_html should show empty-state message when no active runs."""
        from gallery_index import _build_index_html
        html = _build_index_html(active_runs=[])
        assert "No galleries found" in html

    def test_format_run_timestamp_produces_expected_output(self):
        """format_run_timestamp should produce a readable timestamp string."""
        from utils import format_run_timestamp
        result = format_run_timestamp("20240101_120000")
        assert isinstance(result, str)
        # Should not be the raw input unchanged
        assert result != "20240101_120000"

    def test_build_flat_archive_card_shows_saved_badge(self):
        """_build_flat_archive_card_html should render 'Saved' for manual_archive backup reason."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "manual save",
            "display_time": "2024-01-01 12:00",
            "image_count": 3,
            "model": "test-model",
            "first_image": None,
            "backup_reason": "manual_archive",
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert '<span class="archive-badge">Saved</span>' in html

    def test_build_flat_archive_card_shows_pre_enhance_badge(self):
        """_build_flat_archive_card_html should render 'Pre-Enhance' for pre_enhance backup reason."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "pre enhance",
            "display_time": "2024-01-01 12:00",
            "image_count": 3,
            "model": "test-model",
            "first_image": None,
            "backup_reason": "pre_enhance",
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert '<span class="archive-badge">Pre-Enhance</span>' in html

    def test_build_flat_archive_card_unknown_backup_reason(self):
        """_build_flat_archive_card_html should render 'Backup' for unrecognized backup reason."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "unknown",
            "display_time": "2024-01-01 12:00",
            "image_count": 3,
            "model": "test-model",
            "first_image": None,
            "backup_reason": "some_obscure_reason",
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert '<span class="archive-badge">Backup</span>' in html

    def test_extract_run_info_falls_back_to_nested_model(self, temp_dir):
        """_extract_run_info should fall back to image_generation.model when top-level model is absent."""
        from gallery_index import _extract_run_info
        active_run = temp_dir / "prompts" / "20240101_120000_xyz789"
        active_run.mkdir(parents=True)
        (active_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "fallback test prompt",
            "image_generation": {"model": "Flux-dev-v2"},
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        result = _extract_run_info(active_run, is_archive=False)
        assert result is not None
        assert result["model"] == "Flux-dev-v2"

    def test_extract_run_info_unknown_prompt_fallback(self, temp_dir):
        """_extract_run_info should return 'Unknown prompt' when neither display_title nor user_prompt is set."""
        from gallery_index import _extract_run_info
        active_run = temp_dir / "prompts" / "20240101_120000_unknown"
        active_run.mkdir(parents=True)
        (active_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        result = _extract_run_info(active_run, is_archive=False)
        assert result is not None
        assert result["user_prompt"] == "Unknown prompt"

    def test_build_index_html_empty_flat_archives_section(self):
        """_build_index_html should not render flat-archive section when there are none."""
        from gallery_index import _build_index_html
        html = _build_index_html(active_runs=[], archived_runs=[], flat_archives=[])
        assert "Archived Images" not in html

    def test_extract_run_info_timestamp_from_archive_dirname(self, temp_dir):
        """_extract_run_info should extract only the first two underscore segments as timestamp."""
        from gallery_index import _extract_run_info
        # Archive-style directory has 4+ underscore-separated components.
        archive_run = temp_dir / "saved" / "20240101_100000_def456_20240101_130000"
        archive_run.mkdir(parents=True)
        (archive_run / "test.metaprompt.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
        }))
        (archive_run / "test_gallery.html").write_text("<html></html>")

        result = _extract_run_info(archive_run, is_archive=True)
        assert result is not None
        # The timestamp must be only the first two segments — a change to the split
        # logic should not silently start consuming later path components.
        assert result["timestamp"] == "20240101_100000"

    def test_build_card_html_singular_plural_for_images(self):
        """Active card uses 'image' for 1, 'images' otherwise."""
        from gallery_index import _build_card_html
        run = {
            "user_prompt": "test",
            "display_time": "2024-01-01 12:00",
            "image_count": 1,
            "prompt_count": 3,
            "model": "test-model",
            "dir_name": "20240101_120000_xxx",
            "gallery_path": "prompts/20240101_120000_xxx/test_gallery.html",
            "thumbnail_file": None,
            "thumbnail": None,
        }
        html = _build_card_html(run, interactive=False)
        assert "1 image" in html and "| 3 prompts" in html

    def test_build_card_html_singular_plural_for_prompts(self):
        """Active card uses 'prompt' for 1, 'prompts' otherwise."""
        from gallery_index import _build_card_html
        run = {
            "user_prompt": "test",
            "display_time": "2024-01-01 12:00",
            "image_count": 5,
            "prompt_count": 1,
            "model": "test-model",
            "dir_name": "20240101_120000_xxx",
            "gallery_path": "prompts/20240101_120000_xxx/test_gallery.html",
            "thumbnail_file": None,
            "thumbnail": None,
        }
        html = _build_card_html(run, interactive=False)
        assert "5 images" in html and "| 1 prompt" in html

    def test_build_flat_archive_card_singular_plural(self):
        """Flat archive card uses 'image' for 1, 'images' otherwise."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "test",
            "display_time": "2024-01-01 12:00",
            "image_count": 1,
            "model": "test-model",
            "first_image": None,
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert "1 image" in html

    def test_build_flat_archive_card_plural(self):
        """Flat archive card uses 'images' for counts other than 1."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "test",
            "display_time": "2024-01-01 12:00",
            "image_count": 5,
            "model": "test-model",
            "first_image": None,
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert "5 images" in html and "images" in html

    def test_build_flat_archive_card_zero_images(self):
        """Flat archive card renders '0 images' (not '1 image')."""
        from gallery_index import _build_flat_archive_card_html
        archive = {
            "user_prompt": "test",
            "display_time": "2024-01-01 12:00",
            "image_count": 0,
            "model": "test-model",
            "first_image": None,
        }
        html = _build_flat_archive_card_html(archive, interactive=False)
        assert "0 images" in html
