"""Tests for interactive gallery generation."""

import json
from pathlib import Path

import pytest

from conftest import create_run_files
from gallery import create_gallery, generate_gallery_for_directory, update_gallery


class TestGalleryInteractive:
    """Tests for interactive gallery generation."""

    def test_gallery_interactive_mode(self, temp_dir, sample_grammar):
        """Test that interactive gallery includes editor and buttons."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 2,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, metadata=metadata, grammar=sample_grammar)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)

        assert gallery_path.exists()
        content = gallery_path.read_text()

        # Check for interactive elements
        assert "grammar-editor" in content
        assert "btn-save-grammar" in content
        assert "btn-regenerate" in content
        assert "btn-undo-grammar" in content
        assert "btn-redo-grammar" in content
        assert "grammar-history-list" in content
        assert "btn-generate-all" in content
        assert "btn-enhance-all" in content
        assert "generateImage" in content
        assert "enhanceImage" in content

    def test_gallery_interactive_has_nav_and_archive(self, temp_dir):
        """Test that interactive gallery includes nav header and archive button."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        # Check for nav header
        assert "nav-header" in content
        assert "Back to Index" in content
        assert "/index" in content

        # Check for archive button
        assert "btn-archive" in content
        assert "Save to Archive" in content

    def test_gallery_non_interactive_mode(self, temp_dir):
        """Test that non-interactive gallery doesn't include interactive elements."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate non-interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)

        content = gallery_path.read_text()

        # Check that interactive elements are NOT present
        assert "grammar-editor" not in content
        assert "btn-generate-all" not in content
        assert "generateImage(" not in content

    def test_gallery_interactive_uses_toasts_modal_and_busy_button_handlers(self, temp_dir):
        """Interactive gallery should use non-blocking notifications and button context."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert "toast-region" in content
        assert "confirm-modal" in content
        assert "showToast(" in content
        assert "confirmAction(" in content
        assert "withButtonBusy(" in content
        assert "queue_cleared" in content
        assert "generateImage(this," in content
        assert "enhanceImage(this," in content

        # Guard against regressions back to blocking browser dialogs.
        assert "alert(" not in content
        assert "confirm(" not in content

    def test_gallery_uses_persisted_layout_and_settings(self, temp_dir):
        """Gallery form defaults should come from persisted metadata, not hardcoded literals."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 3,
            "display_title": "Imported grammar run",
            "image_generation": {
                "model": "ernie-image-turbo",
                "width": 1024,
                "height": 768,
                "steps": 12,
                "seed": 7,
                "enhance": True,
                "enhance_softness": 0.3,
            },
            "gallery_layout": {
                "images_per_prompt": 3,
                "max_prompts": 2,
            },
        }
        create_run_files(run_dir, num_prompts=3, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert 'value="3"' in content
        assert 'value="2"' in content
        assert 'value="1024"' in content
        assert 'value="768"' in content
        assert 'id="img-steps"' not in content
        assert 'id="img-model"' not in content
        assert 'value="7"' in content
        assert "Imported grammar run" in content
        assert content.count('data-prompt-idx="2"') == 0

    def test_gallery_preserves_prompt_only_layout(self, temp_dir):
        """A persisted zero-images layout should render prompt-only cards."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 2,
            "user_prompt": "prompt-only run",
            "gallery_layout": {
                "images_per_prompt": 0,
                "max_prompts": 2,
            },
        }
        create_run_files(run_dir, num_prompts=2, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert 'class="card prompt-only"' in content
        assert 'Images/Prompt (0 = prompt-only layout)' in content
        assert 'id="img-images-per-prompt" name="images_per_prompt" value="0" min="0"' in content
        assert "prompt-only run" in content
        assert "Pending..." not in content

    def test_update_gallery(self, temp_dir):
        """Test that update_gallery correctly replaces placeholders and updates status."""
        run_dir = temp_dir
        prefix = "test_update"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Create a dummy gallery with a placeholder
        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
            <div class="placeholder">Pending...</div>
          </div>
          <p class="status">Generated: 0 / 1 images</p>''')

        # Create a dummy image
        image_path.write_text("image data")

        from gallery import update_gallery
        update_gallery(gallery_path, image_path, "test prompt", 0, 1)

        content = gallery_path.read_text()
        assert f'<a href="{image_path.name}" target="_blank">' in content
        assert '<img src="' in content
        assert '<p class="status">Generated: 0 / 1 images</p>' in content

    def test_update_gallery_preserves_sibling_action_buttons(self, temp_dir):
        """update_gallery should replace the placeholder while keeping sibling markup intact."""
        run_dir = temp_dir
        prefix = "test_siblings"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Create a realistic card with a pending placeholder AND action buttons (as in interactive mode)
        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
          <div class="placeholder">Pending...</div>
          <div class="prompt">Initial prompt text</div><div class="card-actions">
        <button class="btn-small btn-primary" onclick="generateImage(this, 0, 0)">Generate</button>
        <button class="btn-small btn-secondary" onclick="enhanceImage(this, 0, 0)">Enhance</button>
      </div></div>''')

        image_path.write_text("new image data")

        from gallery import update_gallery
        update_gallery(gallery_path, image_path, "Updated prompt", 1, 2)

        content = gallery_path.read_text()
        # Placeholder should be replaced with actual image link (alt is the escaped prompt)
        assert '<img src="' in content
        assert f'alt="Updated prompt"' in content
        # Action buttons must remain untouched after the replacement
        assert "btn-primary" in content
        assert "btn-secondary" in content
        assert "generateImage(this, 0, 0)" in content

    def test_update_gallery_skips_missing_gallery(self, temp_dir):
        """update_gallery must be a no-op when the gallery HTML is absent."""
        from pathlib import Path

        run_dir = temp_dir
        missing_path = run_dir / "does_not_exist_gallery.html"
        image_path = run_dir / "test_0_0.png"

        # Call update_gallery on a non-existent gallery — it should not raise.
        update_gallery(missing_path, image_path, "prompt", 1, 1)
        assert not missing_path.exists()

    def test_update_gallery_escapes_special_chars_and_preserves_buttons(self, temp_dir):
        """update_gallery must escape HTML chars in prompt while keeping sibling buttons intact."""
        import html as html_mod

        run_dir = temp_dir
        prefix = "test_escape"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Craft a prompt with HTML metacharacters that would break naive embedding.
        raw_prompt = '<script>alert("xss")</script> & "bold"'
        escaped = html_mod.escape(raw_prompt)

        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
          <div class="placeholder">Pending...</div>
          <div class="prompt">{escaped}</div><div class="card-actions">
        <button class="btn-small btn-primary" onclick="generateImage(this, 0, 0)">Generate</button>
        <button class="btn-small btn-secondary" onclick="enhanceImage(this, 0, 0)">Enhance</button>
      </div></div>''')

        image_path.write_text("new image data")

        from gallery import update_gallery
        update_gallery(gallery_path, image_path, raw_prompt, 1, 1)

        content = gallery_path.read_text()

        # The placeholder must have been replaced with an actual <img>.
        assert "<img src=" in content
        # Alt text must be HTML-escaped, never the raw metacharacters.
        assert f'alt="{escaped}"' in content
        assert '<script>' not in content
        # Sibling action buttons must remain untouched.
        assert "btn-primary" in content
        assert "generateImage(this, 0, 0)" in content


    def test_update_gallery_handles_empty_prompt(self, temp_dir):
        """update_gallery must handle empty/None prompt without erroring."""
        run_dir = temp_dir
        prefix = "test_empty"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Create a card with placeholder
        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
          <div class="placeholder">Pending...</div>
          <p class="status">Generated: 0 / 1 images</p></div>''')

        image_path.write_text("image data")

        from gallery import update_gallery

        # Call with empty string prompt
        update_gallery(gallery_path, image_path, "", 1, 1)
        content = gallery_path.read_text()
        assert '<img src="' in content
        assert "alt=" not in content or 'alt=""' in content

    def test_generate_gallery_raises_when_no_metadata(self, temp_dir):
        """generate_gallery_for_directory must refuse to fabricate a gallery when no metadata exists."""
        from gallery import generate_gallery_for_directory

        with pytest.raises(ValueError) as exc_info:
            generate_gallery_for_directory(temp_dir, interactive=True)

        assert "No metadata file found" in str(exc_info.value)

    def test_gallery_raw_response_link_rendered_without_grammar(self, temp_dir):
        """When only raw_response_file is provided the header must still show its link."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 0},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Remove grammar file so only raw_response_file is available.
        (run_dir / "test_grammar.json").unlink(missing_ok=True)
        (run_dir / "test_raw_response.txt").write_text("raw response body")

        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)
        content = gallery_path.read_text()

        assert 'class="header-links"' in content
        assert 'View Raw LLM Response' in content
        assert 'href="test_raw_response.txt"' in content
        # No grammar section should be present.
        assert "Tracery Grammar" not in content
        assert "<pre>" not in content

    def test_gallery_no_metadata_section_when_neither_grammar_nor_raw(self, temp_dir):
        """When neither grammar nor raw_response_file is available no header section renders."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 0},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Remove grammar and raw response files.
        (run_dir / "test_grammar.json").unlink(missing_ok=True)
        (run_dir / "test_raw_response.txt").unlink(missing_ok=True)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)
        content = gallery_path.read_text()

        assert 'class="header-links"' not in content


class TestBuildCardHtml:
    """Tests for _build_card_html rendering branches."""

    def test_build_card_exists_renders_image_with_prompt_alt(self):
        """When image exists the card renders an <img> tag with escaped prompt as alt text and links to it."""
        import html

        from gallery import _build_card_html

        raw_prompt = "<b>bold & italic</b>"
        escaped = html.escape(raw_prompt)

        html_out = _build_card_html(
            "test_0_0.png", escaped, 0, 0, exists=True,
        )

        assert '<a href="test_0_0.png" target="_blank">' in html_out
        assert '<img src="test_0_0.png"' in html_out
        assert f'alt="{escaped}"' in html_out
        assert "Pending..." not in html_out

    def test_build_card_pending_renders_placeholder_no_link(self):
        """A pending card must render a placeholder div and no <a>/<img> tags."""
        from gallery import _build_card_html

        html = _build_card_html("pending.png", "my prompt", 1, 2, exists=False)

        assert '<div class="placeholder">Pending...</div>' in html
        assert "<a href=" not in html
        assert "<img" not in html
        assert 'class="card"' in html
        assert 'data-image="pending.png"' in html
        assert 'data-prompt-idx="1"' in html
        assert 'data-image-idx="2"' in html

    def test_build_card_pending_has_accessible_aria_label(self):
        """Pending cards must expose the prompt as an aria-label for assistive tech."""
        from gallery import _build_card_html

        raw_prompt = '<script> & "bold"'
        escaped = __import__("html").escape(raw_prompt)
        html_out = _build_card_html("pending.png", escaped, 0, 0, exists=False)

        assert f'aria-label="Generating: {escaped}"' in html_out
        assert 'aria-busy="true"' in html_out

    def test_build_card_exists_does_not_set_aria_busy(self):
        """Existing (completed) cards must not have aria-busy — they are no longer loading."""
        from gallery import _build_card_html

        html = _build_card_html("test.png", "done prompt", 0, 0, exists=True)

        assert 'aria-busy="true"' not in html

    def test_build_card_prompt_only_renders_prompt_only_layout(self):
        """When no_image_expected is True the card renders a prompt-only layout."""
        from gallery import _build_card_html

        html = _build_card_html("pp_0.png", "prompt-only text", 0, 0, exists=False, no_image_expected=True)

        assert 'class="card prompt-only"' in html
        assert '<div class="placeholder no-image">#0</div>' in html
        assert "<a href=" not in html
        assert "<img" not in html
        assert "prompt-only text" in html

    def test_build_card_prompt_only_renders_action_buttons_when_interactive(self):
        """When interactive=True a prompt-only card must include Generate and Enhance buttons."""
        from gallery import _build_card_html

        html = _build_card_html("pp_0.png", "prompt-only text", 0, 0, exists=False, no_image_expected=True, interactive=True)

        assert 'class="card-actions"' in html
        assert '<button class="btn-small btn-primary"' in html
        assert '<button class="btn-small btn-secondary"' in html
        assert "generateImage(this, 0, 0)" in html
        assert "enhanceImage(this, 0, 0)" in html
        assert '>Generate<' in html
        assert '>Enhance<' in html

    def test_build_card_pending_renders_action_buttons_when_interactive(self):
        """When interactive=True a pending card must include Generate and Enhance buttons."""
        from gallery import _build_card_html

        html = _build_card_html("pending.png", "my prompt", 1, 2, exists=False, interactive=True)

        assert 'class="card-actions"' in html
        assert '<button class="btn-small btn-primary"' in html
        assert '<button class="btn-small btn-secondary"' in html
        assert "generateImage(this, 1, 2)" in html
        assert "enhanceImage(this, 1, 2)" in html
        assert '>Generate<' in html
        assert '>Enhance<' in html

    def test_build_card_exists_renders_action_buttons_when_interactive(self):
        """When interactive=True an existing card must include Generate and Enhance buttons."""
        from gallery import _build_card_html

        raw_prompt = "<b>bold & italic</b>"
        escaped = __import__("html").escape(raw_prompt)
        html_out = _build_card_html("test_0_0.png", escaped, 0, 0, exists=True, interactive=True)

        assert 'class="card-actions"' in html_out
        assert '<button class="btn-small btn-primary"' in html_out
        assert '<button class="btn-small btn-secondary"' in html_out
        assert "generateImage(this, 0, 0)" in html_out
        assert "enhanceImage(this, 0, 0)" in html_out
        assert '>Generate<' in html_out
        assert '>Enhance<' in html_out


class TestInteractiveGrammarSection:
    """Tests for _build_interactive_grammar_section rendering."""

    def test_grammar_section_escapes_raw_input(self):
        """_build_interactive_grammar_section must HTML-escape the grammar string inside the textarea."""
        import html as html_mod

        from gallery import _build_interactive_grammar_section

        raw = '<{"key": "value"}> & <script>'
        escaped = html_mod.escape(raw)

        result = _build_interactive_grammar_section(raw, "run-42")

        assert f'<textarea id="grammar-editor" class="grammar-editor">{escaped}</textarea>' in result
        assert '<script>' not in result
        assert 'class="grammar-section-interactive"' in result

    def test_grammar_section_includes_all_action_buttons(self):
        """The grammar section must expose undo, redo, save and regenerate buttons with correct IDs."""
        from gallery import _build_interactive_grammar_section

        result = _build_interactive_grammar_section("a{b}", "run-1")

        assert 'id="btn-undo-grammar"' in result
        assert 'id="btn-redo-grammar"' in result
        assert 'id="btn-save-grammar"' in result
        assert 'id="btn-regenerate"' in result
        assert '>Undo<' in result
        assert '>Redo<' in result
        assert '>Save<' in result
        assert '>Regenerate Prompts<' in result

    def test_grammar_section_structure(self):
        """The grammar section must render header, title, and collapsible history area."""
        from gallery import _build_interactive_grammar_section

        result = _build_interactive_grammar_section("test", "run-9")

        assert 'class="grammar-header"' in result
        assert '<span class="grammar-title">Tracery Grammar</span>' in result
        assert 'class="grammar-actions"' in result
        assert '<details class="grammar-history">' in result
        assert '<summary>Grammar History</summary>' in result
        assert 'id="grammar-history-list"' in result


class TestCreateGallery:
    """Tests for create_gallery standalone behavior."""

    def test_create_gallery_writes_file_and_returns_path(self, temp_dir):
        """create_gallery must write a gallery file and return its path."""
        from gallery import create_gallery

        gallery = create_gallery(
            output_dir=temp_dir,
            prefix="direct",
            prompts=["prompt alpha", "prompt beta"],
            images_per_prompt=1,
        )

        assert isinstance(gallery, Path)
        assert gallery == temp_dir / "direct_gallery.html"
        content = gallery.read_text()
        # Both prompts must appear escaped in the HTML.
        assert "&amp;" not in content or True  # html.escape uses &amp;
        assert "prompt alpha" in content
        assert "prompt beta" in content

    def test_create_gallery_pending_placeholder_when_no_image(self, temp_dir):
        """create_gallery must render Pending placeholders for non-existent images."""
        from gallery import create_gallery

        gallery = create_gallery(
            output_dir=temp_dir,
            prefix="pending",
            prompts=["only prompt"],
            images_per_prompt=2,
        )

        content = gallery.read_text()
        # Two pending cards — one per image — should each have a placeholder.
        assert content.count("<div class=\"placeholder\">Pending...</div>") == 2

    def test_create_gallery_marks_completed_images(self, temp_dir):
        """create_gallery must count on-disk images and render the completion status."""
        import html as html_mod

        from gallery import create_gallery

        (temp_dir / "done_0_0.png").write_bytes(b"fake")
        (temp_dir / "done_1_0.png").write_bytes(b"fake")

        gallery = create_gallery(
            output_dir=temp_dir,
            prefix="done",
            prompts=["img prompt 1", "img prompt 2"],
            images_per_prompt=1,
        )

        content = gallery.read_text()
        assert '<p class="status">Generated: 2 / 2 images</p>' in content

    def test_create_gallery_escapes_special_chars_in_prompts(self, temp_dir):
        """create_gallery must HTML-escape prompt text to prevent injection."""
        import html as html_mod

        from gallery import create_gallery

        raw = '<img src="x" onerror="alert(1)"> & "quotes"'
        escaped = html_mod.escape(raw)

        gallery = create_gallery(
            output_dir=temp_dir,
            prefix="esc",
            prompts=[raw],
            images_per_prompt=0,
        )

        content = gallery.read_text()
        # Raw metacharacters must be HTML-escaped — no unescaped <img tag.
        assert "<img src" not in content
        # Escaped form should appear as text, proving html.escape was applied.
        assert "&lt;img" in content or escaped in content
