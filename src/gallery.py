"""HTML gallery generation for image prompt outputs."""

import html
import json
import re
from pathlib import Path

from filelock import FileLock

from grammar_history import load_grammar_history
from html_components import (
    LogPanel,
    ProgressBar,
    Buttons,
    NavHeader,
    GalleryStyles,
    SSEClient,
    Notifications,
)
from metadata_manager import resolve_gallery_layout


def create_gallery(
    output_dir: Path,
    prefix: str,
    prompts: list[str],
    images_per_prompt: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
    interactive: bool = False,
    run_id: str | None = None,
    user_prompt: str = "",
    image_settings: dict | None = None,
    layout_settings: dict | None = None,
    grammar_history: list[dict] | None = None,
) -> Path:
    """Create initial gallery with placeholders for all expected images.

    Args:
        output_dir: Directory where gallery.html will be created
        prefix: Prefix used for image filenames
        prompts: List of prompt texts
        images_per_prompt: Number of images generated per prompt
        grammar: Optional Tracery grammar JSON to display
        raw_response_file: Optional filename for raw LLM response link
        interactive: If True, include editable grammar and action buttons
        run_id: Run ID for API calls (required if interactive)

    Returns:
        Path to the created gallery.html file
    """
    gallery_path = output_dir / f"{prefix}_gallery.html"
    total_images = len(prompts) * max(images_per_prompt, 1)

    # Build cards for all prompts
    cards_html = []
    for prompt_idx, prompt_text in enumerate(prompts):
        escaped_prompt = html.escape(prompt_text)

        if images_per_prompt == 0:
            # No images expected - just show prompt with generate button
            card = _build_card_html(
                f"{prefix}_{prompt_idx}_0.png", escaped_prompt, prompt_idx, 0,
                exists=False, interactive=interactive, no_image_expected=True
            )
            cards_html.append(card)
        else:
            for image_idx in range(images_per_prompt):
                image_filename = f"{prefix}_{prompt_idx}_{image_idx}.png"

                # Check if image already exists (for resume scenarios)
                image_path = output_dir / image_filename
                if image_path.exists():
                    card = _build_card_html(
                        image_filename, escaped_prompt, prompt_idx, image_idx,
                        exists=True, interactive=interactive
                    )
                else:
                    card = _build_card_html(
                        image_filename, escaped_prompt, prompt_idx, image_idx,
                        exists=False, interactive=interactive
                    )
                cards_html.append(card)

    # Count existing images
    completed = sum(1 for p_idx, p in enumerate(prompts) for i in range(max(images_per_prompt, 1))
                   if (output_dir / f"{prefix}_{p_idx}_{i}.png").exists())

    gallery_html = _build_gallery_html(
        prefix, cards_html, completed, total_images, grammar, raw_response_file,
        interactive=interactive,
        run_id=run_id,
        user_prompt=user_prompt,
        image_settings=image_settings or {},
        layout_settings=layout_settings or {},
        grammar_history=grammar_history or [],
    )
    gallery_path.write_text(gallery_html)

    return gallery_path


def _build_card_html(
    image_filename: str,
    escaped_prompt: str,
    prompt_idx: int,
    image_idx: int,
    exists: bool,
    interactive: bool = False,
    no_image_expected: bool = False,
) -> str:
    """Build HTML for a single image card."""
    action_buttons = ""
    if interactive:
        action_buttons = f'''
      <div class="card-actions">
        <button class="btn-small btn-primary" onclick="generateImage(this, {prompt_idx}, {image_idx})">Generate</button>
        <button class="btn-small btn-secondary" onclick="enhanceImage(this, {prompt_idx}, {image_idx})">Enhance</button>
      </div>'''

    if exists:
        # Use the prompt text as alt text for accessibility.
        # `escaped_prompt` is already HTML-escaped, so it's safe to embed directly.
        return f'''    <div class="card" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}">
      <a href="{image_filename}" target="_blank">
        <img src="{image_filename}" loading="lazy" alt="{escaped_prompt}">
      </a>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''
    elif no_image_expected:
        # Show prompt-only card with generate button
        return f'''    <div class="card prompt-only" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}">
      <div class="placeholder no-image">#{prompt_idx}</div>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''
    else:
        return f'''    <div class="card" data-image="{image_filename}" data-prompt-idx="{prompt_idx}" data-image-idx="{image_idx}" role="status" aria-label="Generating: {escaped_prompt}" aria-busy="true">
      <div class="placeholder">Pending...</div>
      <div class="prompt">{escaped_prompt}</div>{action_buttons}
    </div>'''


def update_gallery(
    gallery_path: Path,
    image_path: Path,
    prompt: str,
    completed: int,
    total: int,
) -> None:
    """Update gallery to show newly generated image.

    Args:
        gallery_path: Path to the gallery.html file
        image_path: Path to the newly generated image
        prompt: The prompt text for this image
        completed: Number of images completed so far
        total: Total number of images to generate
    """
    if not gallery_path.exists():
        return

    # Use file locking to prevent concurrent update corruption
    lock_path = gallery_path.with_suffix('.html.lock')
    with FileLock(lock_path, timeout=10):
        html_content = gallery_path.read_text()
        image_filename = image_path.name

        # Find the card for this image and replace placeholder with actual image
        # Pattern matches the placeholder div for this specific image
        placeholder_pattern = (
            rf'(<div class="card" data-image="{re.escape(image_filename)}"[^>]*>)\s*'
            rf'<div class="placeholder">Pending\.\.\.</div>'
        )

        escaped_prompt = html.escape(prompt) if prompt else ""
        replacement = (
            rf'\1\n      <a href="{image_filename}" target="_blank">\n'
            rf'        <img src="{image_filename}" loading="lazy" alt="{escaped_prompt}">\n'
            rf'      </a>'
        )

        html_content = re.sub(placeholder_pattern, replacement, html_content)

        # Update the status count
        status_pattern = r'<p class="status">Generated: \d+ / \d+ images</p>'
        status_replacement = f'<p class="status">Generated: {completed} / {total} images</p>'
        html_content = re.sub(status_pattern, status_replacement, html_content)

        gallery_path.write_text(html_content)


def generate_gallery_for_directory(prompts_dir: Path, interactive: bool = False) -> Path:
    """Generate a gallery for an existing prompts directory.

    Args:
        prompts_dir: Directory containing prompt files and images
        interactive: If True, include editable grammar and action buttons

    Returns:
        Path to the created gallery.html file

    Raises:
        ValueError: If no metadata file found or no prompts found
    """
    # Find metadata file
    meta_files = []
    for pattern in ["*.metaprompt.json", "*_metadata.json"]:
        meta_files = list(prompts_dir.glob(pattern))
        if meta_files:
            break
    if not meta_files:
        raise ValueError(f"No metadata file found in {prompts_dir}")

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    user_prompt = metadata.get("display_title") or metadata.get("user_prompt", "")

    # Load prompts
    prompt_files = sorted(prompts_dir.glob(f"{prefix}_*.txt"))
    prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

    if not prompt_files:
        raise ValueError(f"No prompt files found in {prompts_dir}")

    prompts = [f.read_text() for f in prompt_files]
    layout_settings = resolve_gallery_layout(metadata, prompt_count=len(prompts))
    images_per_prompt = layout_settings["images_per_prompt"]
    prompts_to_render = prompts[:layout_settings["max_prompts"]] if layout_settings["max_prompts"] else prompts

    # Load grammar if available
    grammar = None
    grammar_file = prompts_dir / f"{prefix}_grammar.json"
    if grammar_file.exists():
        grammar = grammar_file.read_text()

    # Check for raw response file
    raw_response_file = None
    raw_file = prompts_dir / f"{prefix}_raw_response.txt"
    if raw_file.exists():
        raw_response_file = f"{prefix}_raw_response.txt"

    # Get run_id from directory name
    run_id = prompts_dir.name if interactive else None
    grammar_history = load_grammar_history(prompts_dir, prefix, current_grammar=grammar)
    image_settings = metadata.get("image_generation", {}) or {}
    if "model" not in image_settings and metadata.get("model"):
        image_settings = {
            "model": metadata.get("model"),
            **image_settings,
        }

    # Create gallery
    gallery_path = create_gallery(
        prompts_dir,
        prefix,
        prompts_to_render,
        images_per_prompt,
        grammar,
        raw_response_file,
        interactive=interactive,
        run_id=run_id,
        user_prompt=user_prompt,
        image_settings=image_settings,
        layout_settings=layout_settings,
        grammar_history=grammar_history,
    )

    return gallery_path


def _build_nav_header() -> str:
    """Build navigation header with back link."""
    return NavHeader.html()


def _build_interactive_grammar_section(grammar: str, run_id: str) -> str:
    """Build the interactive grammar section with edit capabilities."""
    escaped_grammar = html.escape(grammar)
    return f'''
  <div class="grammar-section-interactive">
    <div class="grammar-header">
      <span class="grammar-title">Tracery Grammar</span>
      <div class="grammar-actions">
        <button id="btn-undo-grammar" class="btn-small btn-secondary">Undo</button>
        <button id="btn-redo-grammar" class="btn-small btn-secondary">Redo</button>
        <button id="btn-save-grammar" class="btn-small btn-primary">Save</button>
        <button id="btn-regenerate" class="btn-small btn-secondary">Regenerate Prompts</button>
      </div>
    </div>
    <textarea id="grammar-editor" class="grammar-editor">{escaped_grammar}</textarea>
    <details class="grammar-history">
      <summary>Grammar History</summary>
      <div id="grammar-history-list" class="grammar-history-list"></div>
    </details>
  </div>
'''


def _build_image_settings_section(image_settings: dict, layout_settings: dict) -> str:
    """Build the collapsible image settings section."""
    width = image_settings.get("width", 864)
    height = image_settings.get("height", 1152)
    seed = image_settings.get("seed")
    enhance = "checked" if image_settings.get("enhance", False) else ""
    enhance_softness = image_settings.get("enhance_softness", 0.5) or 0.5
    images_per_prompt = layout_settings.get("images_per_prompt", 1)
    max_prompts = layout_settings.get("max_prompts")
    return f'''
  <details id="image-settings" class="settings-section" open>
    <summary>Image Settings</summary>
    <div class="settings-form">
      <div class="form-row">
        <div class="form-group">
          <label for="img-images-per-prompt">Images/Prompt (0 = prompt-only layout)</label>
          <input type="number" id="img-images-per-prompt" name="images_per_prompt" value="{images_per_prompt}" min="0">
        </div>
        <div class="form-group">
          <label for="img-max-prompts">Max Prompts</label>
          <input type="number" id="img-max-prompts" name="max_prompts" value="{'' if max_prompts is None else max_prompts}" min="1" placeholder="all">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="img-width">Width</label>
          <input type="number" id="img-width" name="width" value="{width}" step="8">
        </div>
        <div class="form-group">
          <label for="img-height">Height</label>
          <input type="number" id="img-height" name="height" value="{height}" step="8">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="img-seed">Seed</label>
          <input type="number" id="img-seed" name="seed" value="{'' if seed is None else seed}" placeholder="random">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group checkbox-group">
          <label>
            <input type="checkbox" id="img-enhance" name="enhance" {enhance}>
            Enhance after generation
          </label>
        </div>
        <div class="form-group">
          <label for="img-enhance-softness">Softness</label>
          <input type="number" id="img-enhance-softness" name="enhance_softness" value="{enhance_softness}" step="0.1" min="0" max="1">
        </div>
      </div>
    </div>
  </details>
'''


def _build_interactive_action_bar(run_id: str) -> str:
    """Build the action bar with generate/enhance all buttons."""
    return f'''
  <div class="action-bar">
    <button id="btn-generate-all" class="btn-primary">Generate All Images</button>
    <button id="btn-enhance-all" class="btn-secondary">Enhance All</button>
    <button id="btn-archive" class="btn-secondary">Save to Archive</button>
    <div class="action-spacer"></div>
    <button id="btn-clear-queue" class="btn-secondary">Clear Queue</button>
    <button id="btn-kill" class="btn-danger">Kill Current</button>
  </div>
'''


def _build_interactive_progress_bar() -> str:
    """Build the fixed progress bar at bottom."""
    return ProgressBar.html()


def _build_log_panel() -> str:
    """Build the collapsible log panel HTML."""
    return LogPanel.html()


def _build_interactive_js(run_id: str, grammar_history: list[dict]) -> str:
    """Build JavaScript for interactive gallery features."""
    log_js = LogPanel.js()
    sse_js = SSEClient.js()
    notify_js = Notifications.js()
    history_json = json.dumps(grammar_history)

    return f'''
<script>
(function() {{
  const RUN_ID = "{run_id}";
  const initialGrammarHistory = {history_json};
  const grammarEditor = document.getElementById('grammar-editor');
  const btnUndoGrammar = document.getElementById('btn-undo-grammar');

  function readImagesPerPrompt() {{
    const value = document.getElementById('img-images-per-prompt')?.value;
    return value === '' ? 1 : parseInt(value);
  }}
  const btnRedoGrammar = document.getElementById('btn-redo-grammar');
  const btnSaveGrammar = document.getElementById('btn-save-grammar');
  const btnRegenerate = document.getElementById('btn-regenerate');
  const btnGenerateAll = document.getElementById('btn-generate-all');
  const btnEnhanceAll = document.getElementById('btn-enhance-all');
  const btnClearQueue = document.getElementById('btn-clear-queue');
  const btnKill = document.getElementById('btn-kill');
  const progressBar = document.getElementById('progress-bar');
  const progressMessage = document.getElementById('progress-message');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const logPanel = document.getElementById('log-panel');
  const grammarHistoryList = document.getElementById('grammar-history-list');
  const grammarDraftKey = `grammar-draft:${{RUN_ID}}`;
  const layoutInputs = [
    document.getElementById('img-images-per-prompt'),
    document.getElementById('img-max-prompts'),
  ].filter(Boolean);
  let undoStack = grammarEditor ? [grammarEditor.value] : [];
  let redoStack = [];
  let grammarSnapshotTimer = null;
  let layoutSaveTimer = null;
  let suppressSnapshot = false;

  // Shared notification helpers
{notify_js}

  // Shared log panel functions
{log_js}

  // Shared SSE connection logic
{sse_js}

  async function withButtonBusy(btn, busyText, fn) {{
    if (!btn) return fn();
    const original = btn.dataset.originalText || btn.textContent;
    btn.dataset.originalText = original;
    btn.disabled = true;
    btn.textContent = busyText;
    try {{
      return await fn();
    }} finally {{
      btn.disabled = false;
      btn.textContent = original;
    }}
  }}

  function pushUndoSnapshot(value) {{
    if (!grammarEditor || suppressSnapshot) return;
    if (undoStack[undoStack.length - 1] === value) return;
    undoStack.push(value);
    if (undoStack.length > 100) undoStack.shift();
    redoStack = [];
    syncUndoButtons();
  }}

  function syncUndoButtons() {{
    if (btnUndoGrammar) btnUndoGrammar.disabled = undoStack.length <= 1;
    if (btnRedoGrammar) btnRedoGrammar.disabled = redoStack.length === 0;
  }}

  function applyGrammarValue(value) {{
    if (!grammarEditor) return;
    suppressSnapshot = true;
    grammarEditor.value = value;
    suppressSnapshot = false;
  }}

  function renderGrammarHistory(history) {{
    if (!grammarHistoryList) return;
    if (!history.length) {{
      grammarHistoryList.innerHTML = '<p class="history-empty">No saved revisions yet.</p>';
      return;
    }}
    grammarHistoryList.innerHTML = history.slice().reverse().map((entry) => {{
      const when = new Date(entry.created_at).toLocaleString();
      const action = entry.action || 'saved';
      const preview = (entry.grammar || '').split('\\n')[0].slice(0, 80);
      return `
        <button type="button" class="history-item" data-grammar-id="${{entry.id}}">
          <span class="history-item-meta">${{action}} · ${{when}}</span>
          <span class="history-item-preview">${{preview || '(empty grammar)'}} </span>
        </button>
      `;
    }}).join('');

    grammarHistoryList.querySelectorAll('.history-item').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        const match = history.find((entry) => entry.id === btn.dataset.grammarId);
        if (!match) return;
        applyGrammarValue(match.grammar || '');
        pushUndoSnapshot(grammarEditor.value);
        localStorage.setItem(grammarDraftKey, grammarEditor.value);
        showToast('Loaded revision into editor', 'success');
      }});
    }});
  }}

  async function refreshGrammarHistory() {{
    try {{
      const resp = await fetch(`/api/gallery/${{RUN_ID}}/grammar/history`);
      if (!resp.ok) return;
      const data = await resp.json();
      renderGrammarHistory(data.history || []);
    }} catch (_err) {{
      // Best effort only.
    }}
  }}

  async function persistLayoutAndReload() {{
    const imagesPerPrompt = readImagesPerPrompt();
    const maxPromptsValue = document.getElementById('img-max-prompts')?.value;
    const maxPrompts = maxPromptsValue ? parseInt(maxPromptsValue) : null;
    await apiPut(`/api/gallery/${{RUN_ID}}/layout`, {{
      images_per_prompt: imagesPerPrompt,
      max_prompts: maxPrompts,
    }});
    window.location.reload();
  }}

  function initSSE() {{
    const es = connectSSE();
    if (!es) return;

    es.addEventListener('status', (e) => {{
      const data = JSON.parse(e.data);
      if (data.current) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `Running: ${{data.current.type}}`;
        if (data.current.progress) {{
          const p = data.current.progress;
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          progressFill.style.width = pct + '%';
          progressText.textContent = `${{p.current}}/${{p.total}}`;
          if (p.message) progressMessage.textContent = p.message;
        }}
      }}
    }});

    es.addEventListener('task_started', (e) => {{
      progressBar.classList.remove('hidden');
      const task = JSON.parse(e.data);
      progressMessage.textContent = `Running: ${{task.type}}`;
    }});

    es.addEventListener('task_progress', (e) => {{
      const data = JSON.parse(e.data);
      const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
      progressFill.style.width = pct + '%';
      progressText.textContent = `${{data.current}}/${{data.total}}`;
      if (data.message) progressMessage.textContent = data.message;
    }});

    es.addEventListener('task_completed', (e) => {{
      const data = JSON.parse(e.data);
      progressMessage.textContent = 'Completed';
      // Reload page if this was a regenerate_prompts task for this gallery
      if (data.result && data.result.task_type === 'regenerate_prompts' && data.result.run_id === RUN_ID) {{
        setTimeout(() => window.location.reload(), 500);
      }}
    }});

    es.addEventListener('task_failed', (e) => {{
      const data = JSON.parse(e.data);
      progressMessage.textContent = 'Failed: ' + data.error;
    }});

    es.addEventListener('task_cancelled', (e) => {{
      progressMessage.textContent = 'Cancelled';
    }});

    es.addEventListener('queue_cleared', (e) => {{
      progressBar.classList.add('hidden');
      progressFill.style.width = '0%';
      progressText.textContent = '';
      showToast('Queue cleared', 'success');
    }});

    es.addEventListener('queue_updated', (e) => {{
      const data = JSON.parse(e.data);
      if (data.current) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `Running: ${{data.current.type}}`;
        if (data.current.progress) {{
          const p = data.current.progress;
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          progressFill.style.width = pct + '%';
          progressText.textContent = `${{p.current}}/${{p.total}}`;
          if (p.message) progressMessage.textContent = p.message;
        }}
      }} else if (data.pending_count > 0) {{
        progressBar.classList.remove('hidden');
        progressMessage.textContent = `${{data.pending_count}} task(s) pending...`;
        progressFill.style.width = '0%';
        progressText.textContent = '';
      }} else {{
        // No current task and no pending - hide after brief delay
        setTimeout(() => {{
          progressBar.classList.add('hidden');
          progressFill.style.width = '0%';
        }}, 1500);
      }}
    }});

    es.addEventListener('image_ready', (e) => {{
      const data = JSON.parse(e.data);
      if (data.run_id === RUN_ID) {{
        updateImage(data.path);
      }}
    }});

    es.addEventListener('task_log', (e) => {{
      const data = JSON.parse(e.data);
      appendLog(data.timestamp, data.message);
      // Auto-open log panel when logs arrive
      if (logPanel && !logPanel.open) {{
        logPanel.open = true;
      }}
    }});
  }}

  function updateImage(filename) {{
    const card = document.querySelector(`[data-image="${{filename}}"]`);
    if (!card) return;

    const placeholder = card.querySelector('.placeholder');
    if (placeholder) {{
      const link = document.createElement('a');
      link.href = filename;
      link.target = '_blank';
      const img = document.createElement('img');
      img.src = filename + '?t=' + Date.now();
      img.loading = 'lazy';
      link.appendChild(img);
      placeholder.replaceWith(link);
    }} else {{
      const img = card.querySelector('img');
      if (img) img.src = filename + '?t=' + Date.now();
    }}
  }}

  // API helpers
  async function apiPost(url, body = {{}}) {{
    try {{
      const resp = await fetch(url, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body),
      }});
      if (!resp.ok) {{
        let detail = 'Request failed';
        try {{
          const err = await resp.json();
          detail = err.detail || detail;
        }} catch (_e) {{}}
        throw new Error(detail);
      }}
      return await resp.json();
    }} catch (err) {{
      showToast('Error: ' + err.message, 'error', 4200);
      throw err;
    }}
  }}

  async function apiPut(url, body = {{}}) {{
    try {{
      const resp = await fetch(url, {{
        method: 'PUT',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body),
      }});
      if (!resp.ok) {{
        let detail = 'Request failed';
        try {{
          const err = await resp.json();
          detail = err.detail || detail;
        }} catch (_e) {{}}
        throw new Error(detail);
      }}
      return await resp.json();
    }} catch (err) {{
      showToast('Error: ' + err.message, 'error', 4200);
      throw err;
    }}
  }}

  // Button handlers
  if (btnSaveGrammar) {{
    btnSaveGrammar.addEventListener('click', async () => {{
      await withButtonBusy(btnSaveGrammar, 'Saving...', async () => {{
        await apiPut(`/api/gallery/${{RUN_ID}}/grammar`, {{
          grammar: grammarEditor.value
        }});
      }});
      localStorage.removeItem(grammarDraftKey);
      showToast('Grammar saved', 'success');
      refreshGrammarHistory();
    }});
  }}

  if (btnRegenerate) {{
    btnRegenerate.addEventListener('click', async () => {{
      await withButtonBusy(btnRegenerate, 'Queueing...', async () => {{
        await apiPost(`/api/gallery/${{RUN_ID}}/regenerate`, {{
          grammar: grammarEditor.value,
          images_per_prompt: readImagesPerPrompt(),
          max_prompts: document.getElementById('img-max-prompts')?.value ? parseInt(document.getElementById('img-max-prompts').value) : null,
        }});
      }});
      localStorage.removeItem(grammarDraftKey);
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Regenerating prompts...';
      showToast('Prompt regeneration queued', 'success');
    }});
  }}

  if (btnGenerateAll) {{
    btnGenerateAll.addEventListener('click', async () => {{
      const data = {{
        images_per_prompt: readImagesPerPrompt(),
        resume: true,
        width: parseInt(document.getElementById('img-width')?.value) || null,
        height: parseInt(document.getElementById('img-height')?.value) || null,
        seed: document.getElementById('img-seed')?.value ? parseInt(document.getElementById('img-seed').value) : null,
        max_prompts: document.getElementById('img-max-prompts')?.value ? parseInt(document.getElementById('img-max-prompts').value) : null,
        enhance: document.getElementById('img-enhance')?.checked || false,
        enhance_softness: parseFloat(document.getElementById('img-enhance-softness')?.value) || 0.5,
      }};
      await withButtonBusy(btnGenerateAll, 'Queueing...', async () => {{
        await apiPost(`/api/gallery/${{RUN_ID}}/generate-all`, data);
      }});
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Queued image generation...';
      showToast('Image generation queued', 'success');
    }});
  }}

  if (btnEnhanceAll) {{
    btnEnhanceAll.addEventListener('click', async () => {{
      const softness = parseFloat(document.getElementById('img-enhance-softness')?.value) || 0.5;
      await withButtonBusy(btnEnhanceAll, 'Queueing...', async () => {{
        await apiPost(`/api/gallery/${{RUN_ID}}/enhance-all`, {{
          softness: softness
        }});
      }});
      progressBar.classList.remove('hidden');
      progressMessage.textContent = 'Queued enhancement...';
      showToast('Enhancement queued', 'success');
    }});
  }}

  if (btnClearQueue) {{
    btnClearQueue.addEventListener('click', async () => {{
      await withButtonBusy(btnClearQueue, 'Clearing...', async () => {{
        await apiPost('/api/queue/clear');
      }});
      showToast('Queue clear requested', 'success');
    }});
  }}

  if (btnKill) {{
    btnKill.addEventListener('click', async () => {{
      await withButtonBusy(btnKill, 'Killing...', async () => {{
        await apiPost('/api/worker/kill');
      }});
      showToast('Kill signal sent', 'success');
    }});
  }}

  const btnArchive = document.getElementById('btn-archive');
  if (btnArchive) {{
    btnArchive.addEventListener('click', async () => {{
      const confirmed = await confirmAction('Save this gallery to archive?', {{
        confirmText: 'Archive',
        cancelText: 'Cancel'
      }});
      if (!confirmed) return;
      await withButtonBusy(btnArchive, 'Archiving...', async () => {{
        const resp = await apiPost(`/api/gallery/${{RUN_ID}}/archive`);
        showToast(resp.message || 'Archived successfully', 'success');
      }});
    }});
  }}

  // Global functions for per-image buttons
  window.generateImage = async function(btn, promptIdx, imageIdx) {{
    await withButtonBusy(btn, 'Queueing...', async () => {{
      await apiPost(`/api/gallery/${{RUN_ID}}/image/${{promptIdx}}/generate`, {{
        image_idx: imageIdx
      }});
    }});
    progressBar.classList.remove('hidden');
    progressMessage.textContent = `Generating image ${{promptIdx}}_${{imageIdx}}...`;
    showToast(`Queued image ${{promptIdx}}_${{imageIdx}}`, 'success');
  }};

  window.enhanceImage = async function(btn, promptIdx, imageIdx) {{
    const softness = parseFloat(document.getElementById('img-enhance-softness')?.value) || 0.5;
    await withButtonBusy(btn, 'Queueing...', async () => {{
      await apiPost(`/api/gallery/${{RUN_ID}}/image/${{promptIdx}}/enhance`, {{
        image_idx: imageIdx,
        softness: softness
      }});
    }});
    progressBar.classList.remove('hidden');
    progressMessage.textContent = `Enhancing image ${{promptIdx}}_${{imageIdx}}...`;
    showToast(`Queued enhancement for ${{promptIdx}}_${{imageIdx}}`, 'success');
  }};

  if (grammarEditor) {{
    const savedDraft = localStorage.getItem(grammarDraftKey);
    if (savedDraft && savedDraft !== grammarEditor.value) {{
      applyGrammarValue(savedDraft);
      undoStack = [savedDraft];
    }}
    grammarEditor.addEventListener('input', () => {{
      localStorage.setItem(grammarDraftKey, grammarEditor.value);
      clearTimeout(grammarSnapshotTimer);
      grammarSnapshotTimer = setTimeout(() => pushUndoSnapshot(grammarEditor.value), 300);
    }});
  }}

  if (btnUndoGrammar) {{
    btnUndoGrammar.addEventListener('click', () => {{
      if (undoStack.length <= 1 || !grammarEditor) return;
      const current = undoStack.pop();
      redoStack.push(current);
      applyGrammarValue(undoStack[undoStack.length - 1]);
      syncUndoButtons();
    }});
  }}

  if (btnRedoGrammar) {{
    btnRedoGrammar.addEventListener('click', () => {{
      if (!redoStack.length || !grammarEditor) return;
      const next = redoStack.pop();
      applyGrammarValue(next);
      undoStack.push(next);
      syncUndoButtons();
    }});
  }}

  layoutInputs.forEach((input) => {{
    input.addEventListener('change', () => {{
      clearTimeout(layoutSaveTimer);
      layoutSaveTimer = setTimeout(() => {{
        persistLayoutAndReload().catch((err) => {{
          showToast('Error: ' + err.message, 'error', 4200);
        }});
      }}, 150);
    }});
  }});

  // Start SSE
  renderGrammarHistory(initialGrammarHistory);
  syncUndoButtons();
  initSSE();
}})();
</script>
'''


def _build_interactive_styles() -> str:
    """Build additional CSS for interactive gallery."""
    return (
        NavHeader.css() +
        GalleryStyles.css() +
        Buttons.css() +
        Notifications.css() +
        ProgressBar.css() +
        '''
    .progress-container { display: flex; align-items: center; gap: 12px; }
    .progress-fill { height: 100%; background: #4a9eff; transition: width 0.3s; }

    /* Adjust body padding */
    body { padding-bottom: 80px; }

    /* Image settings form */
    .settings-section { margin-bottom: 16px; background: #2a2a2a; border-radius: 8px; }
    .settings-section summary { padding: 12px 16px; cursor: pointer; color: #888; font-size: 14px; }
    .settings-section summary:hover { color: #aaa; }
    .settings-form { padding: 16px; }
    .form-row { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
    .form-row:last-child { margin-bottom: 0; }
    .form-group { display: flex; flex-direction: column; min-width: 120px; }
    .form-group label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .form-group input, .form-group select { padding: 8px; background: #333; border: 1px solid #444; border-radius: 4px; color: #fff; font-size: 14px; }
    .form-group input:focus, .form-group select:focus { border-color: #4a9eff; outline: none; }
    .checkbox-group { flex-direction: row; align-items: center; }
    .checkbox-group label { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #ddd; margin-bottom: 0; }
    .checkbox-group input[type="checkbox"] { width: 16px; height: 16px; }
    .grammar-history { margin-top: 12px; background: #222; border-radius: 6px; }
    .grammar-history summary { padding: 10px 12px; cursor: pointer; color: #888; font-size: 13px; }
    .grammar-history-list { display: flex; flex-direction: column; gap: 8px; padding: 0 12px 12px; }
    .history-item { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; width: 100%; padding: 10px 12px; background: #2c2c2c; border: 1px solid #444; border-radius: 6px; color: #ddd; text-align: left; cursor: pointer; }
    .history-item:hover { border-color: #4a9eff; }
    .history-item-meta { font-size: 12px; color: #8aa; }
    .history-item-preview { font-size: 12px; color: #bbb; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; width: 100%; }
    .history-empty { margin: 0; padding: 0 12px 12px; color: #666; font-size: 13px; }
''' +
        LogPanel.css()
    )


def _build_gallery_html(
    prefix: str,
    cards_html: list[str],
    completed: int,
    total: int,
    grammar: str | None = None,
    raw_response_file: str | None = None,
    interactive: bool = False,
    run_id: str | None = None,
    user_prompt: str = "",
    image_settings: dict | None = None,
    layout_settings: dict | None = None,
    grammar_history: list[dict] | None = None,
) -> str:
    """Build the complete gallery HTML document."""
    cards_joined = "\n".join(cards_html)
    image_settings = image_settings or {}
    layout_settings = layout_settings or {}
    grammar_history = grammar_history or []

    # Base tag for interactive galleries to resolve relative URLs correctly
    base_tag = f'<base href="/gallery/{run_id}/">' if interactive and run_id else ""

    # Build original prompt display
    prompt_display = ""
    if user_prompt:
        escaped_user_prompt = html.escape(user_prompt)
        prompt_display = f'\n  <p class="original-prompt">{escaped_user_prompt}</p>'

    # Build header section with optional grammar display and raw response link
    header_section = ""
    if grammar or raw_response_file:
        header_parts = []
        if raw_response_file:
            header_parts.append(f'<a href="{raw_response_file}" class="raw-link">View Raw LLM Response</a>')
        header_section = f'''
  <div class="header-links">
    {" ".join(header_parts)}
  </div>'''

    # Build grammar section
    grammar_section = ""
    if grammar:
        if interactive and run_id:
            grammar_section = _build_interactive_grammar_section(grammar, run_id)
        else:
            escaped_grammar = html.escape(grammar)
            grammar_section = f'''
  <details class="grammar-section">
    <summary>Tracery Grammar</summary>
    <pre>{escaped_grammar}</pre>
  </details>'''

    # Build interactive sections
    nav_header = _build_nav_header() if interactive else ""
    notifications = Notifications.html() if interactive else ""
    image_settings_html = (
        _build_image_settings_section(image_settings, layout_settings)
        if interactive and run_id else ""
    )
    action_bar = _build_interactive_action_bar(run_id) if interactive and run_id else ""
    log_panel = _build_log_panel() if interactive else ""
    progress_bar = _build_interactive_progress_bar() if interactive else ""
    interactive_js = (
        _build_interactive_js(run_id, grammar_history)
        if interactive and run_id else ""
    )
    extra_styles = _build_interactive_styles() if interactive else ""

    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Gallery: {prefix}</title>
  <meta charset="utf-8">
  {base_tag}
  <style>
    body {{ font-family: system-ui; padding: 20px; background: #1a1a1a; color: #fff; }}
    h1 {{ margin-bottom: 10px; }}
    .original-prompt {{ color: #888; font-size: 14px; margin-bottom: 16px; font-style: italic; }}
    .header-links {{ margin-bottom: 15px; }}
    .header-links a {{ color: #6af; text-decoration: none; margin-right: 20px; }}
    .header-links a:hover {{ text-decoration: underline; }}
    .grammar-section {{ margin-bottom: 20px; background: #2a2a2a; border-radius: 8px; }}
    .grammar-section summary {{ padding: 12px 16px; cursor: pointer; color: #888; font-size: 14px; }}
    .grammar-section summary:hover {{ color: #aaa; }}
    .grammar-section pre {{ margin: 0; padding: 16px; background: #222; font-size: 12px; color: #8f8; overflow-x: auto; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
    .status {{ color: #888; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; aspect-ratio: 3/4; object-fit: cover; cursor: pointer; }}
    .card .placeholder {{ width: 100%; aspect-ratio: 3/4; background: #333; display: flex; align-items: center; justify-content: center; color: #666; }}
    .card .placeholder.no-image {{ aspect-ratio: 1/1; background: #252525; color: #555; font-size: 24px; font-weight: bold; }}
    .card.prompt-only {{ border: 1px dashed #444; }}
    .card .prompt {{ padding: 12px; font-size: 13px; color: #aaa; max-height: 150px; overflow-y: auto; }}
{extra_styles}
  </style>
</head>
<body>{notifications}{nav_header}
  <h1>Gallery: {prefix}</h1>{prompt_display}{header_section}{grammar_section}{image_settings_html}{action_bar}{log_panel}
  <p class="status">Generated: {completed} / {total} images</p>
  <div class="grid">
{cards_joined}
  </div>
{progress_bar}
{interactive_js}
</body>
</html>
'''
