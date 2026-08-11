"""Master gallery index generation for all prompt runs."""

import html
import json
import re
from pathlib import Path

from utils import scan_flat_archives, get_flat_archive_metadata, format_run_timestamp
from html_components import (
    LogPanel,
    QueueStatusBar,
    Buttons,
    FormStyles,
    IndexStyles,
    SSEClient,
    Notifications,
)


def generate_master_index(generated_dir: Path, interactive: bool = False) -> Path:
    """
    Scan generated/prompts/ for all run directories and create
    a master index.html linking to all gallery.html files.

    Args:
        generated_dir: Path to the generated/ directory
        interactive: If True, include generation form and SSE support

    Returns:
        Path to the created index.html file
    """
    prompts_dir = generated_dir / "prompts"
    saved_dir = generated_dir / "saved"
    index_path = generated_dir / "index.html"

    # Scan active runs
    active_runs = []
    if prompts_dir.exists():
        for run_dir in prompts_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if directory name matches timestamp pattern
            if not re.match(r'^\d{8}_\d{6}_', run_dir.name):
                continue

            run_info = _extract_run_info(run_dir, is_archive=False)
            if run_info:
                active_runs.append(run_info)

    # Scan archived runs (directory-based - legacy)
    archived_runs = []
    if saved_dir.exists():
        for archive_dir in saved_dir.iterdir():
            if not archive_dir.is_dir():
                continue

            run_info = _extract_run_info(archive_dir, is_archive=True)
            if run_info:
                archived_runs.append(run_info)

    # Scan flat archives (new format)
    flat_archives = []
    if saved_dir.exists():
        flat_archives = _extract_flat_archive_infos(saved_dir, interactive=interactive)

    # Sort by timestamp (newest first)
    active_runs.sort(key=lambda x: x["timestamp"], reverse=True)
    archived_runs.sort(key=lambda x: x["timestamp"], reverse=True)
    flat_archives.sort(key=lambda x: x["timestamp"], reverse=True)

    # Generate index HTML
    index_html = _build_index_html(
        active_runs, archived_runs, flat_archives, interactive=interactive
    )
    index_path.write_text(index_html)

    return index_path


def _extract_run_info(run_dir: Path, is_archive: bool = False) -> dict | None:
    """
    Extract information about a run directory.

    Args:
        run_dir: Path to a run directory
        is_archive: Whether this is an archived run

    Returns:
        Dictionary with run info, or None if not a valid run
    """
    # Find metadata file - prefer {prefix}.metaprompt.json pattern
    meta_files = list(run_dir.glob("*.metaprompt.json")) + list(
        run_dir.glob("*_metadata.json")
    )

    if not meta_files:
        return None

    # Prefer files matching common prefix patterns when multiple exist
    preferred = [f for f in meta_files if "image" in f.name or "prefix" in f.name]
    chosen_file = preferred[0] if preferred else meta_files[0]

    try:
        metadata = json.loads(chosen_file.read_text())
    except (json.JSONDecodeError, IOError):
        return None

    prefix = metadata.get("prefix", "image")

    # Find gallery file
    gallery_file = run_dir / f"{prefix}_gallery.html"
    if not gallery_file.exists():
        return None

    # Find first image for thumbnail using glob (more efficient)
    thumbnail = None
    thumbnail_file = None
    images = sorted(run_dir.glob(f"{prefix}_*_*.png"))
    if images:
        thumbnail = images[0].relative_to(run_dir.parent.parent)
        thumbnail_file = images[0].name

    # Extract timestamp from directory name
    dir_parts = run_dir.name.split("_")
    if len(dir_parts) >= 2:
        timestamp = f"{dir_parts[0]}_{dir_parts[1]}"
    else:
        timestamp = run_dir.name

    # Format timestamp for display
    display_time = format_run_timestamp(timestamp)

    # Get image count
    image_count = len(list(run_dir.glob(f"{prefix}_*_*.png")))

    # Get prompt count
    prompt_count = metadata.get("count", 0)

    # Build paths based on whether this is an archive
    if is_archive:
        gallery_path = f"saved/{run_dir.name}/{prefix}_gallery.html"
    else:
        gallery_path = f"prompts/{run_dir.name}/{prefix}_gallery.html"

    # Get backup info if available
    backup_info = metadata.get("backup_info", {})
    backup_reason = backup_info.get("backup_reason", "")

    return {
        "dir_name": run_dir.name,
        "timestamp": timestamp,
        "display_time": display_time,
        "user_prompt": metadata.get("display_title") or metadata.get("user_prompt", "Unknown prompt"),
        "prefix": prefix,
        "gallery_path": gallery_path,
        "thumbnail": str(thumbnail) if thumbnail else None,
        "thumbnail_file": thumbnail_file,
        "image_count": image_count,
        "prompt_count": prompt_count,
        "model": metadata.get("model") or metadata.get("image_generation", {}).get("model") or "N/A",
        "is_archive": is_archive,
        "backup_reason": backup_reason,
    }


def _extract_flat_archive_infos(saved_dir: Path, interactive: bool = False) -> list[dict]:
    """Extract information about flat archived images.

    Args:
        saved_dir: Path to saved/ directory
        interactive: Whether running in interactive mode

    Returns:
        List of dictionaries with archive info for display
    """
    archives = scan_flat_archives(saved_dir)
    result = []

    for archive in archives:
        prefix = archive["prefix"]
        timestamp = archive["timestamp"]
        first_image = archive["first_image"]

        # Count unique prompts by extracting prompt indices from image filenames
        import re as _re
        prompt_idx_pattern = _re.compile(rf'^{re.escape(prefix)}_\d+_(\d+)_.+\.png$')
        prompt_indices = set()
        for img_path in archive.get("images", []):
            match = prompt_idx_pattern.match(img_path.name)
            if match:
                prompt_indices.add(match.group(1))

        # Format timestamp for display
        display_time = format_run_timestamp(timestamp)

        # Get metadata from first image (gracefully handle extraction errors)
        metadata = {}
        if first_image:
            try:
                metadata = get_flat_archive_metadata(first_image)
            except (RuntimeError, IOError, OSError):
                # Corrupted or unreadable metadata - use defaults
                pass

        # Get backup reason from embedded metadata
        backup_reason = metadata.get("backup_reason", "archived")

        result.append({
            "prefix": prefix,
            "timestamp": timestamp,
            "display_time": display_time,
            "user_prompt": metadata.get("display_title") or metadata.get("user_prompt", "Archived images"),
            "image_count": archive["image_count"],
            "prompt_count": len(prompt_indices),
            "model": metadata.get("model", "N/A"),
            "first_image": first_image,
            "backup_reason": backup_reason,
            "is_flat_archive": True,
        })

    return result


def _build_generation_form() -> str:
    """Build the new generation form HTML."""
    return '''
  <div class="form-section">
    <h2>New Generation</h2>
    <form id="generate-form">
      <div class="form-row">
        <div class="form-group flex-grow">
          <label for="prompt">Prompt</label>
          <input type="text" id="prompt" name="prompt" required placeholder="a dragon flying over mountains">
        </div>
        <div class="form-group">
          <label for="prefix">Prefix</label>
          <input type="text" id="prefix" name="prefix" value="image" placeholder="image">
        </div>
      </div>

      <details class="settings-section" open>
        <summary>Generation Settings</summary>
        <div class="form-row">
          <div class="form-group">
            <label for="count">Prompt Count</label>
            <input type="number" id="count" name="count" value="50" min="1">
          </div>
          <div class="form-group">
            <label for="temperature">Temperature</label>
            <input type="number" id="temperature" name="temperature" value="0.7" step="0.1" min="0" max="2">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="no_cache" name="no_cache">
              Force regenerate grammar (ignore cache)
            </label>
          </div>
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" id="tiled_vae" name="tiled_vae">
              Tiled VAE (memory efficient)
            </label>
          </div>
        </div>
      </details>

      <div class="form-actions">
        <button type="submit" class="btn-primary">Start Grammar and Prompt Generation</button>
      </div>
    </form>
  </div>

  <div class="form-section">
    <h2>New Gallery From Grammar</h2>
    <form id="generate-from-grammar-form">
      <div class="form-row">
        <div class="form-group flex-grow">
          <label for="grammar_title">Title</label>
          <input type="text" id="grammar_title" name="title" placeholder="optional gallery title">
        </div>
        <div class="form-group">
          <label for="grammar_prefix">Prefix</label>
          <input type="text" id="grammar_prefix" name="prefix" value="image" placeholder="image">
        </div>
        <div class="form-group">
          <label for="grammar_count">Prompt Count</label>
          <input type="number" id="grammar_count" name="count" value="50" min="1">
        </div>
      </div>

      <div class="form-row">
        <div class="form-group flex-grow">
          <label for="grammar_text">Tracery Grammar</label>
          <textarea id="grammar_text" name="grammar" rows="12" required placeholder='{"origin": ["#subject# in #setting#"], "subject": ["cat"], "setting": ["forest"]}'></textarea>
        </div>
      </div>

      <details class="settings-section" open>
        <summary>Image Defaults</summary>
        <div class="form-row">
          <div class="form-group">
            <label for="grammar_images_per_prompt">Images/Prompt (0 = prompt-only layout)</label>
            <input type="number" id="grammar_images_per_prompt" name="images_per_prompt" value="1" min="0">
          </div>
          <div class="form-group">
            <label for="grammar_max_prompts">Max Prompts</label>
            <input type="number" id="grammar_max_prompts" name="max_prompts" min="1" placeholder="all">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label for="grammar_width">Width</label>
            <input type="number" id="grammar_width" name="width" value="864" step="8">
          </div>
          <div class="form-group">
            <label for="grammar_height">Height</label>
            <input type="number" id="grammar_height" name="height" value="1152" step="8">
          </div>
          <div class="form-group">
            <label for="grammar_seed">Seed</label>
            <input type="number" id="grammar_seed" name="seed" placeholder="random">
          </div>
        </div>
      </details>

      <div class="form-actions">
        <button type="submit" class="btn-primary">Create Gallery From Grammar</button>
      </div>
    </form>
  </div>
'''


def _build_queue_status_bar() -> str:
    """Build the queue status bar HTML."""
    return QueueStatusBar.html()


def _build_log_panel() -> str:
    """Build the collapsible log panel HTML."""
    return LogPanel.html()


def _build_notifications() -> str:
    """Build toast + confirm dialog markup."""
    return Notifications.html()


def _build_interactive_js() -> str:
    """Build the JavaScript for form submission and SSE."""
    log_js = LogPanel.js()
    sse_js = SSEClient.js()
    notify_js = Notifications.js()

    return f'''
<script>
document.addEventListener('DOMContentLoaded', function() {{
  const form = document.getElementById('generate-form');
  const grammarForm = document.getElementById('generate-from-grammar-form');
  const queueStatus = document.getElementById('queue-status');
  const queueText = document.getElementById('queue-text');
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const btnKill = document.getElementById('btn-kill');
  const btnClear = document.getElementById('btn-clear');
  const logPanel = document.getElementById('log-panel');
  const submitBtn = form ? form.querySelector('button[type="submit"]') : null;
  const grammarSubmitBtn = grammarForm ? grammarForm.querySelector('button[type="submit"]') : null;

  if (!form || !grammarForm) {{
    return;
  }}

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

  async function postJson(url, data) {{
    const resp = await fetch(url, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(data),
    }});
    if (!resp.ok) {{
      let detail = 'Unknown error';
      try {{
        const err = await resp.json();
        detail = err.detail || detail;
      }} catch (_e) {{}}
      throw new Error(detail);
    }}
    return await resp.json();
  }}

  function initSSE() {{
    const es = connectSSE();
    if (!es) return;

    es.addEventListener('status', (e) => {{
      const data = JSON.parse(e.data);
      updateStatus(data);
    }});

    es.addEventListener('queue_updated', (e) => {{
      const data = JSON.parse(e.data);
      updateStatus(data);
    }});

    es.addEventListener('queue_cleared', (_e) => {{
      btnClear.classList.add('hidden');
      queueStatus.classList.add('hidden');
      showToast('Queue cleared', 'success');
    }});

    es.addEventListener('task_started', (e) => {{
      const task = JSON.parse(e.data);
      queueText.textContent = `Running: ${{task.type}}`;
      btnKill.classList.remove('hidden');
      progressContainer.classList.remove('hidden');
      queueStatus.classList.remove('hidden');
    }});

    es.addEventListener('task_progress', (e) => {{
      const data = JSON.parse(e.data);
      const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
      progressFill.style.width = pct + '%';
      progressText.textContent = `${{data.current}}/${{data.total}}`;
      if (data.message) {{
        queueText.textContent = data.message;
      }}
    }});

    es.addEventListener('task_completed', (e) => {{
      const data = JSON.parse(e.data);
      queueText.textContent = 'Completed';
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
      // Reload to show new gallery
      if (data.result && data.result.run_id) {{
        setTimeout(() => location.reload(), 1000);
      }}
    }});

    es.addEventListener('task_failed', (e) => {{
      const data = JSON.parse(e.data);
      queueText.textContent = 'Failed: ' + (data.error || 'Unknown error');
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
    }});

    es.addEventListener('task_cancelled', (e) => {{
      queueText.textContent = 'Cancelled';
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
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

  function updateStatus(data) {{
    const pending = data.pending_count || 0;
    const current = data.current;

    if (current) {{
      queueStatus.classList.remove('hidden');
      btnKill.classList.remove('hidden');
      progressContainer.classList.remove('hidden');
      queueText.textContent = current.progress?.message || `Running: ${{current.type}}`;

      if (current.progress && current.progress.total > 0) {{
        const pct = Math.round((current.progress.current / current.progress.total) * 100);
        progressFill.style.width = pct + '%';
        progressText.textContent = `${{current.progress.current}}/${{current.progress.total}}`;
      }}
    }} else if (pending > 0) {{
      queueStatus.classList.remove('hidden');
      queueText.textContent = `${{pending}} task(s) pending`;
      btnClear.classList.remove('hidden');
      btnKill.classList.add('hidden');
      progressContainer.classList.add('hidden');
    }} else {{
      queueStatus.classList.add('hidden');
    }}
  }}

  // Form submission
  form.addEventListener('submit', async function(e) {{
    e.preventDefault();

    const formData = new FormData(form);
    const promptValue = formData.get('prompt');

    if (!promptValue || !promptValue.trim()) {{
      showToast('Please enter a prompt', 'error');
      return;
    }}

    const data = {{
      prompt: promptValue.trim(),
      prefix: formData.get('prefix') || 'image',
      count: parseInt(formData.get('count')) || 50,
      temperature: parseFloat(formData.get('temperature')) || 0.7,
      no_cache: form.querySelector('#no_cache')?.checked || false,
      tiled_vae: form.querySelector('#tiled_vae')?.checked ?? false,
    }};

    // Show status immediately
    queueStatus.classList.remove('hidden');
    queueText.textContent = 'Submitting...';
    await withButtonBusy(submitBtn, 'Submitting...', async () => {{
      try {{
        const result = await postJson('/api/generate', data);
        queueText.textContent = result.message;
        showToast(result.message || 'Generation queued', 'success');
      }} catch (err) {{
        showToast('Error: ' + err.message, 'error', 4200);
        queueStatus.classList.add('hidden');
      }}
    }});
  }});

  grammarForm.addEventListener('submit', async function(e) {{
    e.preventDefault();

    const formData = new FormData(grammarForm);
    const grammarValue = formData.get('grammar');

    if (!grammarValue || !grammarValue.trim()) {{
      showToast('Please paste a grammar', 'error');
      return;
    }}

    const data = {{
      grammar: grammarValue.trim(),
      title: (formData.get('title') || '').trim() || null,
      prefix: formData.get('prefix') || 'image',
      count: parseInt(formData.get('count')) || 50,
      images_per_prompt: (() => {{
        const value = formData.get('images_per_prompt');
        return value === null || value === '' ? 1 : parseInt(value);
      }})(),
      max_prompts: formData.get('max_prompts') ? parseInt(formData.get('max_prompts')) : null,
      width: parseInt(formData.get('width')) || 864,
      height: parseInt(formData.get('height')) || 1152,
      seed: formData.get('seed') ? parseInt(formData.get('seed')) : null,
    }};

    queueStatus.classList.remove('hidden');
    queueText.textContent = 'Submitting grammar...';
    await withButtonBusy(grammarSubmitBtn, 'Submitting...', async () => {{
      try {{
        const result = await postJson('/api/generate-from-grammar', data);
        queueText.textContent = result.message;
        showToast(result.message || 'Grammar gallery queued', 'success');
      }} catch (err) {{
        showToast('Error: ' + err.message, 'error', 4200);
        queueStatus.classList.add('hidden');
      }}
    }});
  }});

  // Kill button
  btnKill.addEventListener('click', async () => {{
    await withButtonBusy(btnKill, 'Killing...', async () => {{
      try {{
        await fetch('/api/worker/kill', {{method: 'POST'}});
        showToast('Kill signal sent', 'success');
      }} catch (err) {{
        showToast('Kill failed: ' + err.message, 'error', 4200);
      }}
    }});
  }});

  // Clear queue button
  btnClear.addEventListener('click', async () => {{
    await withButtonBusy(btnClear, 'Clearing...', async () => {{
      try {{
        await fetch('/api/queue/clear', {{method: 'POST'}});
        btnClear.classList.add('hidden');
        queueStatus.classList.add('hidden');
        showToast('Queue clear requested', 'success');
      }} catch (err) {{
        showToast('Clear failed: ' + err.message, 'error', 4200);
      }}
    }});
  }});

  // Start SSE connection
  initSSE();
}});

// Delete gallery function (global so onclick can access it)
window.deleteGallery = async function(runId) {{
  const confirmed = await confirmAction('Delete this gallery and all its images? This cannot be undone.', {{
    confirmText: 'Delete',
    cancelText: 'Cancel'
  }});
  if (!confirmed) {{
    return;
  }}

  try {{
    const resp = await fetch(`/api/gallery/${{runId}}`, {{ method: 'DELETE' }});
    if (!resp.ok) {{
      let detail = 'Unknown error';
      try {{
        const err = await resp.json();
        detail = err.detail || detail;
      }} catch (_e) {{}}
      showToast('Delete failed: ' + detail, 'error', 4200);
      return;
    }}
    showToast('Delete queued', 'success');
    // Reload page to reflect deletion
    location.reload();
  }} catch (err) {{
    showToast('Delete failed: ' + err.message, 'error', 4200);
  }}
}};
</script>
'''


def _build_interactive_styles() -> str:
    """Build additional CSS for interactive mode."""
    return (
        FormStyles.css() +
        Buttons.css() +
        Notifications.css() +
        QueueStatusBar.css() +
        '''
    /* Adjust container padding for status bar */
    body { padding-bottom: 80px; }
''' +
        LogPanel.css() +
        IndexStyles.css()
    )


def _build_card_html(run: dict, interactive: bool, is_archive: bool = False) -> str:
    """Build HTML for a single gallery card."""
    escaped_prompt = html.escape(run["user_prompt"])
    truncated_prompt = (escaped_prompt[:100] + "...") if len(escaped_prompt) > 100 else escaped_prompt

    if run["thumbnail_file"] and interactive:
        # In interactive mode, use the gallery route for images
        if is_archive:
            thumbnail_src = f'/archive/{run["dir_name"]}/{run["thumbnail_file"]}'
        else:
            thumbnail_src = f'/gallery/{run["dir_name"]}/{run["thumbnail_file"]}'
        thumbnail_html = f'<img src="{thumbnail_src}" loading="lazy">'
    elif run["thumbnail"]:
        # In static mode, use relative path
        thumbnail_html = f'<img src="{run["thumbnail"]}" loading="lazy">'
    else:
        thumbnail_html = '<div class="no-thumbnail">No images</div>'

    # Use gallery route in interactive mode, relative path in static mode
    if interactive:
        gallery_href = f'/archive/{run["dir_name"]}' if is_archive else f'/gallery/{run["dir_name"]}'
    else:
        gallery_href = run["gallery_path"]

    card_class = "card archive" if is_archive else "card"

    # Add archive badge if this is an archive
    badge_html = ""
    if is_archive:
        reason = run.get("backup_reason", "archived")
        reason_label = {
            "pre_regenerate": "Pre-Regen",
            "pre_enhance": "Pre-Enhance",
            "manual_archive": "Saved",
        }.get(reason, "Backup")
        badge_html = f'<span class="archive-badge">{reason_label}</span>'

    # Add delete button for active galleries in interactive mode
    delete_btn_html = ""
    if interactive and not is_archive:
        delete_btn_html = f'''
      <button class="btn-delete" onclick="event.preventDefault(); event.stopPropagation(); deleteGallery('{run["dir_name"]}');" title="Delete gallery">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      </button>'''

    return f'''    <a href="{gallery_href}" class="{card_class}">
      <div class="thumbnail">
        {thumbnail_html}{badge_html}{delete_btn_html}
      </div>
      <div class="info">
        <div class="prompt" title="{escaped_prompt}">{truncated_prompt}</div>
        <div class="meta">
          <span class="time">{run["display_time"]}</span>
          <span class="stats">{run["image_count"]} {'image' if run['image_count'] == 1 else 'images'} | {run["prompt_count"]} {'prompt' if run['prompt_count'] == 1 else 'prompts'}</span>
          <span class="model">{run["model"]}</span>
        </div>
      </div>
    </a>'''


def _build_flat_archive_card_html(archive: dict, interactive: bool) -> str:
    """Build HTML for a flat archive card."""
    escaped_prompt = html.escape(archive["user_prompt"])
    truncated_prompt = (escaped_prompt[:100] + "...") if len(escaped_prompt) > 100 else escaped_prompt

    first_image = archive.get("first_image")
    if first_image and interactive:
        # In interactive mode, use the saved route for images
        thumbnail_src = f'/saved/{first_image.name}'
        thumbnail_html = f'<img src="{thumbnail_src}" loading="lazy">'
    elif first_image:
        # In static mode, use relative path
        thumbnail_html = f'<img src="saved/{first_image.name}" loading="lazy">'
    else:
        thumbnail_html = '<div class="no-thumbnail">No images</div>'

    # Flat archives don't have gallery pages - they just display in the grid
    reason = archive.get("backup_reason", "archived")
    reason_label = {
        "pre_regenerate": "Pre-Regen",
        "pre_enhance": "Pre-Enhance",
        "manual_archive": "Saved",
    }.get(reason, "Backup")

    badge_html = f'<span class="archive-badge">{reason_label}</span>'

    return f'''    <div class="card archive flat-archive">
      <div class="thumbnail">
        {thumbnail_html}{badge_html}
      </div>
      <div class="info">
        <div class="prompt" title="{escaped_prompt}">{truncated_prompt}</div>
        <div class="meta">
          <span class="time">{archive["display_time"]}</span>
          <span class="stats">{archive["image_count"]} {'image' if archive['image_count'] == 1 else 'images'}</span>
          <span class="model">{archive["model"]}</span>
        </div>
      </div>
    </div>'''


def _build_index_html(
    active_runs: list[dict],
    archived_runs: list[dict] = None,
    flat_archives: list[dict] = None,
    interactive: bool = False,
) -> str:
    """Build the master index HTML document."""
    if archived_runs is None:
        archived_runs = []
    if flat_archives is None:
        flat_archives = []

    # Build active run cards
    active_cards_html = []
    for run in active_runs:
        card = _build_card_html(run, interactive, is_archive=False)
        active_cards_html.append(card)

    active_cards_joined = "\n".join(active_cards_html) if active_cards_html else '<p class="empty">No galleries found. Generate some images first!</p>'
    run_count = len(active_runs)

    # Build archived run cards (legacy directory-based)
    archived_section = ""
    if archived_runs:
        archived_cards_html = []
        for run in archived_runs:
            card = _build_card_html(run, interactive, is_archive=True)
            archived_cards_html.append(card)
        archived_cards_joined = "\n".join(archived_cards_html)
        archived_section = f'''
    <h2 class="section-title">Saved Archives ({len(archived_runs)})</h2>
    <div class="grid">
{archived_cards_joined}
    </div>'''

    # Build flat archive cards (new format)
    flat_archive_section = ""
    if flat_archives:
        flat_cards_html = []
        for archive in flat_archives:
            card = _build_flat_archive_card_html(archive, interactive)
            flat_cards_html.append(card)
        flat_cards_joined = "\n".join(flat_cards_html)
        flat_archive_section = f'''
    <h2 class="section-title">Archived Images ({len(flat_archives)} sets, {sum(a["image_count"] for a in flat_archives)} images)</h2>
    <div class="grid">
{flat_cards_joined}
    </div>'''

    # Build interactive sections
    form_html = _build_generation_form() if interactive else ""
    log_panel_html = _build_log_panel() if interactive else ""
    notifications_html = _build_notifications() if interactive else ""
    queue_html = _build_queue_status_bar() if interactive else ""
    js_html = _build_interactive_js() if interactive else ""
    extra_styles = _build_interactive_styles() if interactive else ""

    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Image Prompt Generator - Gallery Index</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: system-ui; padding: 20px; background: #1a1a1a; color: #fff; margin: 0; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #888; margin-bottom: 24px; }}
    .section-title {{ margin: 32px 0 16px; font-size: 16px; color: #888; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }}
    .card {{ background: #2a2a2a; border-radius: 12px; overflow: hidden; text-decoration: none; color: inherit; transition: transform 0.2s, box-shadow 0.2s; display: block; }}
    .card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}
    .card.archive {{ border: 2px solid #665500; }}
    .card.archive:hover {{ border-color: #997700; }}
    .card.flat-archive {{ cursor: default; }}
    .card.flat-archive:hover {{ transform: none; box-shadow: none; }}
    .thumbnail {{ aspect-ratio: 3/4; background: #333; overflow: hidden; position: relative; }}
    .thumbnail img {{ width: 100%; height: 100%; object-fit: cover; }}
    .no-thumbnail {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666; }}
    .archive-badge {{ position: absolute; top: 8px; right: 8px; background: #665500; color: #fff; font-size: 10px; padding: 2px 8px; border-radius: 4px; }}
    .info {{ padding: 16px; }}
    .prompt {{ font-size: 13px; color: #ddd; margin-bottom: 12px; line-height: 1.4; height: calc(1.4em * 3); overflow: hidden; word-wrap: break-word; overflow-wrap: break-word; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }}
    .meta {{ display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #888; }}
    .meta .time {{ color: #6af; }}
    .meta .stats {{ color: #aaa; }}
    .meta .model {{ color: #8f8; }}
    .empty {{ color: #666; text-align: center; padding: 60px 20px; }}
{extra_styles}
  </style>
</head>
<body>
{notifications_html}
  <div class="container">
    <h1>Image Prompt Generator</h1>
    <p class="subtitle">{run_count} generation runs</p>
{form_html}{log_panel_html}
    <div class="grid">
{active_cards_joined}
    </div>{archived_section}{flat_archive_section}
  </div>
{queue_html}
{js_html}
</body>
</html>
'''
