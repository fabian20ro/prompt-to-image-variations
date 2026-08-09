"""Shared HTML, JavaScript, and CSS components for galleries.

This module consolidates duplicate UI components from gallery.py and gallery_index.py
to improve maintainability and reduce code duplication.
"""


class LogPanel:
    """Log panel component for displaying generation logs."""

    REQUIRED_HTML_IDS = ("log-panel", "log-content", "log-count", "btn-clear-logs")

    @classmethod
    def validate(cls) -> None:
        """Assert the produced HTML contains every required element ID.

        Raises ValueError if any critical DOM node is missing — which would break
        JavaScript bindings in log panel consumers. Calling this after editing
        LogPanel.html() catches regressions (e.g., renaming an id or removing a
        required button) before they reach production galleries.
        """
        produced = cls.html()
        for element_id in cls.REQUIRED_HTML_IDS:
            if f'id="{element_id}"' not in produced and f"id='{element_id}'" not in produced:
                raise ValueError(
                    f"LogPanel HTML is missing required element id='{element_id}'. "
                    f"The panel will silently fail to bind JavaScript event handlers."
                )

    @staticmethod
    def html() -> str:
        """Build the collapsible log panel HTML."""
        return '''
  <details id="log-panel" class="log-panel">
    <summary>
      <span class="log-title">Generation Logs</span>
      <span id="log-count" class="log-count">0</span>
      <button id="btn-clear-logs" class="btn-small btn-secondary" onclick="event.preventDefault(); clearLogs();">Clear</button>
    </summary>
    <div id="log-content" class="log-content"></div>
  </details>'''

    @staticmethod
    def css() -> str:
        """CSS for log panel."""
        return '''
    /* Log panel */
    .log-panel { background: #2a2a2a; border-radius: 8px; margin-bottom: 20px; }
    .log-panel summary { padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; list-style: none; }
    .log-panel summary::-webkit-details-marker { display: none; }
    .log-title { color: #888; font-size: 14px; }
    .log-count { background: #444; color: #ddd; font-size: 11px; padding: 2px 8px; border-radius: 10px; }
    .log-content { max-height: 300px; overflow-y: auto; padding: 0 16px 16px; font-family: monospace; font-size: 12px; line-height: 1.6; }
    .log-line { color: #aaa; white-space: pre-wrap; word-break: break-all; }
    .log-line .timestamp { color: #6af; margin-right: 8px; }
    .log-line.error { color: #f88; }
    .log-line.warning { color: #fa0; }'''

    @staticmethod
    def js() -> str:
        """JavaScript for log panel functionality."""
        return '''
  let logLineCount = 0;
  const MAX_LOG_LINES = 500;

  function appendLog(timestamp, message) {
    const logContent = document.getElementById('log-content');
    const logCount = document.getElementById('log-count');
    if (!logContent) return;

    const line = document.createElement('div');
    line.className = 'log-line';
    if (message.toLowerCase().includes('error')) line.className += ' error';
    else if (message.toLowerCase().includes('warning')) line.className += ' warning';

    const time = timestamp ? new Date(timestamp).toLocaleTimeString() : '';
    line.innerHTML = `<span class="timestamp">${time}</span>${escapeHtml(message)}`;

    logContent.appendChild(line);
    logLineCount++;
    if (logCount) logCount.textContent = logLineCount;

    // Auto-scroll to bottom
    logContent.scrollTop = logContent.scrollHeight;

    // Trim old lines if too many
    while (logContent.children.length > MAX_LOG_LINES) {
      logContent.removeChild(logContent.firstChild);
      logLineCount--;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.clearLogs = function() {
    const logContent = document.getElementById('log-content');
    const logCount = document.getElementById('log-count');
    if (logContent) {
      logContent.innerHTML = '';
      logLineCount = 0;
      if (logCount) logCount.textContent = '0';
    }
  };'''


class QueueStatusBar:
    """Queue status bar component for the index page."""

    @staticmethod
    def html() -> str:
        """Build the queue status bar HTML."""
        return '''
  <div id="queue-status" class="queue-status hidden">
    <div class="queue-info">
      <span id="queue-text">Idle</span>
    </div>
    <div id="progress-container" class="progress-container hidden">
      <div class="progress-bar">
        <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
      </div>
      <span id="progress-text">0%</span>
    </div>
    <div class="queue-actions">
      <button id="btn-kill" class="btn-danger btn-small hidden">Kill</button>
      <button id="btn-clear" class="btn-secondary btn-small hidden">Clear Queue</button>
    </div>
  </div>'''

    @staticmethod
    def css() -> str:
        """CSS for queue status bar."""
        return '''
    /* Queue status bar */
    .queue-status { position: fixed; bottom: 0; left: 0; right: 0; background: #2a2a2a; border-top: 1px solid #444; padding: 12px 20px; display: flex; align-items: center; gap: 16px; z-index: 1000; flex-wrap: wrap; }
    .queue-status.hidden { display: none; }
    .queue-info { flex: 1; font-size: 14px; color: #ddd; }
    .progress-container { display: flex; align-items: center; gap: 12px; }
    .progress-container.hidden { display: none; }
    .progress-bar { width: min(38vw, 240px); min-width: 140px; height: 8px; background: #444; border-radius: 4px; overflow: hidden; }
    .progress-fill { height: 100%; background: #4a9eff; transition: width 0.3s; }
    .queue-actions { display: flex; gap: 8px; }
    .queue-actions .hidden { display: none; }
    @media (max-width: 720px) {
      .queue-status { padding: 10px 12px; gap: 10px; }
      .queue-info { width: 100%; flex: 1 0 100%; }
      .progress-container { flex: 1; min-width: 0; }
      .queue-actions { width: 100%; justify-content: flex-end; }
    }'''


class ProgressBar:
    """Progress bar component for gallery pages."""

    @staticmethod
    def html() -> str:
        """Build the fixed progress bar HTML."""
        return f'''
  <div id="progress-bar" class="progress-bar-fixed hidden">
    <div class="progress-info">
      <span id="progress-message">Idle</span>
    </div>
    <div class="progress-container">
      <div class="progress-track">
        <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
      </div>
      <span id="progress-text">Generated: 0 / 0 images</span>
    </div>
  </div>'''

    @staticmethod
    def css() -> str:
        """CSS for progress bar."""
        return '''
    /* Progress bar */
    .progress-bar-fixed { position: fixed; bottom: 0; left: 0; right: 0; background: #2a2a2a; border-top: 1px solid #444; padding: 12px 20px; display: flex; align-items: center; gap: 16px; z-index: 1000; flex-wrap: wrap; }
    .progress-bar-fixed.hidden { display: none; }
    .progress-info { flex: 1; font-size: 14px; color: #ddd; }
    .progress-track { width: min(38vw, 240px); min-width: 140px; height: 8px; background: #444; border-radius: 4px; overflow: hidden; }
    @media (max-width: 720px) {
      .progress-bar-fixed { padding: 10px 12px; gap: 10px; }
      .progress-info { width: 100%; flex: 1 0 100%; }
    }'''


class Notifications:
    """Toast and modal helpers for non-blocking UI feedback."""

    @staticmethod
    def html() -> str:
        """Build toast region + confirm modal markup."""
        return '''
  <div id="toast-region" class="toast-region" aria-live="polite" aria-atomic="true"></div>
  <div id="confirm-modal" class="confirm-modal hidden" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
    <div class="confirm-card">
      <h3 id="confirm-title">Confirm Action</h3>
      <p id="confirm-message"></p>
      <div class="confirm-actions">
        <button id="confirm-cancel" type="button" class="btn-secondary">Cancel</button>
        <button id="confirm-ok" type="button" class="btn-danger">Confirm</button>
      </div>
    </div>
  </div>'''

    @staticmethod
    def css() -> str:
        """CSS for toast and confirm dialog."""
        return '''
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
    .toast-region { position: fixed; top: 12px; right: 12px; z-index: 2500; display: flex; flex-direction: column; gap: 8px; max-width: min(92vw, 360px); }
    .toast { border-radius: 8px; padding: 10px 12px; font-size: 14px; color: #fff; background: #2f3a48; border: 1px solid #44556b; box-shadow: 0 4px 14px rgba(0,0,0,0.25); animation: fadeIn 0.2s ease-out; }
    .toast.success { background: #1f4b2f; border-color: #2f7244; }
    .toast.error { background: #562727; border-color: #8f3d3d; }
    .confirm-modal { position: fixed; inset: 0; z-index: 2600; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.45); padding: 16px; }
    .confirm-modal.hidden { display: none; }
    .confirm-card { background: #222; border: 1px solid #444; border-radius: 10px; padding: 16px; width: min(92vw, 420px); }
    .confirm-card h3 { margin: 0 0 8px; font-size: 18px; color: #eee; }
    .confirm-card p { margin: 0 0 14px; color: #bbb; font-size: 14px; line-height: 1.5; }
    .confirm-actions { display: flex; justify-content: flex-end; gap: 10px; }'''

    @staticmethod
    def js() -> str:
        """JavaScript for toast and modal behavior."""
        return '''
  function showToast(message, type = 'info', timeoutMs = 3200) {
    const region = document.getElementById('toast-region');
    if (!region) return;
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    el.style.cursor = 'pointer';
    el.title = 'Click to dismiss';
    el.onclick = () => el.remove();
    region.appendChild(el);
    setTimeout(() => {
      el.remove();
    }, timeoutMs);
  }

  function confirmAction(message, opts = {}) {
    const modal = document.getElementById('confirm-modal');
    const msg = document.getElementById('confirm-message');
    const ok = document.getElementById('confirm-ok');
    const cancel = document.getElementById('confirm-cancel');
    if (!modal || !msg || !ok || !cancel) return Promise.resolve(false);

    msg.textContent = message;
    ok.textContent = opts.confirmText || 'Confirm';
    cancel.textContent = opts.cancelText || 'Cancel';

    return new Promise((resolve) => {
      const cleanup = () => {
        ok.onclick = null;
        cancel.onclick = null;
        modal.onclick = null;
        document.removeEventListener('keydown', onEsc);
        modal.classList.add('hidden');
      };
      const onEsc = (e) => {
        if (e.key === 'Escape') {
          cleanup();
          resolve(false);
        }
      };
      ok.onclick = () => {
        cleanup();
        resolve(true);
      };
      cancel.onclick = () => {
        cleanup();
        resolve(false);
      };
      modal.onclick = (e) => {
        if (e.target === modal) {
          cleanup();
          resolve(false);
        }
      };
      document.addEventListener('keydown', onEsc);
      modal.classList.remove('hidden');
      ok.focus();
    });
  }'''


class SSEClient:
    """SSE client JavaScript for real-time updates."""

    @staticmethod
    def js() -> str:
        """Build SSE connection and reconnection logic."""
        return '''
  let eventSource = null;
  let sseRetryCount = 0;
  const MAX_SSE_RETRIES = 10;

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    if (sseRetryCount >= MAX_SSE_RETRIES) {
      console.warn('SSE: Max retries reached, giving up');
      return;
    }

    try {
      eventSource = new EventSource('/api/events');
    } catch (e) {
      console.error('SSE: Failed to create EventSource', e);
      return;
    }

    eventSource.onopen = () => {
      console.log('SSE connected');
      sseRetryCount = 0;
    };

    eventSource.onerror = (e) => {
      console.error('SSE error', e);
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      sseRetryCount++;
      const delay = Math.min(3000 * Math.pow(2, sseRetryCount - 1), 30000);
      console.log(`SSE: Retry ${sseRetryCount}/${MAX_SSE_RETRIES} in ${delay}ms`);
      setTimeout(connectSSE, delay);
    };

    eventSource.addEventListener('ping', () => {});

    return eventSource;
  }'''


class Buttons:
    """Button style components."""

    @staticmethod
    def css() -> str:
        """CSS for button styles."""
        return '''
    /* Buttons */
    .btn-primary { background: #4a9eff; color: #fff; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; cursor: pointer; font-weight: 500; min-height: 40px; }
    .btn-primary:hover { background: #3d8be0; }
    .btn-secondary { background: #444; color: #fff; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; cursor: pointer; min-height: 40px; }
    .btn-secondary:hover { background: #555; }
    .btn-danger { background: #d44; color: #fff; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; cursor: pointer; min-height: 40px; }
    .btn-danger:hover { background: #c33; }
    .btn-small { padding: 6px 12px; font-size: 12px; min-height: 32px; }
    .btn-primary:disabled, .btn-secondary:disabled, .btn-danger:disabled { opacity: 0.55; cursor: not-allowed; }
    @media (pointer: coarse) {
      .btn-primary, .btn-secondary, .btn-danger, .btn-small { min-height: 44px; padding: 10px 14px; }
    }'''


class NavHeader:
    """Navigation header component."""

    @staticmethod
    def html() -> str:
        """Build navigation header with back link."""
        return '''
  <nav class="nav-header">
    <a href="/index" class="nav-link">&larr; Back to Index</a>
  </nav>'''

    @staticmethod
    def css() -> str:
        """CSS for navigation header."""
        return '''
    /* Navigation header */
    .nav-header { margin-bottom: 16px; }
    .nav-link { color: #6af; text-decoration: none; font-size: 14px; }
    .nav-link:hover { text-decoration: underline; }'''


class InteractiveStyles:
    """Combined interactive styles for galleries."""

    @staticmethod
    def css() -> str:
        """All interactive mode CSS."""
        return (
            NavHeader.css() +
            Buttons.css() +
            LogPanel.css() +
            ProgressBar.css() +
            '''
    /* Body padding for fixed bars */
    body { padding-bottom: 80px; }'''
        )


class FormStyles:
    """Form styles for the index page."""

    @staticmethod
    def css() -> str:
        """CSS for form elements."""
        return '''
    /* Form styles */
    .form-section { background: #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .form-section h2 { margin: 0 0 20px 0; font-size: 18px; }
    .form-row { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
    .form-group { display: flex; flex-direction: column; gap: 6px; min-width: 120px; }
    .form-group.flex-grow { flex: 1; min-width: 200px; }
    .form-group label { font-size: 12px; color: #888; }
    .form-group input, .form-group select, .form-group textarea { background: #1a1a1a; border: 1px solid #444; border-radius: 6px; padding: 8px 12px; color: #fff; font-size: 14px; }
    .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: #6af; }
    .form-group input::placeholder { color: #666; }
    .form-group textarea { resize: vertical; font-family: monospace; min-height: 160px; }
    .checkbox-group { flex-direction: row; align-items: center; }
    .checkbox-group label { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #ddd; cursor: pointer; }
    .checkbox-group input[type="checkbox"] { width: 16px; height: 16px; }
    .settings-section { background: #222; border-radius: 8px; margin-bottom: 16px; }
    .settings-section summary { padding: 12px 16px; cursor: pointer; color: #888; font-size: 14px; list-style: none; }
    .settings-section summary::-webkit-details-marker { display: none; }
    .settings-section summary::before { content: "\\25B6"; margin-right: 8px; font-size: 10px; display: inline-block; transition: transform 0.2s; }
    .settings-section[open] summary::before { transform: rotate(90deg); }
    .settings-section[open] { padding-bottom: 16px; }
    .settings-section > div { padding: 0 16px; }
    .form-actions { margin-top: 20px; }'''


class GalleryStyles:
    """Gallery-specific interactive styles."""

    @staticmethod
    def css() -> str:
        """CSS for interactive gallery features."""
        return '''
    /* Interactive grammar section */
    .grammar-section-interactive { background: #2a2a2a; border-radius: 8px; margin-bottom: 20px; padding: 16px; }
    .grammar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .grammar-title { font-size: 14px; color: #888; }
    .grammar-actions { display: flex; gap: 8px; }
    .grammar-editor { width: 100%; height: 200px; background: #1a1a1a; border: 1px solid #444; border-radius: 6px; padding: 12px; color: #8f8; font-family: monospace; font-size: 12px; resize: vertical; }
    .grammar-editor:focus { outline: none; border-color: #6af; }

    /* Action bar */
    .action-bar { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
    .action-spacer { flex: 1; }

    /* Card actions */
    .card-actions { padding: 8px 12px; display: flex; gap: 8px; border-top: 1px solid #333; }'''


class IndexStyles:
    """Index page specific styles."""

    @staticmethod
    def css() -> str:
        """CSS for delete button on index cards."""
        return '''
    /* Delete button on cards */
    .btn-delete { position: absolute; top: 8px; right: 8px; background: rgba(200, 50, 50, 0.85); color: #fff; border: none; border-radius: 6px; padding: 6px 8px; cursor: pointer; opacity: 0; transition: opacity 0.2s, background 0.2s; z-index: 10; }
    .btn-delete:hover { background: rgba(220, 60, 60, 1); }
    .card:hover .btn-delete { opacity: 1; }
    .card:focus-within .btn-delete { opacity: 1; }
    @media (hover: none), (pointer: coarse) {
      .btn-delete { opacity: 1; min-width: 44px; min-height: 44px; }
    }'''
