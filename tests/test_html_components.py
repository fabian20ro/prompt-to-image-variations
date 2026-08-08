"""Tests for html_components.py - shared HTML/CSS/JS components."""

from html_components import (
    LogPanel,
    QueueStatusBar,
    ProgressBar,
    Notifications,
    SSEClient,
    Buttons,
    NavHeader,
    InteractiveStyles,
    FormStyles,
    GalleryStyles,
    IndexStyles,
)


class TestButtons:
    """Tests for Buttons component."""

    def test_css_returns_string(self):
        css = Buttons.css()
        assert ".btn" in css

    def test_all_button_variants_defined(self):
        """Buttons.css() must define every button variant used by the UI.

        The gallery relies on .btn-primary, .btn-secondary, and .btn-danger for all
        interactive controls (generate, clear queue, kill job). Losing any one of them
        silently breaks a user-facing action — buttons become invisible or unstyled.
        Each selector must survive as a substring so regression cannot remove one
        without the test catching it. The disabled-state rule is also required: users
        need visual feedback when actions are unavailable (e.g. during generation).
        """
        css = Buttons.css()
        assert ".btn-primary" in css, "Primary button style must be defined."
        assert ".btn-secondary" in css, "Secondary button style must be defined."
        assert ".btn-danger" in css, "Danger button style must be defined."
        assert ".btn-small" in css, "Small button modifier must be defined."
        assert ":disabled" in css or "disabled {" in css, (
            "Disabled state styling must exist so users see unavailable actions.\""
        )

    def test_touch_target_minimum_height(self):
        """Buttons.css() enforces a 44px minimum height on touch targets.

        Mobile users need adequately-sized tap targets per accessibility guidelines.
        Without this, gallery controls become hard to hit on phones — causing failed
        submissions and frustrating UX. The (pointer: coarse) media query is the
        observable mechanism that triggers the increased sizing for touch devices.
        """
        css = Buttons.css()
        assert "44px" in css or "min-height: 44" in css, (
            "Touch targets must have a 44px minimum height per accessibility standards."
        )
        assert "@media" in css and "(pointer:" in css, (
            "Touch-specific sizing must be gated by the pointer media query feature."
        )


class TestLogPanel:
    """Tests for LogPanel component."""

    def test_html_contains_log_panel(self):
        html = LogPanel.html()
        assert "log-panel" in html
        assert "log-content" in html
        assert "log-count" in html

    def test_css_returns_string(self):
        css = LogPanel.css()
        assert ".log-panel" in css
        assert ".log-content" in css

    def test_js_returns_string(self):
        js = LogPanel.js()
        assert "appendLog" in js
        assert "clearLogs" in js
        assert "MAX_LOG_LINES" in js

    def test_escape_html_function_exists(self):
        """escapeHtml must be a standalone function callable from log-line rendering."""
        js = LogPanel.js()
        # The function is defined with 'function escapeHtml(text)' at top level.
        assert "function escapeHtml(" in js or "escapeHtml = function" in js

    def test_escape_html_produces_escaped_output(self):
        """escapeHtml uses DOM textContent→innerHTML, so raw HTML is escaped.

        This is a security contract: without proper escaping, user-supplied log
        messages could inject HTML/script tags into the gallery UI (XSS).
        The implementation creates a div, sets .textContent, then reads .innerHTML —
        which forces browsers to escape < > & characters.
        """
        js = LogPanel.js()
        # Verify the DOM-based pattern is used: createElement + textContent + innerHTML.
        assert "createElement('div')" in js or 'createElement("div")' in js
        assert "textContent" in js
        assert "innerHTML" in js

    def test_error_warning_classification_uses_else_if(self):
        """Log line classification uses exclusive branching for error/warning classes.

        The implementation assigns `.error` and `.warning` CSS classes based on the
        message content to make errors visually distinct from normal log lines. Using
        two independent `if` blocks would allow a single line like 'Error in warning
        handler' to get BOTH classes — doubling its visual weight and potentially
        confusing users into thinking it's an error when it's really just a warning.
        The `else if` chain guarantees at most one styling class is applied per line,
        preserving the intended visual hierarchy: error > warning > info.
        """
        js = LogPanel.js()
        # Pattern: 'if (...) className += " error"; else if (...) className += " warning"'
        assert "else if" in js, (
            "Error/warning classification must use `else if` to prevent double-class "
            "assignment on lines containing both keywords."
        )


class TestQueueStatusBar:
    """Tests for QueueStatusBar component."""

    def test_html_contains_queue_elements(self):
        html = QueueStatusBar.html()
        assert "queue-status" in html
        assert "progress-fill" in html
        assert "btn-kill" in html
        assert "btn-clear" in html

    def test_css_returns_string(self):
        css = QueueStatusBar.css()
        assert ".queue-status" in css
        assert ".progress-bar" in css

    def test_css_all_selectors_defined(self):
        """QueueStatusBar CSS must define every selector used in its HTML.

        Losing any selector silently breaks the component layout (e.g. hiding
        the progress bar or action buttons when their class is missing from
        CSS but present in markup). Each selector below must appear as a
        substring so regression cannot remove one without the test catching it.
        """
        css = QueueStatusBar.css()
        assert ".queue-info" in css
        assert ".progress-container" in css
        assert ".progress-fill" in css
        assert ".queue-actions" in css


class TestProgressBar:
    """Tests for ProgressBar component."""

    def test_html_contains_progress_elements(self):
        html = ProgressBar.html()
        assert "progress-bar" in html
        assert "progress-fill" in html
        assert "progress-message" in html

    def test_css_returns_string(self):
        css = ProgressBar.css()
        assert ".progress-bar" in css or "progress" in css

    def test_css_all_selectors_defined(self):
        """ProgressBar CSS must define every selector used in its HTML.

        The component uses .progress-track and .progress-info alongside the
        progress bar itself; losing either silently breaks layout on pages
        that include this component. Each selector below is a hard requirement.
        """
        css = ProgressBar.css()
        assert ".progress-track" in css
        assert ".progress-info" in css

    def test_html_progress_track_present(self):
        """ProgressBar.html must include the track container used by .progress-fill."""
        html = ProgressBar.html()
        assert "progress-track" in html


class TestNotifications:
    """Tests for Notifications component."""

    def test_html_contains_toast_region(self):
        html = Notifications.html()
        assert "toast-region" in html

    def test_css_returns_string(self):
        css = Notifications.css()
        assert ".toast" in css

    def test_js_returns_string(self):
        js = Notifications.js()
        assert "showToast" in js

    def test_toast_fade_in_animation_present(self):
        """Toasts must animate in with a fade-in effect so they feel less jarring.

        Without animation, messages pop into view instantly — which works but feels
        abrupt and lowers perceived polish. The CSS should define @keyframes fadeIn
        and apply it to .toast elements as a short ease-out transition.
        """
        css = Notifications.css()
        assert "@keyframes fadeIn" in css, (
            "Fade-in animation keyframes must be defined for toast appearances."
        )
        assert ".toast {".startswith("    .toast") or "animation:" in css, (
            "The fade-in animation must be applied to the .toast class so it actually runs."
        )

    def test_confirm_action_returns_false_when_dom_absent(self):
        """confirmAction must short-circuit with Promise.resolve(false) when the
        modal DOM is missing — otherwise callers crash on a page that hasn't
        finished loading. The guard clause checks all four required elements and
        returns early before any listener setup.
        """
        js = Notifications.js()
        assert "return Promise.resolve(false)" in js, (
            "confirmAction must return Promise.resolve(false) when DOM elements are absent."
        )

    def test_showtoast_uses_textcontent_not_innerhtml(self):
        """showToast must set toast element content via textContent, not innerHTML.

        The message parameter is user-supplied (e.g., a prompt status update).
        If showToast used el.innerHTML = message instead of el.textContent = message,
        an attacker could inject HTML/script tags into the toast region. Using
        textContent forces browser escaping of any embedded markup, preventing
        XSS via notification messages. This is observable as 'textContent' in
        the produced JS source — confirming the security contract holds.
        """
        js = Notifications.js()
        assert "el.textContent" in js, (
            "showToast must set el.textContent to prevent XSS via user-supplied messages."
        )

    def test_showtoast_default_timeout_observable(self):
        """showToast defines a default timeout of 3200ms as an observable constant.

        The auto-dismiss timeout is exposed as the third parameter's default value,
        making it inspectable from source without runtime execution. Losing this
        constant silently changes dismiss timing and makes it impossible to tune
        without modifying the function signature directly.
        """
        js = Notifications.js()
        assert "3200" in js, (
            "showToast default timeout of 3200ms must be observable as a literal "
            "in the produced JS source for visibility and tuning."
        )

    def test_showtoast_click_to_dismiss_wired(self):
        """showToast wires onclick to remove the toast element on click.

        Users need an immediate way to dismiss notifications — waiting 3.2s is
        wasteful when they've already read the message. The implementation must
        attach an onclick handler that calls el.remove() so clicking a toast
        dismisses it instantly, while the setTimeout timeout still runs for
        non-interacted toasts. Observable as 'onclick' in the produced JS source.
        """
        js = Notifications.js()
        assert "onclick" in js, (
            "showToast must attach an onclick handler so users can click-to-dismiss toasts."
        )

class TestSSEClient:
    """Tests for SSEClient component."""

    def test_js_contains_event_source(self):
        js = SSEClient.js()
        assert "EventSource" in js

    def test_js_contains_reconnect_logic(self):
        js = SSEClient.js()
        assert "connectSSE" in js
        assert "MAX_SSE_RETRIES" in js

    def test_js_exponential_backoff_with_cap(self):
        """SSE reconnection must use exponential backoff capped at a max delay.

        The implementation uses Math.min(3000 * 2^(retries-1), 30000). Losing the
        cap silently turns network blips into permanent busy-loops; losing the
        exponent turns it into linear retries that flood the server during outage.
        Both behaviors are observable as substrings in the produced JS.
        """
        js = SSEClient.js()
        assert "Math.pow(2," in js or "** 2" in js, (
            "SSE reconnection must use exponential backoff to avoid flooding a "
            "server that is still recovering."
        )
        assert "30000" in js, (
            "Retry delay must be capped; without a ceiling the client would "
            "retry indefinitely with ever-growing intervals."
        )

    def test_js_contains_retry_cap_constant(self):
        """SSEClient.js() defines MAX_SSE_RETRIES as an observable constant.

        The retry cap is exported verbatim in the produced JS string so callers
        cannot accidentally drift from the intended ceiling without noticing.
        Losing it would silently allow unbounded reconnection attempts during
        extended outages, burning resources on a dead server.
        """
        js = SSEClient.js()
        assert "MAX_SSE_RETRIES" in js, (
            "Retry cap constant MAX_SSE_RETRIES must be defined as a named "
            "constant in the produced JS."
        )

    def test_js_contains_connect_sse_function(self):
        """SSEClient.js() exports connectSSE as a callable function.

        The reconnection logic is entry-point for SSE setup; callers invoke it
        without arguments to establish or refresh the event stream. Losing its
        definition silently breaks real-time updates on all gallery pages that
        include this component.
        """
        js = SSEClient.js()
        assert "function connectSSE" in js, (
            "connectSSE function must be defined as a standalone callable in "
            "the produced JS."
        )

    def test_js_references_event_source_constructor(self):
        """SSEClient.js() references the global EventSource constructor.

        The component constructs a new EventSource to open the SSE stream;
        losing this reference silently prevents real-time updates from
        functioning, even if connectSSE remains defined.
        """
        js = SSEClient.js()
        assert "EventSource" in js, (
            "The global EventSource constructor must be referenced in the "
            "produced JS to establish SSE connections."
        )

    def test_ping_handler_neutralizes_server_pings(self):
        """Server-initiated 'ping' events must not trigger reconnection.

        The server sends periodic ping events as heartbeats; without a dedicated
        handler, each ping would fire onerror (since there is no listener for an
        unknown event type), incrementing sseRetryCount and eventually exhausting
        the retry budget even though the connection is healthy. Registering an
        empty addEventListener('ping', () => {}) short-circuits pings before they
        reach the error handler, preserving retry headroom for genuine outages.
        """
        js = SSEClient.js()
        assert (
            "addEventListener('ping'" in js or 'addEventListener("ping"' in js
        ), (
            "A ping event listener must be registered to prevent server heartbeats "
            "from exhausting the retry budget during healthy connections."
        )


class TestNavHeader:
    """Tests for NavHeader component."""

    def test_html_contains_nav(self):
        html = NavHeader.html()
        assert "nav-header" in html
        assert "/index" in html

    def test_css_returns_string(self):
        css = NavHeader.css()
        assert "nav-header" in css


class TestStyleClasses:
    """Tests for style classes."""

    def test_interactive_styles_css(self):
        css = InteractiveStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_interactive_styles_composes_all_sections(self):
        """InteractiveStyles.css() must include every sub-component's CSS section.

        The class concatenates NavHeader, Buttons, LogPanel, ProgressBar CSS and a
        body-padding rule. Losing any one of these sections silently breaks the
        gallery layout (e.g., missing button styles make controls invisible). Each
        selector must survive as a substring so regression cannot remove one without
        the test catching it — mirroring the same defensive pattern used by other
        component tests in this file.
        """
        css = InteractiveStyles.css()
        assert ".nav-header" in css, "NavHeader CSS section must be composed."
        assert ".btn-primary" in css, "Buttons CSS section must be composed."
        assert ".log-panel" in css, "LogPanel CSS section must be composed."
        assert ".progress-bar-fixed" in css, "ProgressBar CSS section must be composed."
        assert "padding-bottom:" in css, (
            "Body padding rule for fixed bars must be included so content is not "
            "hidden behind the bottom status bar."
        )

    def test_form_styles_css(self):
        """FormStyles.css() must define every form selector used by the index page.

        Losing any selector silently breaks a form element (e.g., missing .form-group
        makes labels and inputs invisible). Each selector must survive as a substring
        so regression cannot remove one without the test catching it — mirroring the
        same defensive pattern used by other component tests in this file.
        """
        css = FormStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0
        assert ".form-section" in css
        assert ".form-row" in css
        assert ".form-group" in css

    def test_gallery_styles_css(self):
        """GalleryStyles.css() must define every gallery-specific selector.

        GalleryStyles defines three gallery-specific style groups; losing any of
        them silently breaks the user's gallery UI (action bar, grammar editor,
        card actions). Each selector must survive as a substring so regression
        cannot remove one without the test catching it — mirroring the same
        defensive pattern used by other component tests in this file.
        """
        css = GalleryStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0
        assert ".grammar-section-interactive" in css
        assert ".action-bar" in css
        assert ".card-actions" in css

    def test_index_styles_css(self):
        """IndexStyles.css() must define the delete-button selector used on index cards.

        The .btn-delete class is essential — it provides per-card deletion control.
        Losing its CSS silently hides the button, removing a user-facing action.
        Each selector below is a hard requirement verified as substring in CSS output.
        """
        css = IndexStyles.css()
        assert isinstance(css, str)
        assert len(css) > 0
        assert ".btn-delete" in css
        assert "opacity: 1;" in css
