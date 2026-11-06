"""Persisted grammar revision history helpers."""

import json
from datetime import datetime
from pathlib import Path


def _history_path(run_dir: Path, prefix: str) -> Path:
    return run_dir / f"{prefix}_grammar_history.json"


def load_grammar_history(run_dir: Path, prefix: str, current_grammar: str | None = None) -> list[dict]:
    """Load grammar history, falling back to a single current revision."""
    history_path = _history_path(run_dir, prefix)
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    if current_grammar is None:
        grammar_path = run_dir / f"{prefix}_grammar.json"
        current_grammar = grammar_path.read_text(encoding="utf-8") if grammar_path.exists() else ""
    
    if not current_grammar:
        return []

    return [{
        "id": "initial",
        "created_at": datetime.now().isoformat(),
        "action": "initial",
        "grammar": current_grammar,
    }]


def save_grammar_history(run_dir: Path, prefix: str, history: list[dict]) -> Path:
    """Write grammar history to disk."""
    history_path = _history_path(run_dir, prefix)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history_path


def append_grammar_revision(
    run_dir: Path,
    prefix: str,
    grammar: str,
    action: str,
    max_revisions: int = 100,
) -> list[dict]:
    """Append a revision if the grammar changed or action is new.

    Args:
        run_dir: Directory where history files are stored.
        prefix: Filename prefix for history and grammar files.
        grammar: The grammar string to record as a revision.
        action: Semantic label for this revision (e.g. "initial", "update").
        max_revisions: Trim the history to this many recent entries when saving.
            Defaults to 100 to cap disk usage in long sessions.

    Returns:
        The updated history list.
    """
    if not grammar.strip():
        return load_grammar_history(run_dir, prefix)
    history_path = _history_path(run_dir, prefix)
    history_exists = history_path.exists()
    history = load_grammar_history(run_dir, prefix)
    last = history[-1] if history else None
    if last and last.get("grammar", "").strip() == grammar.strip() and last.get("action") == action:
        return history

    history.append({
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "created_at": datetime.now().isoformat(),
        "action": action,
        "grammar": grammar,
    })
    if max_revisions and len(history) > max_revisions:
        history = history[-max_revisions:]
    save_grammar_history(run_dir, prefix, history)
    return history


def get_recent_revisions(
    history: list[dict], n: int = 10, *, include_action: bool | None = False,
    action_filter: str | None = None,
) -> list[dict]:
    """Return the last ``n`` entries from ``history``.

    Args:
        history: The full grammar-history list (not mutated).
        n: Number of recent entries to return. Defaults to 10. Pass ``0`` or a
            negative value to receive a shallow copy of the entire history.
        include_action: When True, each returned entry is reduced to only its
            ``action`` key; when False (default), full entries are returned.
        action_filter: Optional filter — only entries whose ``action`` field equals
            this string are included in the result. A non-matching value returns an
            empty list. Defaults to ``None`` (no filtering).

    Returns:
        A new list containing at most ``n`` recent revisions (or all entries).
    """
    if n <= 0:
        result = list(history)
    else:
        result = history[-n:]

    if action_filter is not None and isinstance(result, list):
        result = [e for e in result if e.get("action") == action_filter]

    if include_action and isinstance(result, list):
        return [{"action": e.get("action")} for e in result]

    return result
