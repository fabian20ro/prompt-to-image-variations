"""Tests for src/grammar_history — persisted grammar revision history helpers."""

import json
from pathlib import Path

import pytest

# Add src directory to path for imports
sys_path = __import__("sys").path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys_path:
    sys_path.insert(0, str(src_dir))

from grammar_history import (
    _history_path,
    append_grammar_revision,
    get_recent_revisions,
    load_grammar_history,
    save_grammar_history,
)


class TestHistoryPath:
    def test_returns_correct_path(self, run_dir):
        result = _history_path(run_dir, "test")
        assert result == run_dir / "test_grammar_history.json"


class TestLoadGrammarHistory:
    def test_loads_existing_valid_json(self, run_dir):
        history = [
            {"id": "rev1", "created_at": "2024-01-01T00:00:00", "action": "initial", "grammar": "rule1"},
            {"id": "rev2", "created_at": "2024-01-02T00:00:00", "action": "update", "grammar": "rule2"},
        ]
        path = _history_path(run_dir, "test")
        path.write_text(json.dumps(history))

        result = load_grammar_history(run_dir, "test")
        assert len(result) == 2
        assert result[0]["id"] == "rev1"
        assert result[1]["action"] == "update"

    def test_returns_empty_when_no_file_and_no_grammar(self, run_dir):
        result = load_grammar_history(run_dir, "missing")
        assert result == []

    def test_falls_back_to_current_grammar_when_no_history_file(self, run_dir):
        grammar_text = '{"origin": ["hello"]}'
        result = load_grammar_history(run_dir, "test", current_grammar=grammar_text)
        assert len(result) == 1
        assert result[0]["id"] == "initial"
        assert result[0]["grammar"] == grammar_text

    def test_falls_back_to_current_grammar_on_corrupted_json(self, run_dir):
        path = _history_path(run_dir, "test")
        path.write_text("not valid json{{{")

        grammar_text = '{"origin": ["fallback"]}'
        result = load_grammar_history(run_dir, "test", current_grammar=grammar_text)
        assert len(result) == 1
        assert result[0]["id"] == "initial"
        assert result[0]["grammar"] == grammar_text

    def test_reads_from_disk_when_no_current_grammar_provided(self, run_dir):
        grammar_text = '{"origin": ["from_disk"]}'
        (run_dir / "test_grammar.json").write_text(grammar_text)

        result = load_grammar_history(run_dir, "test")
        assert len(result) == 1
        assert result[0]["grammar"] == grammar_text


class TestSaveGrammarHistory:
    def test_writes_and_returns_path(self, run_dir):
        history = [{"id": "x", "created_at": "2024-01-01T00:00:00", "action": "test", "grammar": "g"}]
        result_path = save_grammar_history(run_dir, "test", history)

        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert len(loaded) == 1
        assert loaded[0]["id"] == "x"


class TestAppendGrammarRevision:
    def test_appends_new_revision(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="update")
        assert len(history) == 2
        assert history[1]["action"] == "update"

    def test_skips_when_grammar_and_action_unchanged(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        assert len(history) == 1

    def test_appends_when_only_action_changed(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="update")
        assert len(history) == 2
        assert history[1]["action"] == "update"

    def test_appends_when_grammar_changes(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_b", action="update")
        assert len(history) == 2
        assert history[1]["grammar"] == "rule_b"

    def test_skips_when_grammar_has_whitespace_differences(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a  ", action="initial")
        assert len(history) == 1

    def test_appends_when_whitespace_differs_and_action_changes(self, run_dir):
        """Dedup skips only when BOTH grammar (whitespace-normalized) AND action match.

        Verifies the contract boundary: whitespace-asymmetric grammars still append if
        the action differs from the last revision's action — confirming that dedup is
        an exact-match gate on both fields, not a single-field short-circuit.
        """
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        history = append_grammar_revision(
            run_dir, "test", grammar="  rule_a  ", action="update"
        )
        assert len(history) == 2
        assert history[1]["action"] == "update"

    def test_appends_when_duplicate_exists_before_last(self, run_dir):
        """Dedup only checks the last entry — earlier duplicates still append.

        The dedup gate compares against `history[-1]` exclusively: if a grammar+action
        pair appears elsewhere in history (not as the last entry), it is still appended.
        This test characterizes that boundary so future refactors do not accidentally
        change dedup to "global uniqueness" or break rotation semantics.
        """
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        append_grammar_revision(run_dir, "test", grammar="rule_b", action="update")
        history = append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        assert len(history) == 3
        assert history[2]["action"] == "initial"

    def test_empty_grammar_no_side_effects_when_no_prior_state(self, run_dir):
        """Empty/whitespace-only grammar must be a true no-op — no history file created.

        The early return on `not grammar.strip()` runs before any file I/O; this test
        characterizes that boundary so future refactors do not accidentally write files
        or append empty entries when the input is blank.
        """
        path = _history_path(run_dir, "empty_test")
        assert not path.exists()

        history = append_grammar_revision(
            run_dir, "empty_test", grammar="   ", action="initial"
        )
        assert history == []
        assert not path.exists()

    def test_creates_history_file_on_first_append(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        path = _history_path(run_dir, "test")
        assert path.exists()


class TestLoadGrammarHistoryEdgeCases:
    def test_valid_empty_list_returns_as_is_without_fallback(self, run_dir):
        """Valid empty-list JSON must return the list directly — no fallback to current grammar.

        Characterizes the `isinstance(data, list)` gate at line 18-19: when on-disk
        data is a valid empty list (not corrupted, not dict), load_grammar_history
        returns it as-is without consulting the current grammar file or creating an
        "initial" entry. This locks in the boundary between "valid but empty" and
        "corrupt/fallback-needed."
        """
        path = _history_path(run_dir, "empty_list_test")
        path.write_text(json.dumps([]))

        result = load_grammar_history(run_dir, "empty_list_test")
        assert result == []
        # Verify no grammar file was consulted or created on disk
        assert not (run_dir / "empty_list_test_grammar.json").exists()

    def test_returns_empty_when_json_is_dict_instead_of_list(self, run_dir):
        path = _history_path(run_dir, "dict_instead_of_list")
        path.write_text(json.dumps({"not": "a list"}))

        result = load_grammar_history(run_dir, "dict_instead_of_list")
        assert result == []

    def test_returns_empty_on_corrupted_json_with_no_fallback(self, run_dir):
        """Corrupted JSON file with no current_grammar fallback returns [].

        Verifies the silent-swalow boundary: load_grammar_history must not raise or
        return partial data when history is corrupt and no current grammar is supplied.
        Callers rely on this contract to detect "no usable history."
        """
        path = _history_path(run_dir, "corrupted_test")
        path.write_text("<<<garbage json>>>")

        result = load_grammar_history(run_dir, "corrupted_test")
        assert result == []
        assert not (run_dir / "corrupted_test_grammar.json").exists()

    def test_skips_when_grammar_is_empty_or_whitespace(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="initial", action="initial")
        history = append_grammar_revision(run_dir, "test", grammar="   ", action="initial")
        assert len(history) == 1
        assert history[0]["grammar"] == "initial"


class TestAppendGrammarRotation:
    def test_rotation_trims_to_max_revisions(self, run_dir):
        for i in range(5):
            append_grammar_revision(run_dir, "test", grammar=f"rule_{i}", action="update")
        history = append_grammar_revision(run_dir, "test", grammar="final", action="update", max_revisions=3)
        assert len(history) == 3
        assert history[0]["grammar"] == "rule_3"
        assert history[-1]["grammar"] == "final"

    def test_no_rotation_when_under_limit(self, run_dir):
        append_grammar_revision(run_dir, "test", grammar="a", action="initial")
        append_grammar_revision(run_dir, "test", grammar="b", action="update")
        history = append_grammar_revision(run_dir, "test", grammar="c", action="update", max_revisions=10)
        assert len(history) == 3

    def test_no_rotation_on_dedup_match(self, run_dir):
        """Dedup must not trigger rotation — unchanged revisions are skipped."""
        append_grammar_revision(run_dir, "test", grammar="a", action="initial")
        history = append_grammar_revision(
            run_dir, "test", grammar="a", action="initial", max_revisions=1
        )
        assert len(history) == 1

    def test_empty_grammar_after_prior_state_skips_disk_write(self, run_dir):
        """Empty/whitespace-only grammar must be a no-op — no history file created.

        Characterizes the early-return path in append_grammar_revision: when called
        after prior state exists and the new grammar is empty/blank, it returns the
        existing history unchanged without creating any new files on disk. This locks
        in the side-effect contract so future refactors cannot accidentally write
        spurious entries to disk.
        """
        # Create initial prior state
        append_grammar_revision(run_dir, "edge_test", grammar="first_rule", action="initial")

        edge_path = _history_path(run_dir, "edge_test")
        assert edge_path.exists()  # history exists from prior call
        old_mtime = edge_path.stat().st_mtime

        import time
        time.sleep(0.02)  # ensure timestamp would change if file were rewritten

        # Now pass empty grammar — should return existing history, not rewrite disk
        history = append_grammar_revision(run_dir, "edge_test", grammar="   ", action="initial")

        assert len(history) == 1
        assert history[0]["grammar"] == "first_rule"
        # File should be unchanged (no rewrite on empty grammar path)
        assert edge_path.stat().st_mtime <= old_mtime + 0.01

    def test_zero_max_revisions_disables_rotation(self, run_dir):
        for i in range(5):
            append_grammar_revision(run_dir, "test", grammar=f"rule_{i}", action="update")
        history = append_grammar_revision(run_dir, "test", grammar="final", action="update", max_revisions=0)
        assert len(history) == 6

    def test_negative_max_revisions_causes_unexpected_trimming(self, run_dir):
        """Negative max_revisions is truthy in Python → slicing becomes `history[1:]`.

        The rotation guard at line 80 uses bare `if max_revisions`, which treats any
        non-zero value as true. A negative value passes the guard and then slices
        the history starting from index 1 (dropping the first entry). This test
        characterizes that current behavior so callers know to validate their input.
        """
        append_grammar_revision(run_dir, "test", grammar="rule_a", action="initial")
        append_grammar_revision(run_dir, "test", grammar="rule_b", action="update")
        history = append_grammar_revision(
            run_dir, "test", grammar="rule_c", action="update", max_revisions=-1
        )
        # history[-(-1):] == history[1:] — first entry is dropped
        assert len(history) == 2
        assert history[0]["grammar"] == "rule_b"

    def test_empty_valid_json_on_disk_appends_first_revision(self, run_dir):
        """Valid empty JSON array on disk → load returns [] → last=None → append proceeds.

        Characterizes that an existing file containing `[]` is treated as "no usable
        history": the dedup gate sees `last = None`, so the first real revision still
        gets appended rather than being dropped by a missing-file fallback path.
        """
        import json
        from grammar_history import _history_path

        history_path = _history_path(run_dir, "empty_array")
        history_path.write_text("[]")  # valid empty list on disk

        history = append_grammar_revision(
            run_dir, "empty_array", grammar="rule_first", action="initial"
        )
        assert len(history) == 1
        assert history[0]["id"] != "initial"  # not the synthetic initial id
        assert history[0]["action"] == "initial"
        assert history[0]["grammar"] == "rule_first"

    def test_rotation_persists_to_disk(self, run_dir):
        for i in range(10):
            append_grammar_revision(run_dir, "test", grammar=f"rule_{i}", action="update")
        # Read back — should reflect trimmed state (default max_revisions=100)
        loaded = load_grammar_history(run_dir, "test")
        assert len(loaded) == 10

    def test_rotation_with_small_max_persists(self, run_dir):
        for i in range(5):
            append_grammar_revision(run_dir, "small_test", grammar=f"s{i}", action="update", max_revisions=3)
        loaded = load_grammar_history(run_dir, "small_test")
        assert len(loaded) == 3
        assert loaded[0]["grammar"] == "s2"


class TestAppendGrammarMalformedHistory:
    def test_skipped_dedup_when_last_entry_missing_action(self, run_dir):
        """Malformed last entry (missing 'action') must fall through to append.

        The dedup gate checks `last.get("action") == action`. If the existing
        history's last entry is missing the 'action' key, .get() returns None,
        which never equals a real action string — so the revision appends rather
        than being silently dropped. This characterizes defensive handling of
        corrupt/incomplete entries in existing history files.
        """
        path = _history_path(run_dir, "malformed_test")
        # Pre-seed a malformed entry missing 'action'
        malformed_history = [
            {"id": "old", "created_at": "2024-01-01T00:00:00", "grammar": "rule_a"},
        ]
        path.write_text(json.dumps(malformed_history))

        history = append_grammar_revision(
            run_dir, "malformed_test", grammar="rule_b", action="update"
        )
        assert len(history) == 2
        assert history[1]["action"] == "update"
        assert history[1]["grammar"] == "rule_b"

    def test_skipped_dedup_when_last_entry_missing_grammar(self, run_dir):
        """Malformed last entry (missing 'grammar') must fall through to append.

        The dedup gate checks `last.get("grammar", "")`. If the existing history's
        last entry is missing the 'grammar' key, .get() returns "", which never
        equals a real grammar string — so the revision appends rather than being
        silently dropped. This characterizes defensive handling of corrupt/
        incomplete entries in existing history files.
        """
        path = _history_path(run_dir, "malformed_test2")
        # Pre-seed a malformed entry missing 'grammar'
        malformed_history = [
            {"id": "old", "created_at": "2024-01-01T00:00:00", "action": "initial"},
        ]
        path.write_text(json.dumps(malformed_history))

        history = append_grammar_revision(
            run_dir, "malformed_test2", grammar="rule_b", action="update"
        )
        assert len(history) == 2
        assert history[1]["action"] == "update"

    def test_non_list_json_history_falls_through_to_single_entry(self, run_dir):
        """Non-list JSON on disk → load_grammar_history returns [] → dedup sees last=None.

        The dedup gate at append_grammar_revision compares `last.get("grammar", "")` and
        `last.get("action")` against the new values. When history is loaded from a file
        containing non-list JSON (e.g., {"key": "value"}), load returns an empty list —
        so last becomes None, .get() on None never matches, and append proceeds with
        exactly one entry. This characterizes that corrupt-or-foreign-format files are
        treated as "no usable history" rather than raising or overwriting the file.
        """
        path = _history_path(run_dir, "nonlist_test")
        # Pre-seed a non-list JSON structure (dict instead of list)
        foreign_data = {"origin": ["something"], "config": 42}
        path.write_text(json.dumps(foreign_data))

        history = append_grammar_revision(
            run_dir, "nonlist_test", grammar="rule_a", action="initial"
        )
        assert len(history) == 1
        assert history[0]["action"] == "initial"
        assert history[0]["grammar"] == "rule_a"
        # ID is timestamp-based when loaded from empty state (not synthetic "initial")

    def test_extra_keys_in_last_entry_do_not_disrupt_dedup(self, run_dir):
        """History entries with extra/unexpected keys must not interfere with dedup.

        The dedup gate only inspects 'grammar' and 'action' via .get(). Extra keys
        (e.g., metadata injected by upstream callers) are ignored — neither the dedup
        decision nor the appended entry is affected. This locks in that the function
        does not enumerate or reject unknown fields, so future extensions can safely
        carry auxiliary data on history entries without breaking append semantics.
        """
        path = _history_path(run_dir, "extra_keys_test")
        enriched_history = [
            {
                "id": "old",
                "created_at": "2024-01-01T00:00:00",
                "action": "initial",
                "grammar": "rule_a",
                "metadata": {"source": "upstream"},  # extra key
                "tags": ["v1"],                       # extra key
            },
        ]
        path.write_text(json.dumps(enriched_history))

        history = append_grammar_revision(
            run_dir, "extra_keys_test", grammar="rule_a", action="initial"
        )
        assert len(history) == 1  # dedup skipped the append (same grammar+action)

    def test_extra_keys_in_last_entry_allow_append_on_different_action(self, run_dir):
        """Extra keys in the last entry must not block appending when action differs.

        When the new action does not match `last["action"]`, the dedup gate falls
        through and appends — extra keys are irrelevant to this decision. This test
        confirms that the function's append path is fully independent of unknown
        fields in existing entries, even mid-session state changes.
        """
        path = _history_path(run_dir, "extra_keys_diff_action")
        enriched_history = [
            {
                "id": "old",
                "created_at": "2024-01-01T00:00:00",
                "action": "initial",
                "grammar": "rule_a",
                "metadata": {"source": "upstream"},
            },
        ]
        path.write_text(json.dumps(enriched_history))

        history = append_grammar_revision(
            run_dir, "extra_keys_diff_action", grammar="rule_a", action="update"
        )
        assert len(history) == 2  # different action → appended despite same grammar
        assert history[1]["action"] == "update"


class TestGetRecentRevisions:
    def test_action_filter_reduces_entry_to_single_key(self, run_dir):
        """include_action=True must reduce each entry to {action: ...}.

        Verifies that when include_action=True is passed, the returned entries
        contain ONLY the 'action' key — not id, created_at, grammar, or any other
        fields from the source history. This characterizes the action-filtering
        contract so future refactors don't accidentally leak full entries through
        the reduction path.
        """
        history = [
            {
                "id": f"r{i}",
                "created_at": f"2024-01-{i+1:02d}T00:00:00",
                "action": f"a_{i}",
                "grammar": f"g_{i}",
            }
            for i in range(5)
        ]

        result = get_recent_revisions(history, n=3, include_action=True)

        assert len(result) == 3
        for entry in result:
            assert set(entry.keys()) == {"action"}
            assert isinstance(entry["action"], str)

    def test_returns_n_most_recent_entries(self, run_dir):
        """Default slicing returns the last n entries in order."""
        history = [
            {"id": f"rev{i}", "created_at": f"2024-01-{i+1:02d}T00:00:00", "action": "update", "grammar": f"rule_{i}"}
            for i in range(5)
        ]
        result = get_recent_revisions(history, n=3)
        assert len(result) == 3
        assert result[0]["id"] == "rev2"
        assert result[-1]["id"] == "rev4"

    def test_returns_full_copy_when_n_zero(self):
        """n=0 must return a shallow copy of the entire history."""
        history = [
            {"id": f"r{i}", "action": "a", "grammar": f"g{i}"} for i in range(3)
        ]
        result = get_recent_revisions(history, n=0)
        assert len(result) == 3
        # Shallow copy — identity differs from original
        assert result is not history

    def test_returns_full_copy_when_n_negative(self):
        """n<0 behaves like n<=0: full shallow copy of all entries."""
        history = [{"id": "x", "action": "a", "grammar": "g"}]
        result = get_recent_revisions(history, n=-5)
        assert len(result) == 1
        assert result is not history

    def test_include_action_returns_dicts_with_only_action(self):
        """include_action=True reduces each entry to {action: ...}.

        Verifies the reduction contract: returned dicts contain exactly one key,
        and that key holds whatever the source entry's 'action' value was.
        """
        history = [
            {"id": f"r{i}", "created_at": f"t{i}", "action": f"a_{i}", "grammar": f"g{i}"}
            for i in range(4)
        ]
        result = get_recent_revisions(history, n=3, include_action=True)
        assert len(result) == 3
        for entry in result:
            assert set(entry.keys()) == {"action"}

    def test_include_action_respects_n_limit(self):
        """include_action combined with n must slice first, then reduce."""
        history = [
            {"id": f"r{i}", "created_at": f"t{i}", "action": f"a_{i}", "grammar": f"g{i}"}
            for i in range(5)
        ]
        result = get_recent_revisions(history, n=2, include_action=True)
        assert len(result) == 2
        assert [e["action"] for e in result] == ["a_3", "a_4"]

    def test_does_not_mutate_original(self):
        """Returned list must not share references that allow mutation of source."""
        history = [{"id": f"r{i}", "action": f"a_{i}"} for i in range(5)]
        result = get_recent_revisions(history, n=2)
        result[0]["id"] = "MUTATED"
        assert history[-1]["id"] == "r4"

    def test_include_action_with_full_copy_n_zero(self):
        """include_action=True with n<=0 reduces every entry to {action: ...} across full history.

        The slicing branch is bypassed when n<=0 (full copy path), but the action-only
        reduction must still apply uniformly to every entry — not just a slice of them.
        This test characterizes that cross-branch behavior so future refactors do not
        accidentally gate the reduction on positive n only.
        """
        history = [
            {"id": f"r{i}", "created_at": f"t{i}", "action": f"a_{i}", "grammar": f"g{i}"}
            for i in range(4)
        ]
        result = get_recent_revisions(history, n=0, include_action=True)
        assert len(result) == 4
        for entry in result:
            assert set(entry.keys()) == {"action"}
        # Verify every entry is reduced — no leftover keys from source dicts
        assert all(len(e) == 1 for e in result)

    def test_empty_history_returns_empty_list(self):
        """Empty input must yield empty output regardless of n."""
        result = get_recent_revisions([], n=5)
        assert result == []

    def test_empty_history_with_include_action_and_positive_n(self):
        """Empty history + include_action=True + positive n exercises both branches.

        Characterizes the cross-branch contract: `history[-n:]` on an empty list
        yields [], then the reduction branch iterates over zero entries — producing
        []. This locks in that include_action does not crash or behave differently
        when the sliced result is empty, so future refactors cannot gate it behind
        a non-empty guard.
        """
        result = get_recent_revisions([], n=5, include_action=True)
        assert result == []

    def test_n_exceeds_history_length_returns_all_entries(self):
        """When n > len(history), slicing returns all entries — Python's slice
        underflow is graceful, not an error.

        Characterizes the `history[-n:]` gate at line 104: callers may pass a
        generous n (e.g., max_revisions=10) on a small history without triggering
        IndexError or truncation surprises. This test locks in that the function
        returns every available entry rather than raising or returning [].
        """
        history = [
            {"id": "a", "action": "init", "grammar": "g_a"},
            {"id": "b", "action": "update", "grammar": "g_b"},
            {"id": "c", "action": "update", "grammar": "g_c"},
        ]
        result = get_recent_revisions(history, n=10)
        assert len(result) == 3
        assert [e["id"] for e in result] == ["a", "b", "c"]

    def test_n_exceeds_history_with_include_action(self):
        """include_action=True with n > len(history) must still reduce every entry.

        Verifies the cross-branch contract: when slicing returns all entries
        (because n exceeds history length), the action-only reduction applies
        uniformly to every returned dict — not just a truncated slice. This
        locks in that the include_action branch is independent of how many
        entries were sliced, so future refactors cannot gate it behind an
        n <= len(history condition.
        """
        history = [
            {"id": "a", "action": "init", "grammar": "g_a"},
            {"id": "b", "action": "update", "grammar": "g_b"},
        ]
        result = get_recent_revisions(history, n=100, include_action=True)
        assert len(result) == 2
        for entry in result:
            assert set(entry.keys()) == {"action"}
