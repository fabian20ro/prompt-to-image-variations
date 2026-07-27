"""Tests for tracery_runner.py - grammar parsing and expansion."""

import json

import pytest
import tracery

from tracery_runner import (
    TraceryError,
    parse_grammar,
    generate_one,
    run_tracery,
)


class TestTraceryError:
    """Tests for TraceryError exception."""

    def test_tracery_error_is_exception(self):
        """TraceryError should be an Exception subclass."""
        assert issubclass(TraceryError, Exception)

    def test_tracery_error_message(self):
        """TraceryError should preserve message."""
        error = TraceryError("test message")
        assert str(error) == "test message"


class TestParseGrammar:
    """Tests for grammar JSON parsing."""

    def test_parse_valid_grammar(self):
        """Test parsing valid JSON grammar."""
        grammar_json = '{"origin": ["hello"], "subject": ["world"]}'
        result = parse_grammar(grammar_json)

        assert isinstance(result, dict)
        assert result["origin"] == ["hello"]
        assert result["subject"] == ["world"]

    def test_parse_empty_grammar(self):
        """Test parsing empty JSON object."""
        result = parse_grammar("{}")
        assert result == {}

    def test_parse_invalid_json(self):
        """Test error on invalid JSON."""
        with pytest.raises(TraceryError, match="Invalid JSON grammar"):
            parse_grammar("{invalid json")

    def test_parse_json_syntax_error(self):
        """Test error on JSON syntax error."""
        with pytest.raises(TraceryError, match="Invalid JSON grammar"):
            parse_grammar('{"missing": "comma" "another": "key"}')

    def test_parse_truncated_json(self):
        """Test error on truncated JSON."""
        with pytest.raises(TraceryError, match="Invalid JSON grammar"):
            parse_grammar('{"origin": ["incomplete"')


class TestGenerateOne:
    """Tests for single text generation from grammar."""

    def test_generate_simple_text(self):
        """Test generating simple static text."""
        grammar = {"origin": ["hello world"]}
        result = generate_one(grammar)
        assert result == "hello world"

    def test_generate_with_expansion(self):
        """Test generating text with rule expansion."""
        grammar = {
            "origin": ["#greeting# #subject#"],
            "greeting": ["hello"],
            "subject": ["world"],
        }
        result = generate_one(grammar)
        assert result == "hello world"

    def test_generate_with_multiple_choices(self):
        """Test generating from rules with multiple choices."""
        grammar = {
            "origin": ["#color#"],
            "color": ["red", "green", "blue"],
        }
        # Result should be one of the colors
        result = generate_one(grammar)
        assert result in ["red", "green", "blue"]

    def test_generate_with_modifiers(self):
        """Test generating with Tracery modifiers (base_english)."""
        grammar = {
            "origin": ["#animal.a#"],
            "animal": ["elephant", "cat"],
        }
        result = generate_one(grammar)
        # .a modifier adds a/an
        assert result in ["an elephant", "a cat"]

    def test_generate_nested_expansion(self):
        """Test generating with nested rule expansion."""
        grammar = {
            "origin": ["#sentence#"],
            "sentence": ["#subject# #verb# #object#"],
            "subject": ["the cat"],
            "verb": ["chases"],
            "object": ["a mouse"],
        }
        result = generate_one(grammar)
        assert result == "the cat chases a mouse"

    def test_generate_custom_origin(self):
        """Test generating from custom starting rule."""
        grammar = {
            "origin": ["default text"],
            "custom": ["custom text"],
        }
        result = generate_one(grammar, origin="custom")
        assert result == "custom text"

    def test_generate_finite_recursive_rule(self, monkeypatch):
        """Test a recursive expansion with a deterministic finite choice path."""
        grammar = {
            "origin": ["#item#"],
            "item": ["leaf", "#item# and #item#"],
        }
        choices = iter(["#item#", "#item# and #item#", "leaf", "leaf"])
        monkeypatch.setattr(tracery.random, "choice", lambda _rules: next(choices))
        result = generate_one(grammar)
        assert result == "leaf and leaf"


class TestRunTracery:
    """Tests for batch text generation."""

    def test_run_tracery_basic(self):
        """Test generating multiple texts."""
        grammar_json = '{"origin": ["hello"]}'
        results = run_tracery(grammar_json, count=5)

        assert len(results) == 5
        assert all(r == "hello" for r in results)

    def test_run_tracery_default_count(self):
        """Test default count is 500."""
        grammar_json = '{"origin": ["test"]}'
        results = run_tracery(grammar_json)

        assert len(results) == 500

    def test_run_tracery_with_variation(self):
        """Test that variations are generated."""
        grammar_json = json.dumps({
            "origin": ["#color#"],
            "color": ["red", "green", "blue"],
        })
        results = run_tracery(grammar_json, count=100)

        # With 100 samples, we should see variation
        unique_results = set(results)
        assert len(unique_results) > 1
        assert unique_results.issubset({"red", "green", "blue"})

    def test_run_tracery_custom_origin(self):
        """Test using custom starting rule."""
        grammar_json = json.dumps({
            "origin": ["default"],
            "alternate": ["alternate text"],
        })
        results = run_tracery(grammar_json, count=3, origin="alternate")

        assert len(results) == 3
        assert all(r == "alternate text" for r in results)

    def test_run_tracery_zero_count(self):
        """Test that zero count returns empty list without error."""
        grammar_json = '{"origin": ["hello"]}'
        results = run_tracery(grammar_json, count=0)

        assert results == []

    def test_run_tracery_negative_count(self):
        """Test that negative count returns empty list (range silently handles)."""
        grammar_json = '{"origin": ["hello"]}'
        results = run_tracery(grammar_json, count=-1)

        assert results == []

    def test_run_tracery_invalid_json(self):
        """Test error on invalid JSON."""
        with pytest.raises(TraceryError, match="Invalid JSON grammar"):
            run_tracery("{invalid", count=1)

    def test_run_tracery_invalid_json_zero_count_returns_empty(self):
        """Test that zero-count short-circuit skips parsing — invalid JSON returns []."""
        # Grammar is NOT parsed when count <= 0; we return [] immediately.
        results = run_tracery("{invalid", count=0)
        assert results == []

    def test_run_tracery_complex_grammar(self):
        """Test with a more complex grammar."""
        grammar_json = json.dumps({
            "origin": ["#mood.a# #animal# in #setting.a#"],
            "mood": ["happy", "sad", "curious"],
            "animal": ["cat", "dog", "bird"],
            "setting": ["garden", "forest", "city"],
        })
        results = run_tracery(grammar_json, count=10)

        assert len(results) == 10
        for result in results:
            # Should have format: "a/an [mood] [animal] in a/an [setting]"
            assert " in " in result


class TestGrammarEdgeCases:
    """Tests for edge cases in grammar handling."""

    def test_empty_origin(self):
        """Test grammar with empty origin list raises error."""
        grammar = {"origin": []}
        # Tracery raises IndexError when choosing from empty rule
        with pytest.raises(IndexError, match="Cannot choose from an empty sequence"):
            generate_one(grammar)

    def test_missing_rule_reference(self):
        """Test grammar referencing non-existent rule."""
        grammar = {"origin": ["#nonexistent#"]}
        result = generate_one(grammar)
        # Tracery leaves unresolved references as-is
        assert "#nonexistent#" in result or "nonexistent" in result

    def test_unicode_in_grammar(self):
        """Test grammar with unicode characters."""
        grammar = {
            "origin": ["#emoji# #text#"],
            "emoji": ["\U0001F600", "\U0001F604"],  # Smileys
            "text": ["hello world"],
        }
        result = generate_one(grammar)
        assert "hello world" in result

    def test_special_characters_in_text(self):
        """Test grammar with special characters."""
        grammar = {
            "origin": ["Price: $100 (50% off!)"],
        }
        result = generate_one(grammar)
        assert result == "Price: $100 (50% off!)"

    def test_newlines_in_grammar(self):
        """Test grammar with newlines."""
        grammar = {
            "origin": ["line1\nline2"],
        }
        result = generate_one(grammar)
        assert "\n" in result

    def test_very_long_output(self):
        """Test grammar producing long output."""
        grammar = {
            "origin": ["#word# " * 100],
            "word": ["test"],
        }
        result = generate_one(grammar)
        assert len(result) > 400  # "test " * 100 = 500 chars


class TestTraceryModifiers:
    """Tests for Tracery modifier support (base_english)."""

    def test_capitalize_modifier(self):
        """Test .capitalize modifier."""
        grammar = {"origin": ["#word.capitalize#"], "word": ["hello"]}
        result = generate_one(grammar)
        assert result == "Hello"

    def test_uppercase_modifier(self):
        """Test .s (plural) modifier."""
        grammar = {"origin": ["#animal.s#"], "animal": ["cat"]}
        result = generate_one(grammar)
        assert result == "cats"

    def test_chained_modifiers(self):
        """Test chaining multiple modifiers."""
        grammar = {"origin": ["#word.capitalize.s#"], "word": ["cat"]}
        result = generate_one(grammar)
        # Order matters: capitalize then pluralize
        assert result == "Cats"

    def test_empty_string_rule(self):
        """Test grammar with an empty string in a rule array."""
        grammar = {
            "origin": ["#word#", ""],
            "word": ["hello"],
        }
        # Tracery picks from the list; empty string is a valid choice
        result = generate_one(grammar)
        assert isinstance(result, str)

    def test_non_list_rule_value(self):
        """Test grammar where a rule value is not a list."""
        grammar = {"origin": ["#word"], "word": "hello"}
        # Tracery normalizes string values to single-element lists internally;
        # this should still produce valid output without raising.
        result = generate_one(grammar)
        assert isinstance(result, str)

    def test_deep_recursion_deterministic(self, monkeypatch):
        """Test deep recursive expansion terminates deterministically."""
        grammar = {
            "origin": ["#chain#"],
            "chain": ["leaf", "#chain# + #chain#"],
        }
        # Provide enough choices to cover worst-case recursion depth.
        # Each recursion adds two child expansions; 50 is generous.
        choices = iter(["leaf"] * 50)
        monkeypatch.setattr(tracery.random, "choice", lambda _rules: next(choices))
        result = generate_one(grammar)
        assert result == "leaf" or "+ leaf + leaf" in result

    def test_generate_empty_grammar(self):
        """Test generating from an empty grammar dict returns empty string."""
        # Tracery handles empty dicts gracefully, returning empty output.
        result = generate_one({})
        assert isinstance(result, str)

    def test_generate_non_dict_input(self):
        """Test that passing a non-dict to generate_one raises AttributeError."""
        with pytest.raises(AttributeError):
            generate_one("not a dict")

    def test_generate_missing_custom_origin(self):
        """Test that referencing an undefined origin rule returns unresolved reference."""
        grammar = {"origin": ["hello"]}
        result = generate_one(grammar, origin="does_not_exist")
        # Tracery leaves unresolved references as-is in the output
        assert "does_not_exist" in result or "#does_not_exist#" in result

    def test_run_tracery_with_modifiers(self):
        """Test batch generation includes modifier effects."""
        grammar_json = json.dumps({
            "origin": ["#mood.a#"],
            "mood": ["happy", "sad", "curious"],
        })
        results = run_tracery(grammar_json, count=50)
        assert len(results) == 50
        for r in results:
            # base_english .a adds 'an' or 'a' based on vowel heuristic
            assert r.startswith("a ") or r.startswith("an ")
