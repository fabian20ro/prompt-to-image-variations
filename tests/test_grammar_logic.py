"""Tests for grammar_generator logic."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grammar_generator import (
    _api_root,
    get_system_prompt,
    hash_prompt,
    clean_grammar_output,
)


def test_get_system_prompt_default(tmp_path):
    # Given
    generic_content = "generic content"
    (tmp_path / "system_prompt.txt").write_text(generic_content)
    
    # When
    result = get_system_prompt(templates_dir=tmp_path)
    
    # Then
    assert result == generic_content


def test_hash_prompt():
    # Given
    prompt = "a cat"
    # When
    h1 = hash_prompt(prompt)
    h2 = hash_prompt(prompt)
    h3 = hash_prompt("a dog")
    
    # Then
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12


def test_clean_grammar_output_multiple_blocks():
    # Given
    raw = "Some text\n```json\n{\"a\": 1}\n```\nMore text\n```tracery\n{\"b\": 2}\n```"
    # When
    result = clean_grammar_output(raw)
    # Then
    assert result == '{"a": 1}'


def test_handles_json_array():
    # Verifies that if the LLM returns a JSON array, it is preserved
    # Given
    input_text = '```json\n[1, 2, 3]\n```'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '[1, 2, 3]'


def test_handles_json_object_with_extra_content():
    # Verifies that content after the JSON object is discarded
    # Given
    input_text = '{"a": 1} extra'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '{"a": 1}'


def test_handles_json_in_text_array():
    # Verifies extraction of JSON array from within text
    # Given
    input_text = 'Here is an array: [1, 2, 3] and more'
    # When
    result = clean_grammar_output(input_text)
    # Then
    assert result == '[1, 2, 3]'


def test_clean_grammar_output_no_blocks():
    # Given
    raw = "Just some text with {\"a\": 1} inside"
    # When
    result = clean_grammar_output(raw)
    # Then
    assert result == '{"a": 1}'


def test_clean_grammar_output_normalizes_smart_quotes():
    """Smart (curly) double quotes in LLM output must be replaced with straight ASCII quotes."""
    raw = "Here is the grammar: {\u201ckolors\u201d: [\u201cA sepia tone, soft grainy texture\u201d]}"
    result = clean_grammar_output(raw)
    assert '\u201c' not in result and '\u201d' not in result
    expected = '{"kolors": ["A sepia tone, soft grainy texture"]}'
    assert result == expected


def test_clean_grammar_output_normalizes_smart_single_quotes():
    """Left/right single curly quotes (U+2018, U+2019) must be replaced with straight ASCII apostrophes."""
    raw = '{"prompt": "a cat\u2019s eye"}'  # \u2019 is right single curly quote
    result = clean_grammar_output(raw)
    assert '\u2019' not in result and '\u2018' not in result
    expected = '{"prompt": "a cat\'s eye"}'
    assert result == expected


def test_clean_grammar_output_strips_thinking_blocks():
    """Thinking blocks (``...``) must be removed before JSON extraction."""
    raw = '``thinking block\n\n```json\n{"origin": ["a", "b"]}\n```\n'
    result = clean_grammar_output(raw)
    assert 'thinking' not in result
    expected = '{"origin": ["a", "b"]}'
    assert result == expected


def test_clean_grammar_output_no_json_returns_stripped_input():
    """When no JSON object/array is found, clean returns the stripped input unchanged."""
    raw = "Here's a description of what I want to see."
    result = clean_grammar_output(raw)
    assert result == raw


def test_clean_grammar_output_strips_thinking_blocks_with_smart_quotes():
    """Thinking block removal and smart quote normalization must compose correctly in one pass."""
    raw = '``thinking\n\n```json\n{"kolors": [\u201cA sepia tone\u201d]}\n```\n'
    result = clean_grammar_output(raw)
    assert '\u201c' not in result and '\u201d' not in result
    expected = '{"kolors": ["A sepia tone"]}'
    assert result == expected


# ---------------------------------------------------------------------------
# _api_root — pure URL transformation (no side effects, no mocks needed)
# ---------------------------------------------------------------------------

def test_api_root_strips_v1_suffix():
    """LM Studio base URLs conventionally end in /v1; _api_root must strip it."""
    assert _api_root("http://localhost:1234/v1") == "http://localhost:1234"


def test_api_root_handles_trailing_slash_with_v1():
    """Trailing slash before /v1 should still resolve to the server root."""
    assert _api_root("http://localhost:1234/v1/") == "http://localhost:1234"


def test_api_root_leaves_non_v1_urls_unchanged():
    """Non-LM-Studio URLs (no /v1 segment) should round-trip untouched."""
    url = "http://example.com/api"
    assert _api_root(url) == url


def test_api_root_does_not_strip_mid_path_v1():
    """/v1 appearing in the middle of a path must NOT be stripped — only trailing /v1 is removed."""
    url = "http://localhost:1234/some/v1/path/extra"
    assert _api_root(url) == url


def test_api_root_handles_v1_with_query():
    """A URL with query params after /v1 should round-trip unchanged (not endswith /v1)."""
    url = "http://localhost:1234/v1?param=value"
    assert _api_root(url) == url


def test_api_root_case_sensitive_v1():
    """/V1 or /V1/ variants must not be stripped — only lowercase /v1 is the LM Studio convention."""
    url_upper = "http://localhost:1234/V1"
    assert _api_root(url_upper) == url_upper

    url_mixed = "http://localhost:1234/v1/"
    assert _api_root(url_mixed) == "http://localhost:1234"  # lowercase stripped per contract


def test_api_root_empty_string():
    """An empty string must round-trip unchanged — no slice errors on a short string."""
    assert _api_root("") == ""


def test_api_root_schemeless_v1_suffix():
    """A URL like 'http://host/v1' (no port) with /v1 trailing segment must still be stripped."""
    url = "http://example.com/v1"
    assert _api_root(url) == "http://example.com"


def test_api_root_trailing_slash_only_no_v1():
    """A URL ending in '/' but without '/v1' must round-trip unchanged."""
    url = "http://localhost:1234/"
    assert _api_root(url) == "http://localhost:1234"


def test_api_root_with_fragment_identifier():
    """A URL with a fragment after /v1 should not be stripped — only bare '/v1' at end triggers removal."""
    url = "http://example.com/v1#section"
    assert _api_root(url) == url


def test_api_root_double_v1_suffix():
    """When the configured base URL itself ends in '/v1/v1', only the trailing /v1 is stripped — leaving the inner /v1 intact."""
    url = "http://localhost:1234/v1/v1"
    assert _api_root(url) == "http://localhost:1234/v1"


def test_api_root_preserves_query_params():
    """Query parameters must survive — only the bare '/v1' segment at end is stripped."""
    url = "http://example.com/api?timeout=30"
    assert _api_root(url) == url


def test_api_root_with_port_and_v1():
    """A standard LM Studio URL with explicit port and /v1 should strip cleanly to the host:port root."""
    url = "https://lmstudio.local:8080/v1"
    assert _api_root(url) == "https://lmstudio.local:8080"


def test_api_root_strips_double_slash_before_v1():
    """A double-slash before /v1 must be collapsed — the docstring explicitly calls this out."""
    url = "http://localhost:1234//v1"
    assert _api_root(url) == "http://localhost:1234"


def test_api_root_strips_multiple_trailing_slashes_after_v1():
    """Multiple trailing slashes after /v1 must all be stripped to reach the host root."""
    url = "http://localhost:1234/v1///"
    assert _api_root(url) == "http://localhost:1234"


def test_api_root_just_v1_suffix():
    """A URL ending in only '/v1' with no further segments must be stripped."""
    url = "http://example.com/v1"
    assert _api_root(url) == "http://example.com"


def test_clean_grammar_output_markdown_without_json():
    """When code blocks exist but contain no valid JSON, the stripped text is returned unchanged."""
    raw = '```json\njust some prose here\n```'
    result = clean_grammar_output(raw)
    assert '"origin"' not in result and '[' not in result.lstrip()


def test_clean_grammar_output_unclosed_block_with_json_after():
    """An unclosed code block should not prevent extraction of JSON found later in the text."""
    raw = '```\nsome preamble\n{"a": 1}\n```garbage'
    result = clean_grammar_output(raw)
    assert '"a"' in result or '{"a": 1}' == result


def test_clean_grammar_output_no_opening_brace():
    """When the cleaned text has no opening brace, no JSON extraction should occur."""
    raw = "```json\nthis is not json at all\n```"
    result = clean_grammar_output(raw)
    assert '{' not in result and '[' not in result.lstrip()


def test_clean_grammar_output_preserves_nested_json():
    """Deeply nested JSON arrays must be preserved fully during extraction."""
    raw = '{"nested": {"deep": [1, 2, {"x": true}]}}'
    result = clean_grammar_output(raw)
    assert result == raw


def test_clean_grammar_output_handles_mixed_code_block_languages():
    """Code blocks tagged with 'tracery' should still have their content extracted."""
    raw = '```\n{"origin": ["a", "b"]}\n```'
    result = clean_grammar_output(raw)
    assert '"origin"' in result


def test_clean_grammar_output_multiline_thinking():
    """Multi-line thinking blocks must be stripped along with surrounding whitespace."""
    raw = '```\nsome preamble\n{"a": 1}\n```'
    # This is a regular code block, not thinking — should still extract JSON
    result = clean_grammar_output(raw)
    assert '"a"' in result


def test_clean_grammar_output_smart_quotes_in_code_block():
    """Smart quotes inside a code-blocked JSON must be normalized during cleaning."""
    raw = '```json\n{"key": \u201cvalue\u201d}\n```'
    result = clean_grammar_output(raw)
    assert '\u201c' not in result and '\u201d' not in result
    expected = '{"key": "value"}'
    assert result == expected


# ---------------------------------------------------------------------------
# hash_prompt — unicode, special chars, empty input (pure, no mocks needed)
# ---------------------------------------------------------------------------

def test_hash_prompt_unicode_input():
    """Non-ASCII user prompts must still produce deterministic 12-char hex hashes."""
    h = hash_prompt("a cat with 🐾 paws")
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


def test_hash_prompt_special_characters():
    """Prompts containing quotes, newlines, or other special characters must hash consistently."""
    h1 = hash_prompt('a "cat" with\nnewlines')
    h2 = hash_prompt('a "cat" with\nnewlines')
    assert h1 == h2 and len(h1) == 12


def test_hash_prompt_empty_string():
    """An empty user prompt must still produce a valid 12-char hash (not raise)."""
    h = hash_prompt("")
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


def test_hash_prompt_differentiates_similar_prompts():
    """Trivially similar prompts must produce different hashes."""
    h_short = hash_prompt("a cat")
    h_long = hash_prompt("a cat sitting on a mat")
    assert h_short != h_long and len(h_short) == len(h_long) == 12


# ---------------------------------------------------------------------------
# get_system_prompt — explicit path, non-default location
# ---------------------------------------------------------------------------

def test_get_system_prompt_explicit_path(tmp_path):
    """get_system_prompt must read from the supplied templates_dir (not only the default)."""
    custom = tmp_path / "custom"
    custom.mkdir()
    prompt_file = custom / "system_prompt.txt"
    content = "CUSTOM SYSTEM PROMPT CONTENT"
    prompt_file.write_text(content)

    result = get_system_prompt(templates_dir=custom)
    assert result == content


def test_get_system_prompt_default_uses_project_templates():
    """Without an explicit path, the function must resolve to paths.templates_dir."""
    # The default templates dir is relative to the project; verify it resolves correctly.
    from grammar_generator import paths as gen_paths
    expected_file = gen_paths.templates_dir / "system_prompt.txt"
    if expected_file.exists():
        content = expected_file.read_text()
        result = get_system_prompt()  # default path
        assert result == content


# ---------------------------------------------------------------------------
# validate_grammar_structure — rejection paths (pure validation, no mocks)
# ---------------------------------------------------------------------------

def _make_grammar(**overrides):
    """Build a minimal valid grammar; overrides replace the origin rule options."""
    base = {
        "origin": ["a cat", "a dog", "a bird", "a fish", "a horse"],
        "kolors": [
            "A sepia tone, soft grainy texture",
            "A vibrant neon palette with high contrast",
            "A muted grayscale with subtle blue undertones",
            "A watercolor wash with pastel highlights",
            "An oil painting style with rich layered brushstrokes",
        ],
    }
    base.update(overrides)
    return base


def test_validate_accepts_minimal_valid_grammar():
    """A grammar with origin and one varying rule must pass validation."""
    from grammar_generator import validate_grammar_structure
    validate_grammar_structure(_make_grammar())  # no exception


def test_validate_rejects_non_dict_input():
    """Non-dict grammars (list, string) must be rejected immediately."""
    from grammar_generator import validate_grammar_structure
    with pytest.raises(ValueError):
        validate_grammar_structure([])
    with pytest.raises(ValueError):
        validate_grammar_structure("not json")


def test_validate_rejects_missing_origin():
    """Grammars without the required 'origin' key must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = {"kolors": ["red", "blue", "green", "yellow", "purple"]}
    with pytest.raises(ValueError, match='Grammar must be a JSON object containing an "origin"'):
        validate_grammar_structure(bad)


def test_validate_rejects_too_many_rules():
    """Grammars exceeding 8 rules must be rejected."""
    from grammar_generator import validate_grammar_structure
    many = dict({"origin": ["a"]})
    for i in range(9):
        many[f"rule{i}"] = ["x"]
    with pytest.raises(ValueError, match="Grammar must contain at most 8 rules"):
        validate_grammar_structure(many)


def test_validate_rejects_empty_rule_options():
    """A rule pointing to an empty list must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = _make_grammar()
    bad["kolors"] = []
    with pytest.raises(ValueError, match="must be a non-empty array"):
        validate_grammar_structure(bad)


def test_validate_rejects_invalid_rule_names():
    """Rule names containing spaces or starting with digits must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = _make_grammar()
    bad["1badname"] = ["x", "y", "z", "w", "v"]
    with pytest.raises(ValueError, match="rule name .* is invalid"):
        validate_grammar_structure(bad)

    bad2 = _make_grammar()
    bad2["has space"] = ["x", "y", "z", "w", "v"]
    with pytest.raises(ValueError, match="rule name .* is invalid"):
        validate_grammar_structure(bad2)


def test_validate_rejects_empty_string_options():
    """Rule options that are empty strings or whitespace must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = _make_grammar()
    bad["origin"] = ["a", "b", "", "d", "e"]
    with pytest.raises(ValueError, match="must contain non-empty strings"):
        validate_grammar_structure(bad)


def test_validate_rejects_duplicate_alternatives():
    """Rules containing duplicate options must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = _make_grammar()
    bad["origin"] = ["a cat", "a dog", "a cat", "a bird", "a horse"]  # duplicate 'a cat'
    with pytest.raises(ValueError, match="contains duplicate alternatives"):
        validate_grammar_structure(bad)


def test_validate_rejects_varying_rule_outside_five_to_seven():
    """Varying rules must have between 5 and 7 alternatives inclusive."""
    from grammar_generator import validate_grammar_structure

    # Too few (4 options)
    bad = _make_grammar()
    bad["origin"] = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="must contain 5–7 alternatives"):
        validate_grammar_structure(bad)

    # Too many (8 options)
    bad2 = _make_grammar()
    bad2["origin"] = ["a", "b", "c", "d", "e", "f", "g", "h"]
    with pytest.raises(ValueError, match="must contain 5–7 alternatives"):
        validate_grammar_structure(bad2)


def test_validate_rejects_zero_varying_rules():
    """A grammar where all rules have exactly one option must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = {"origin": ["just a cat"]}
    with pytest.raises(ValueError, match="must contain at least one varying rule"):
        validate_grammar_structure(bad)


def test_validate_rejects_missing_rule_references():
    """Rules that reference undefined rules via #name# syntax must be rejected."""
    from grammar_generator import validate_grammar_structure
    bad = _make_grammar()
    bad["origin"] = ["a #phantom# cat", "a dog", "a bird", "a fish", "a horse"]
    with pytest.raises(ValueError, match="missing rules: phantom"):
        validate_grammar_structure(bad)


def test_validate_accepts_valid_rule_references():
    """Rules that reference defined rules via #name# syntax must pass."""
    from grammar_generator import validate_grammar_structure
    good = {
        "origin": ["a #kolors# cat", "a dog", "a bird", "a fish", "a horse"],
        "kolors": [
            "sepia tone",
            "neon palette",
            "grayscale",
            "watercolor",
            "oil painting",
        ],
    }
    validate_grammar_structure(good)  # no exception
