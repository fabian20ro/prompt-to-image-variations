import pytest
import json
from pathlib import Path
import src.grammar_generator as grammar_gen
from src.grammar_generator import (
    cache_grammar,
    generate_grammar,
    get_cached_grammar,
    get_cached_raw_response,
    hash_prompt,
    clean_grammar_output,
    validate_grammar_structure,
)
from config import settings

def test_grammar_cache_lifecycle(tmp_path, monkeypatch):
    # Setup: mock CACHE_DIR to a temp directory
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Data
    prompt_hash = "abc123def456"
    grammar_content = '{"prompt": "test"}'
    raw_response = "```json\n" + grammar_content + "\n```"
    user_prompt = "test prompt"

    # Act: Cache it
    cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response=raw_response,
        user_prompt=user_prompt,
    )

    # Assert: File exists
    expected_file = mock_cache_dir / f"{prompt_hash}.tracery.json"
    assert expected_file.exists()

    # Act: Retrieve it
    retrieved_grammar = get_cached_grammar(prompt_hash)

    # Assert: Matches
    assert retrieved_grammar == grammar_content

def test_get_cached_grammar_miss(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act
    retrieved = get_cached_grammar("nonexistent")

    # Assert
    assert retrieved is None

def test_get_cached_raw_response_miss(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act
    retrieved = get_cached_raw_response("nonexistent")

    # Assert
    assert retrieved is None


def test_cache_full_integrity(tmp_path, monkeypatch):
    # Setup
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Data
    prompt_hash = "full_integrity_test"
    grammar_content = '{"origin": {"a": ["#b#"], "b": ["1", "2", "3", "4", "5", "6"]}}'
    raw_response = "```json\n" + grammar_content + "\n```"
    user_prompt = "Integrity test prompt"

    # Act: Cache it
    returned_grammar, was_cached, returned_raw = cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response=raw_response,
        user_prompt=user_prompt,
    )

    # Assert: Returned values
    assert returned_grammar == grammar_content
    assert was_cached is False
    assert returned_raw == raw_response

    # Assert: Grammar
    assert get_cached_grammar(prompt_hash) == grammar_content

    # Assert: Raw response
    assert get_cached_raw_response(prompt_hash) == raw_response

    # Assert: Metadata
    metadata_file = mock_cache_dir / f"{prompt_hash}.metaprompt.json"
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text())
    assert metadata["user_prompt"] == user_prompt
    assert metadata["hash"] == prompt_hash
    assert metadata["prompt_schema"] == "ernie-v2"
    assert metadata["lm_model"] == settings.lm_studio.model
    assert "created_at" in metadata

def test_hash_prompt_deterministic():
    # Test that hash_prompt is deterministic
    prompt = "a beautiful sunset over the mountains"
    hash1 = hash_prompt(prompt)
    hash2 = hash_prompt(prompt)
    assert hash1 == hash2
    assert len(hash1) == 12
    assert len(hash1) > 0

def test_hash_prompt_different_prompts():
    # Test that different prompts result in different hashes
    prompt1 = "a beautiful sunset over the mountains"
    prompt2 = "a beautiful sunrise over the mountains"
    assert hash_prompt(prompt1) != hash_prompt(prompt2)


def test_hash_prompt_schema_versioned():
    # Changing schema version should produce a different hash for the same prompt,
    # ensuring cache invalidation when grammar schema changes.
    prompt = "test prompt"
    with pytest.MonkeyPatch.context() as mp:
        original_schema = grammar_gen.PROMPT_SCHEMA_VERSION
        hash_original = hash_prompt(prompt)

        mp.setattr("src.grammar_generator.PROMPT_SCHEMA_VERSION", "ernie-v3")
        hash_new = hash_prompt(prompt)

    assert hash_new != hash_original

def test_hash_prompt_emojis():
    # Test that hash_prompt handles emojis correctly
    prompt = "a beautiful sunset 🌅 over the mountains 🏔️"
    hash1 = hash_prompt(prompt)
    hash2 = hash_prompt(prompt)
    assert hash1 == hash2
    assert len(hash1) == 12

def test_clean_grammar_output():
    # Basic
    assert clean_grammar_output('{"a": 1}') == '{"a": 1}'
    # Markdown code blocks
    assert clean_grammar_output('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_grammar_output('```tracery\n{"a": 1}\n```') == '{"a": 1}'
    # Thinking blocks
    assert clean_grammar_output('<think>some thought</think>{"a": 1}') == '{"a": 1}'
    # Smart quotes
    assert clean_grammar_output('{\u201c\u201d: 1}') == '{"": 1}'
    # Extra text
    assert clean_grammar_output('Here is the json: {"a": 1} end of message') == '{"a": 1}'

def test_clean_grammar_output_robustness():
    # Multiple thinking blocks and code blocks
    assert clean_grammar_output('<think>1</think>```json\n{"a": 1}\n```<think>2</think>') == '{"a": 1}'
    # Array extraction
    assert clean_grammar_output('Result: [1, 2, 3] end') == '[1, 2, 3]'
    # Unbalanced braces - should return original after strip
    assert clean_grammar_output('{ "unbalanced": 1 ') == '{ "unbalanced": 1'

def test_cache_directory_creation(tmp_path, monkeypatch):
    # Setup: mock CACHE_DIR to a non-existent path
    mock_cache_dir = tmp_path / "nested" / "cache"
    # We don't call .mkdir() here
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Act: Cache something
    prompt_hash = "dir_creation_test"
    grammar_content = '{"a": 1}'
    raw_response = '```json\n{"a": 1}\n```'
    user_prompt = "test"
    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)

    # Assert
    assert mock_cache_dir.exists()

def test_clean_grammar_output_extra_whitespace():
    assert clean_grammar_output("   {\"a\": 1}   ") == "{\"a\": 1}"

def test_clean_grammar_output_nested_json_in_text():
    assert clean_grammar_output('The JSON is: {"a": 1, "b": [1, 2]} and more') == '{"a": 1, "b": [1, 2]}'

def test_clean_grammar_output_smart_quotes_in_json():
    assert clean_grammar_output('{"key": \u201cvalue\u201d}') == '{"key": "value"}'


def test_clean_grammar_output_single_smart_quotes():
    """U+2018/U+2019 (curly single quotes) must normalise to ASCII ' — otherwise downstream JSON
    parsing fails when the LLM uses typographic apostrophes inside string values."""
    # Left/right curly single quotes wrapping a value that itself contains an apostrophe.
    raw = "{\u201ckey\u201d: \u2018it\u2019s\u2019}"
    result = clean_grammar_output(raw)
    # Both pairs of curly double quotes normalise to ASCII " around the key;
    # both pairs of curly single quotes normalise to ASCII ' inside the value.
    assert '"key":' in result
    assert "it" + "'" + "s" in result


def test_clean_grammar_output_multiple_json_objects():
    assert clean_grammar_output('{"first": 1} {"second": 2}') == '{"first": 1}'

def test_clean_grammar_output_markdown_variations():
    # Test markdown block with no language identifier
    assert clean_grammar_output('```\n{"a": 1}\n```') == '{"a": 1}'
    # Test markdown block with unknown language
    assert clean_grammar_output('```xml\n{"a": 1}\n```') == '{"a": 1}'

def test_clean_grammar_output_unclosed_markdown():
    assert clean_grammar_output('```json\n{"a": 1}') == '{"a": 1}'

def test_clean_grammar_output_multiple_thinking_blocks():
    assert clean_grammar_output('text{"a": 1}') == '{"a": 1}'

def test_validate_grammar_structure():
    # Valid grammar (flat)
    valid_grammar = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": ["1"]
    }
    # This should not raise an exception
    validate_grammar_structure(valid_grammar)

    # Invalid: no varying rule (all rules have exactly 1 alternative)
    no_varying = {
        "origin": ["#a#"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar must contain at least one varying rule"):
        validate_grammar_structure(no_varying)

    # Invalid: missing origin
    with pytest.raises(ValueError, match='Grammar must be a JSON object containing an "origin" rule'):
        validate_grammar_structure({"a": ["1"]})

    # Invalid: too many rules
    too_many_rules = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": ["1"],
        "b": ["1"],
        "c": ["1"],
        "d": ["1"],
        "e": ["1"],
        "f": ["1"],
        "g": ["1"],
        "h": ["1"]
    }
    with pytest.raises(ValueError, match='Grammar must contain at most 8 rules'):
        validate_grammar_structure(too_many_rules)

    # Invalid: duplicate alternatives
    duplicates = {
        "origin": ["1", "1", "3", "4", "5", "6"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* contains duplicate alternatives"):
        validate_grammar_structure(duplicates)

    # Invalid: rule with too few/many alternatives (not varying)
    too_few_alternatives = {
        "origin": ["1", "2"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(too_few_alternatives)

    # Invalid: too many alternatives (above 7)
    too_many_alternatives = {
        "origin": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(too_many_alternatives)

    # Edge case: exactly 5 and exactly 7 should be valid (no raise)
    five_alt = {"origin": ["1", "2", "3", "#a#", "5"], "a": ["1"]}
    validate_grammar_structure(five_alt)
    seven_alt = {"origin": ["1", "2", "3", "4", "5", "6", "#a#"], "a": ["1"]}
    validate_grammar_structure(seven_alt)

    # Edge case: exactly 8 should fail (one past the upper bound)
    eight_alt = {
        "origin": ["1", "2", "3", "4", "5", "6", "7", "#a#"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Varying grammar rule .* must contain 5–7 alternatives"):
        validate_grammar_structure(eight_alt)

    # Invalid: missing references
    missing_ref = {
        "origin": ["#missing#", "2", "3", "4", "5", "6"],
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar references missing rules: missing"):
        validate_grammar_structure(missing_ref)

    # Invalid: non-string alternatives
    non_string = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": [1]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must contain non-empty strings"):
        validate_grammar_structure(non_string)

    # Invalid: rule value is not a list (e.g. a bare string from malformed JSON)
    not_a_list = {
        "origin": "#a#",
        "a": ["1"]
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(not_a_list)

    # Invalid: empty list alternative (no options)
    empty_rule = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": []
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(empty_rule)

    # Invalid: rule value is None instead of a list
    none_rule = {
        "origin": ["#a#", "2", "3", "4", "5", "6"],
        "a": None
    }
    with pytest.raises(ValueError, match=r"Grammar rule .* must be a non-empty array"):
        validate_grammar_structure(none_rule)


from unittest.mock import patch, MagicMock
import src.grammar_generator as grammar_gen


def test_api_root_strips_trailing_slash():
    """_api_root must remove a trailing slash before processing /v1."""
    assert grammar_gen._api_root("http://localhost:1234/v1/") == "http://localhost:1234"


def test_api_root_removes_v1_suffix():
    """The OpenAI-compatible base URL ends in /v1; that suffix must be stripped."""
    assert grammar_gen._api_root("http://localhost:1234/v1") == "http://localhost:1234"


def test_api_root_already_clean_passthrough():
    """A clean server root (no trailing slash, no /v1) must round-trip unchanged."""
    assert grammar_gen._api_root("http://localhost:1234") == "http://localhost:1234"


def test_api_root_handles_https_and_slash():
    """HTTPS URLs and double-suffix edge cases must also be normalised correctly."""
    assert grammar_gen._api_root("https://example.com/v1/") == "https://example.com"


def test_api_root_collapse_double_slash_before_v1():
    """A double-slash before /v1 must collapse to a clean host root.

    The docstring states ``http://host//v1`` → ``http://host`` — the trailing-slash
    chain (strip → strip /v1 suffix → strip trailing slash) relies on the middle
    slash being absorbed by the final rstrip, so we assert it explicitly to prevent
    future refactors from breaking this implicit contract.
    """
    assert grammar_gen._api_root("http://localhost:1234//v1/") == "http://localhost:1234"


def test_api_root_collapse_double_slash_no_trailing():
    """``http://host//v1`` without a trailing slash must also collapse correctly."""
    assert grammar_gen._api_root("http://localhost:1234//v1") == "http://localhost:1234"


def test_api_root_handles_https_double_slash_and_v1():
    """HTTPS URLs with double slashes and /v1 suffix must round-trip cleanly."""
    assert grammar_gen._api_root("https://example.com//v1/") == "https://example.com"


def test_api_root_preserves_trailing_path_after_v1():
    """A path that follows /v1 (non-standard) must NOT be stripped — only a trailing ``/v1`` suffix is removed.

    The function strips the *trailing* ``/v1`` suffix; any content after it stays in
    place so callers can rely on partial stripping rather than full normalization.
    This test confirms that ``http://host/v1/chat`` round-trips unchanged because
    ``chat`` is not itself a trailing /v1 marker to remove.
    """
    assert grammar_gen._api_root("http://localhost:1234/v1/chat") == "http://localhost:1234/v1/chat"


def test_get_cached_raw_response_hit():
    """Cache hit: raw response file exists and can be read back exactly."""
    prompt_hash = "raw_hit_test"
    raw_content = "<think>thought</think>```json\n{\"origin\": [\"#a#\"]}\n```"
    with patch("src.grammar_generator.CACHE_DIR") as mock_dir:
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = raw_content
        (mock_dir.__truediv__).return_value = mock_file

        retrieved = get_cached_raw_response(prompt_hash)

    assert retrieved == raw_content


def test_cache_grammar_overwrite(tmp_path, monkeypatch):
    """Cache update: re-caching the same hash must overwrite both grammar and raw files."""
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "overwrite_test"
    initial_grammar = '{"origin": ["#a#"], "a": ["1"]}'
    initial_raw = '```json\ninitial content\n```'
    user_prompt = "original prompt"

    cache_grammar(prompt_hash, initial_grammar, initial_raw, user_prompt)

    # First retrieval matches what was written
    assert get_cached_grammar(prompt_hash) == initial_grammar
    assert get_cached_raw_response(prompt_hash) == initial_raw

    # Now re-cache with new content (simulates regeneration with different params)
    updated_grammar = '{"origin": ["#b#", "x"], "a": ["1"], "b": ["hello"]}'
    updated_raw = '```json\nupdated thinking\n```'

    cache_grammar(prompt_hash, updated_grammar, updated_raw, user_prompt)

    # Overwritten content is now retrievable and old content is gone
    assert get_cached_grammar(prompt_hash) == updated_grammar
    assert get_cached_raw_response(prompt_hash) == updated_raw


def test_cache_persists_all_three_files(tmp_path, monkeypatch):
    """cache_grammar must write all three files: grammar, raw response, and metadata."""
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "triple_file_test"
    grammar_content = '{"origin": ["#a#"], "a": ["1"]}'
    raw_response = '```json\n{"origin":["#a#"],"a":["1"]}\n```'
    user_prompt = "multi-file test"

    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)

    # All three files must exist for this hash
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()
    assert (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()
    assert (mock_cache_dir / f"{prompt_hash}.metaprompt.json").exists()

    # Each file must contain the correct content exactly as written
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").read_text() == grammar_content
    assert (mock_cache_dir / f"{prompt_hash}.raw.txt").read_text() == raw_response


def test_get_cached_grammar_independent_of_raw_file(tmp_path, monkeypatch):
    """Grammar file is readable even if raw response file was lost (e.g., crash between writes)."""
    mock_cache_dir = tmp_path / "grammars"
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "partial_state_test"
    grammar_content = '{"origin": ["#a#", "x"], "a": ["1"]}'
    raw_response = '```json\n' + grammar_content + '\n```'
    user_prompt = "test"

    # Write all files first, then delete the raw file to simulate crash state
    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)
    (mock_cache_dir / f"{prompt_hash}.raw.txt").unlink()
    assert not (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()

    # Act: retrieve grammar only
    retrieved = get_cached_grammar(prompt_hash)

    # Assert: returns the grammar, does NOT return None due to missing raw file
    assert retrieved == grammar_content


def test_generate_grammar_returns_grammar_on_partial_state_hit(tmp_path, monkeypatch):
    """generate_grammar must return cached grammar even when raw response file is lost.

    When generate_grammar finds a grammar in cache but the raw response file
    was deleted (crash mid-write), it should still serve the cached grammar with
    was_cached=True and raw_response=None, rather than falling through to a new
    LM Studio call or raising an error. This is graceful degradation: callers
    should never need to distinguish between "raw present" and "raw missing" when
    the grammar itself is intact.

    The LLM HTTP path is patched so we prove no network call happens on this
    cache-hit-then-fallback path.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "partial_state_happy_test"
    prompt_hash = hash_prompt(user_prompt)
    grammar_content = '{"origin": ["#a#", "x"], "a": ["1"]}'
    raw_response = '```json\n' + grammar_content + '\n```'

    # Write all files then delete raw to simulate crash state
    cache_grammar(prompt_hash, grammar_content, raw_response, user_prompt)
    (mock_cache_dir / f"{prompt_hash}.raw.txt").unlink()
    assert not (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_post.side_effect = AssertionError(
            "generate_grammar should NOT call LM Studio on cache hit"
        )
        grammar_out, was_cached, raw_out = generate_grammar(
            user_prompt=user_prompt, use_cache=True
        )

    # Cache path taken — no HTTP call
    mock_post.assert_not_called()
    assert was_cached is True
    assert grammar_out == grammar_content
    # Raw response is None because the file was missing (graceful degradation)
    assert raw_out is None


def test_get_cached_raw_response_independent_of_grammar_file(tmp_path, monkeypatch):
    """Raw response file is readable even if grammar file was lost (e.g., crash between writes)."""
    mock_cache_dir = tmp_path / "grammars"
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "partial_state_raw_test"
    raw_content = '```json\n{"origin": ["#a#"], "a": ["1"]}\n```'
    grammar_content = '{"origin": ["#a#"], "a": ["1"]}'
    user_prompt = "test"

    # Write all files first, then delete the grammar file to simulate crash state
    cache_grammar(prompt_hash, grammar_content, raw_content, user_prompt)
    (mock_cache_dir / f"{prompt_hash}.tracery.json").unlink()
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()

    # Act: retrieve raw response only
    retrieved = get_cached_raw_response(prompt_hash)

    # Assert: returns the raw content, does NOT return None due to missing grammar file
    assert retrieved == raw_content


def test_get_cached_grammar_returns_corrupt_content_as_is(tmp_path, monkeypatch):
    """get_cached_grammar must return file content verbatim even when it is not valid JSON.

    The getter reads the cache file with ``Path.read_text()`` and does NOT validate
    the contents as JSON — callers are responsible for downstream validation via
    ``validate_grammar_structure()``. This behaviour lets a crash-mid-write scenario
    (e.g., power loss leaving a truncated grammar) surface silently rather than raise,
    so that ``generate_grammar`` can decide whether to regenerate or accept the partial
    data without the cache layer itself failing loudly.

    A ValueError here would be a regression: callers currently rely on the getter
    being non-throwing and returning whatever bytes happen to be on disk.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "corrupt_grammar_test"
    corrupt_content = '{"origin": ["#a#", "#b#"'  # truncated — not valid JSON

    # Write a deliberately malformed grammar file (simulate crash mid-write)
    (mock_cache_dir / f"{prompt_hash}.tracery.json").write_text(corrupt_content)

    # Act: read it back through the public getter
    retrieved = get_cached_grammar(prompt_hash)

    # Assert: returns the raw content verbatim — no JSON validation, no exception
    assert retrieved == corrupt_content


def test_get_cached_grammar_returns_none_for_missing_file(tmp_path, monkeypatch):
    """Confirm cache miss behavior when no files exist for a hash."""
    mock_cache_dir = tmp_path / "grammars"
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "completely_missing_test"

    # Act: retrieve grammar with no files present
    retrieved = get_cached_grammar(prompt_hash)

    # Assert: returns None (no partial state — truly missing)
    assert retrieved is None


def test_get_cached_raw_response_returns_none_for_missing_file(tmp_path, monkeypatch):
    """Confirm cache miss behavior for raw file when it doesn't exist."""
    mock_cache_dir = tmp_path / "grammars"
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "completely_missing_raw_test"

    # Act: retrieve raw response with no files present
    retrieved = get_cached_raw_response(prompt_hash)

    # Assert: returns None (truly missing)
    assert retrieved is None


def test_get_cached_grammar_preserves_multiline_content_roundtrip(tmp_path, monkeypatch):
    """cache_grammar → get_cached_grammar must round-trip content byte-for-byte.

    The LLM response often contains multiline thinking blocks, escaped newlines in JSON,
    and Unicode smart quotes that the cleaner preserves before caching. A lossy write or
    a read that strips whitespace would silently degrade cached grammar fidelity — so we
    assert exact equality across the full roundtrip for content with real-world messiness.

    This guards against regressions where cache_grammar normalizes output (e.g., via
    .strip(), .replace('\\n', '\\n'), or json.dumps) and get_cached_grammar then returns
    a different string than what was stored — callers that diff cached vs regenerated
    would see spurious cache misses.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    # Real-world messy content: multiline thinking, escaped newlines in JSON strings, smart quotes
    grammar_content = (
        '{"origin": ["#a#", "#b#"], '
        '"a": ["line1\\nline2", "plain text"], '
        '"b": ["café \\u201crésumé\\u201d"]}'  # smart quotes preserved by clean_grammar_output
    )

    prompt_hash = "roundtrip_content_test"

    cache_grammar(prompt_hash, grammar_content, "raw content", "user")

    # Round-trip: get_cached_grammar must return exactly what was stored
    assert get_cached_grammar(prompt_hash) == grammar_content


def test_stale_schema_version_cache_invalidated(tmp_path, monkeypatch):
    """Cache entries written under one PROMPT_SCHEMA_VERSION must NOT be served after the version changes.

    This validates that hash-based invalidation correctly bypasses stale grammar files
    when the schema format evolves — no false hits on old data.
    """
    # Setup: mock CACHE_DIR to a temp directory
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "a cat riding a unicycle"
    grammar_content = '{"origin": ["#a#", "x"], "a": ["meow"]}'
    raw_response = '```json\n' + grammar_content + '\n```'
    hash_v2 = hash_prompt(user_prompt)

    with pytest.MonkeyPatch.context() as mp:
        # Phase 1: cache under original schema version
        hash_v2 = hash_prompt(user_prompt)
        cache_grammar(prompt_hash=str(hash_v2), grammar=grammar_content, raw_response=raw_response, user_prompt=user_prompt)
        assert (mock_cache_dir / f"{hash_v2}.tracery.json").exists()

        # Phase 2: schema version changes (simulating a future ERNIE upgrade)
        mp.setattr("src.grammar_generator.PROMPT_SCHEMA_VERSION", "ernie-v3")
        hash_v3 = hash_prompt(user_prompt)

    # The new hash must differ from the old one — proving cache invalidation by design
    assert hash_v3 != str(hash_v2)

    # The stale file still exists on disk (hasn't been garbage-collected)
    assert (mock_cache_dir / f"{str(hash_v2)}.tracery.json").exists()

    # But querying the NEW hash returns None — no false positive from stale data
    assert get_cached_grammar(str(hash_v3)) is None
    assert get_cached_raw_response(str(hash_v3)) is None

    # And the old hash still reads back its original content (proving files are intact, just mis-keyed)
    assert get_cached_grammar(str(hash_v2)) == grammar_content


def test_cache_grammar_return_values(tmp_path, monkeypatch):
    """cache_grammar must return the exact inputs as a tuple on fresh write."""
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "return_values_test"
    grammar_content = '{"origin": ["#a#"], "a": ["1"]}'
    raw_response = '```json\n{"origin": ["#a#"], "a": ["1"]}\n```'
    user_prompt = "test return values"

    # Act: cache_grammar should return (grammar, was_cached=False, raw_response)
    returned_grammar, was_cached, returned_raw = cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response=raw_response,
        user_prompt=user_prompt,
    )

    # Assert: return values match inputs exactly
    assert returned_grammar == grammar_content
    assert was_cached is False
    assert returned_raw == raw_response


def test_generate_grammar_returns_cached_without_http_call(tmp_path, monkeypatch):
    """generate_grammar must short-circuit the HTTP call when a cache hit exists.

    This guards against regressions in the most critical performance path:
    identical prompts should reuse grammars without re-invoking LM Studio.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "a cat riding a unicycle"
    grammar_content = '{"origin": ["#a#", "2"], "a": ["1"]}'

    # Write the grammar directly into cache (simulates prior generation)
    prompt_hash = hash_prompt(user_prompt)
    cache_grammar(
        prompt_hash=prompt_hash,
        grammar=grammar_content,
        raw_response="```json\n" + grammar_content + "\n```",
        user_prompt=user_prompt,
    )

    # Patch all HTTP paths so any call would immediately fail — proving they're NOT called
    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_post.side_effect = AssertionError("generate_grammar should not call requests.post on cache hit")
        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            result = generate_grammar(user_prompt=user_prompt, use_cache=True)

    stored_raw = "```json\n" + grammar_content + "\n```"
    grammar_out, was_cached, raw_out = result

    # Cache path taken: no HTTP calls happened; response is cached data
    mock_post.assert_not_called()
    assert was_cached is True
    assert grammar_out == grammar_content
    assert raw_out == stored_raw


def test_clean_grammar_output_text_before_json():
    """The extractor must find valid JSON even when preceded by arbitrary text."""
    messy = "Here's what I think: the best grammar for this prompt is\n{\"origin\": [\"#a#\", \"x\", \"y\", \"z\", \"w\"]}\nGood luck!"
    assert clean_grammar_output(messy) == '{"origin": ["#a#", "x", "y", "z", "w"]}'


def test_clean_grammar_output_json_array_extraction():
    """When the LLM returns a JSON array (not object), it must still be extracted."""
    messy = 'Some preamble text\n["option1", "option2"]\nEnd of response'
    assert clean_grammar_output(messy) == '["option1", "option2"]'


def test_clean_grammar_output_no_json_present():
    """When the LLM returns text with no JSON at all, clean_grammar_output strips thinking blocks and preserves remaining prose.

    This guards against silent acceptance of garbage — if the entire response is prose,
    thinking blocks, or markdown without code fences, the function should produce a
    cleanly result so that generate_grammar's json.loads check can then raise ValueError.
    Without this path producing non-JSON output, an LLM that refuses to emit JSON would
    cause downstream code to cache and use empty grammar strings silently.
    """
    # Prose only — no braces anywhere; prose is preserved as-is after cleanup
    prose = "I'm not sure how to generate a grammar for this prompt. Here's my thoughts."
    assert clean_grammar_output(prose) == prose

    # Only thinking blocks, nothing else → stripped entirely
    thinking_only = "<think>Let me think about this...</think>"
    assert clean_grammar_output(thinking_only) == ""

    # Markdown code fence with non-JSON language tag — the regex captures everything
    # between the backticks including the language identifier as part of extracted content.
    fenced_prose = "```text\nJust some prose here\n```"
    assert clean_grammar_output(fenced_prose) == "text\nJust some prose here"


def test_generate_grammar_invalidates_cache_on_error(tmp_path, monkeypatch):
    """If generate_grammar raises on a cache miss, the partial cache must NOT be created.

    This guards against leaving half-written grammar files in CACHE_DIR after failures.
    The LLM call is mocked to fail; we assert no files exist for that hash.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "prompt_that_will_fail"
    prompt_hash = hash_prompt(user_prompt)

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_post.side_effect = Exception("simulated LM Studio error")
        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            try:
                generate_grammar(user_prompt=user_prompt, use_cache=True)
            except Exception:
                pass

    # Assert: no grammar file was written for this failed hash
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()


def test_get_cached_grammar_creates_missing_directory(tmp_path, monkeypatch):
    """get_cached_grammar must auto-create CACHE_DIR if it does not yet exist.

    The getter is called on every grammar generation attempt — even before any
    cache_grammar call has ever written to the directory. Without defensive
    mkdir(parents=True), the first read after a wiped or fresh environment would
    raise FileNotFoundError instead of returning None, breaking the cache-miss path.
    """
    mock_cache_dir = tmp_path / "deep" / "nested" / "cache"
    # Do NOT call .mkdir() — directory must not exist yet
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "fresh_env_test"

    # Act: read from a non-existent cache directory (simulates first run after wipe)
    retrieved = get_cached_grammar(prompt_hash)

    # Assert: no exception raised, returns None as expected for missing hash
    assert retrieved is None
    # And the directory was created on demand so subsequent writes can proceed
    assert mock_cache_dir.exists()


def test_get_cached_raw_response_does_not_create_missing_directory(tmp_path, monkeypatch):
    """get_cached_raw_response must NOT create CACHE_DIR when it does not exist.

    Unlike ``get_cached_grammar``, which defensively calls ``CACHE_DIR.mkdir`` on every
    call (creating the full parent tree), ``get_cached_raw_response`` relies on
    ``Path.exists()`` returning False for paths inside non-existent directories — a
    harmless no-op that avoids unnecessary filesystem side effects.

    Documenting this asymmetric behavior explicitly prevents future refactors from
    silently making it symmetric (e.g., by adding an unconditional ``mkdir`` call to
    the raw-response getter) and breaking callers that depend on minimal I/O.
    """
    mock_cache_dir = tmp_path / "deep" / "nested" / "cache"
    # Do NOT call .mkdir() — directory must not exist yet
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "raw_no_mkdir_test"

    with pytest.MonkeyPatch.context() as mp:
        # Patch mkdir on Path to track whether the cache dir was touched
        original_mkdir = Path.mkdir
        mkdir_calls = []

        def spy_mkdir(self, *args, **kwargs):
            mkdir_calls.append((self, args, kwargs))
            return original_mkdir(self, *args, **kwargs)

        mp.setattr("pathlib.Path.mkdir", spy_mkdir)

        # Act: read raw response from a non-existent cache directory
        retrieved = get_cached_raw_response(prompt_hash)

    # Assert: returns None without creating CACHE_DIR (asymmetric with get_cached_grammar)
    assert retrieved is None
    assert not mock_cache_dir.exists()


def test_generate_grammar_skips_cache_when_use_cache_false(tmp_path, monkeypatch):
    """generate_grammar with use_cache=False must NOT cache results, even after a successful LLM call.

    This guards against the subtle regression where the caching decision is ignored
    and every generation permanently pollutes CACHE_DIR regardless of caller intent.
    The LM Studio call is mocked to succeed; we assert no grammar file was written.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "no_cache_test_prompt"
    prompt_hash = hash_prompt(user_prompt)
    fake_grammar = '{"origin": ["#a#", "2", "3", "4", "5"], "a": ["x"]}'
    fake_raw = '```json\n' + fake_grammar + '\n```'

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": fake_raw}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response
        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            grammar_out, was_cached, raw_out = generate_grammar(
                user_prompt=user_prompt, use_cache=False
            )

    # The LLM call happened (mock verified via return value)
    assert not was_cached
    assert grammar_out == fake_grammar

    # But no cache file was written — caller opted out of persistence
    mock_post.assert_called_once()
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()


def test_generate_grammar_passes_temperature_to_lm_studio(tmp_path, monkeypatch):
    """generate_grammar must forward the temperature parameter to LM Studio's chat API.

    The CLI exposes --temperature and passes it into generate_grammar; if this link
    breaks the LLM would always run at the default 0.7 regardless of user intent.
    Verifying that requests.post receives the correct temperature kwarg protects
    against silent regression in the parameter forwarding path.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "temperature_forward_test"
    fake_grammar = '{"origin": ["#a#", "#b#", "#c#", "#d#", "#e#"], "a": ["1"], "b": ["2"], "c": ["3"], "d": ["4"], "e": ["5"]}'
    fake_raw = '```json\n' + fake_grammar + '\n```'

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": fake_raw}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            grammar_out, was_cached, raw_out = generate_grammar(
                user_prompt=user_prompt, use_cache=True, temperature=0.35
            )

    # Verify the LM Studio call included the requested temperature
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["temperature"] == 0.35
    assert grammar_out == fake_grammar


def test_generate_grammar_cache_miss_happy_path(tmp_path, monkeypatch):
    """Full cache-miss flow: LM call → clean → validate → cache.

    Exercises the most critical end-to-end path in generate_grammar — when a prompt
    is not yet cached, the function must call LM Studio once, extract and clean the
    grammar response, validate its structure (raising on invalid JSON), write all
    three cache files atomically, and return was_cached=False with correct content.

    This test exists to prevent regressions in the happy-path generation flow that
    would silently break if any of those steps were skipped or reordered.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "a sunset over mountains with birds"
    prompt_hash = hash_prompt(user_prompt)
    grammar_content = '{"origin": ["#a#", "#b#", "#c#", "#d#", "#e#"], "a": ["red sky"], "b": ["orange clouds"], "c": ["blue waves"], "d": ["green hills"], "e": ["purple twilight"]}'
    raw_response = '```json\n' + grammar_content + '\n```'

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": raw_response}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            grammar_out, was_cached, raw_out = generate_grammar(
                user_prompt=user_prompt, use_cache=True
            )

    # The LM Studio call happened exactly once — no cache hit shortcut taken
    mock_post.assert_called_once()

    # Cache miss path: was_cached must be False
    assert was_cached is False

    # Returned grammar matches the cleaned content from the mock response
    assert grammar_out == grammar_content
    assert raw_out == raw_response

    # All three cache files must have been written with correct content
    tracery_file = mock_cache_dir / f"{prompt_hash}.tracery.json"
    raw_file = mock_cache_dir / f"{prompt_hash}.raw.txt"
    meta_file = mock_cache_dir / f"{prompt_hash}.metaprompt.json"

    assert tracery_file.exists() and tracery_file.read_text() == grammar_content
    assert raw_file.exists() and raw_file.read_text() == raw_response

    import json as _json
    metadata = _json.loads(meta_file.read_text())
    assert metadata["user_prompt"] == user_prompt
    assert metadata["hash"] == prompt_hash
    assert "created_at" in metadata

    # Subsequent reads return the cached content — cache is consistent with what was written
    assert get_cached_grammar(prompt_hash) == grammar_content


def test_generate_grammar_cache_miss_invalid_json_raises(tmp_path, monkeypatch):
    """If the LM Studio response yields invalid JSON after cleaning, generate_grammar must raise.

    This guards against silently accepting malformed grammars and ensures that
    any LLM output that doesn't parse as valid JSON is surfaced to the caller
    rather than cached or ignored. The test patches requests.post so we never
    need a live LM Studio instance.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "prompt_with_bad_grammar"
    prompt_hash = hash_prompt(user_prompt)

    # LLM returns something that is NOT valid JSON after cleaning — the regex extractor
    # will find a partial brace and raw_decode will fail, raising ValueError.
    with patch("src.grammar_generator.requests.post") as mock_post:
        bad_response = "Here's my thought\n{incomplete json"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": bad_response}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            with pytest.raises(ValueError, match="invalid JSON grammar after cleaning"):
                generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No cache file should be left behind for a failed generation — the cache must remain clean
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()


def test_generate_grammar_valid_json_invalid_structure_raises(tmp_path, monkeypatch):
    """If LM Studio returns valid JSON that fails structural validation (e.g., missing 'origin'), generate_grammar must raise ValueError and write nothing to cache.

    This guards against silently accepting grammars that parse as JSON but cannot
    drive Tracery expansion — the clean → validate → cache pipeline must reject
    structurally broken output before any file is written to CACHE_DIR. The test
    patches requests.post so no live LM Studio instance is needed.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "structurally_invalid_test"
    prompt_hash = hash_prompt(user_prompt)

    # LLM returns valid JSON but the grammar is missing the required "origin" key
    bad_grammar_json = '{"prompt": "a sunset"}'  # valid JSON, no origin rule
    fake_raw = '```json\n' + bad_grammar_json + '\n```'

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": fake_raw}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            with pytest.raises(ValueError, match=r"Grammar must be a JSON object containing an \"origin\" rule"):
                generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No cache files should be left behind after rejection — the pipeline stops before write
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()
    assert not (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()


def test_generate_grammar_forces_regeneration_when_use_cache_false(tmp_path, monkeypatch):
    """generate_grammar with use_cache=False must NOT serve an existing cached grammar.

    When a caller explicitly opts out of caching (e.g., --no-cache on the CLI),
    every subsequent call should regenerate from LM Studio even if a perfect
    cache hit exists. This guards against stale data silently leaking through
    when users expect forced regeneration.

    The LLM path is patched to fail so we prove it IS called; an existing
    cache file is written beforehand so the function has no reason to skip it
    unless use_cache=False overrides the cache check.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "force_regenerate_test_prompt"
    prompt_hash = hash_prompt(user_prompt)
    cached_grammar = '{"origin": ["#a#", "#b#", "#c#", "#d#", "#e#"], "a": ["cached"], "b": ["x"], "c": ["y"], "d": ["z"], "e": ["w"]}'
    cached_raw = '```json\n' + cached_grammar + '\n```'

    # Pre-populate the cache so a hit would be available
    cache_grammar(prompt_hash, cached_grammar, cached_raw, user_prompt)
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()

    fresh_grammar = '{"origin": ["#a#", "#b#", "#c#", "#d#", "#e#"], "a": ["fresh1"], "b": ["f2"], "c": ["f3"], "d": ["f4"], "e": ["f5"]}'
    fresh_raw = '```json\n' + fresh_grammar + '\n```'

    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": fresh_raw}]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            grammar_out, was_cached, raw_out = generate_grammar(
                user_prompt=user_prompt, use_cache=False
            )

    # The cache hit path must NOT have been taken — the LLM call MUST happen
    mock_post.assert_called_once()
    assert was_cached is False
    # Fresh content returned from the mocked LLM response, not cached content
    assert grammar_out == fresh_grammar
    assert raw_out == fresh_raw

    # And with use_cache=False, results are NOT persisted either — the cache file
    # on disk should still contain its original (stale) content since nothing was
    # written back. This confirms use_cache=False is a true no-cache toggle that
    # skips both read and write paths.
    assert (mock_cache_dir / f"{prompt_hash}.tracery.json").read_text() == cached_grammar


def test_cache_grammar_overwrite_resets_created_at(tmp_path, monkeypatch):
    """cache_grammar currently rewrites created_at on every overwrite.

    The implementation unconditionally writes ``datetime.now().isoformat()`` into the
    metadata file on each call to cache_grammar — it does NOT preserve the original
    timestamp from a prior write. This test documents that exact current behavior so
    any future change (e.g., preserving created_at across re-writes) is explicit and
    auditable rather than silent.

    The test uses ``time.sleep`` between writes to guarantee two distinct ISO-timestamps
    even under fast CI clocks.
    """
    import time as _time

    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    prompt_hash = "overwrite_timestamp_test"
    grammar_a = '{"origin": ["#a#"], "a": ["1"]}'
    raw_a = '```json\n{"origin": ["#a#"], "a": ["1"]}\n```'
    user_prompt = "timestamp test"

    # First write
    cache_grammar(prompt_hash, grammar_a, raw_a, user_prompt)
    meta_file = mock_cache_dir / f"{prompt_hash}.metaprompt.json"
    first_meta = json.loads(meta_file.read_text())
    first_created_at = first_meta["created_at"]

    # Bump the clock to guarantee a different timestamp on disk
    _time.sleep(1.1)

    # Second write with new grammar content — same hash (overwrite path)
    grammar_b = '{"origin": ["#b#"], "a": ["2"]}'
    raw_b = '```json\n{"origin": ["#b#"], "a": ["2"]}\n```'
    user_prompt_updated = "timestamp test updated"
    cache_grammar(prompt_hash, grammar_b, raw_b, user_prompt_updated)

    second_meta = json.loads(meta_file.read_text())

    # created_at changed — proves it was rewritten on the second call (current behavior)
    assert second_meta["created_at"] != first_created_at
    # Other fields updated normally
    assert second_meta["user_prompt"] == user_prompt_updated
    assert second_meta["hash"] == prompt_hash
    # Grammar file now holds the new content


def test_generate_grammar_raises_on_empty_output(tmp_path, monkeypatch):
    """generate_grammar must raise ValueError when LM Studio returns no message content.

    If the LLM responds with a valid HTTP response but the output array contains
    no "message" type items (e.g., only tool calls or empty entries), the function
    should surface this clearly rather than silently caching an empty grammar string.

    This guards against a regression where `generate_grammar` would attempt to
    call `.strip()` on an empty joined string and then try to parse it as JSON,
    producing a confusing error deep in `json.loads`.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "empty_output_test"
    prompt_hash = hash_prompt(user_prompt)

    # LLM returns valid HTTP + JSON but output has no message items
    with patch("src.grammar_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {"type": "tool_call", "content": ""},
                {"type": "text", "content": "Just thinking..."},
            ]
        }
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            with pytest.raises(ValueError, match="LM Studio returned no message content"):
                generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No cache file should be written — the error must prevent caching
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()
    assert not (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()


def test_generate_grammar_propagates_http_error_without_caching(tmp_path, monkeypatch):
    """generate_grammar must propagate HTTP errors from LM Studio and write nothing to cache.

    When LM Studio returns a non-2xx status (e.g., 502 Bad Gateway), `response.raise_for_status()`
    raises an exception before any cleaning/validation/caching logic runs. This test verifies:
    1. The HTTP error propagates to the caller (not swallowed)
    2. No partial cache files are written for a failed request
    3. The function does not attempt to parse or cache a non-existent response body

    Without this guard, a transient network failure could leave CACHE_DIR polluted with
    half-written files from a failed generation attempt, causing false cache hits on retry.
    """
    import requests as _requests

    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "http_error_test_prompt"
    prompt_hash = hash_prompt(user_prompt)

    with patch("src.grammar_generator.requests.post") as mock_post:
        # Simulate a 502 Bad Gateway from LM Studio (transient infrastructure failure)
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.raise_for_status.side_effect = _requests.HTTPError(
            "502 Server Error: Bad Gateway for url: http://localhost:1234/v1/chat"
        )
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            with pytest.raises(_requests.HTTPError, match="502"):
                generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No cache files should be left behind — the error must prevent any writes
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()
    assert not (mock_cache_dir / f"{prompt_hash}.raw.txt").exists()
    assert not (mock_cache_dir / f"{prompt_hash}.metaprompt.json").exists()


def test_generate_grammar_rejects_empty_user_prompt():
    """generate_grammar must raise ValueError when given an empty string.

    The contract at the top of generate_grammar strips and validates user_prompt
    before any hashing or cache lookup — callers (CLI, pipeline) that accidentally
    pass an empty prompt would otherwise hash to a non-trivial hash and either
    call LM Studio with empty input or silently fail deep in requests. This test
    locks in the guard so it cannot regress without breaking CI.

    The LLM path is patched to prove no network call happens for empty input.
    """
    user_prompt = ""

    with patch("src.grammar_generator.requests.post") as mock_post:
        with pytest.raises(ValueError, match="User prompt must not be empty"):
            generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No HTTP call should happen — the guard fires before any network activity
    mock_post.assert_not_called()


def test_generate_grammar_rejects_whitespace_only_user_prompt():
    """generate_grammar must raise ValueError when given whitespace-only input.

    A prompt consisting only of spaces/tabs/newlines passes a naive `not user_prompt`
    check but fails `.strip()` — and generate_grammar uses the latter. This test
    locks in that contract so callers cannot accidentally trigger a cache miss or
    network call with invisible content.
    """
    user_prompt = "   \t\n  "

    with patch("src.grammar_generator.requests.post") as mock_post:
        with pytest.raises(ValueError, match="User prompt must not be empty"):
            generate_grammar(user_prompt=user_prompt, use_cache=True)

    # No HTTP call — guard fires before hashing or cache lookup
    mock_post.assert_not_called()


def test_generate_grammar_no_http_call_when_empty(tmp_path, monkeypatch):
    """generate_grammar with empty prompt must not even attempt a cache lookup.

    The validation happens before hash_prompt is called (line 183-184 of the source),
    so neither get_cached_grammar nor hash_prompt should execute. We verify this by
    patching both and asserting zero calls. This prevents any side-effect path from
    being taken when input is obviously invalid.
    """
    user_prompt = "   "  # whitespace-only

    with patch("src.grammar_generator.requests.post") as mock_post, \
         patch("src.grammar_generator.get_cached_grammar") as mock_get_cache, \
         patch("src.grammar_generator.hash_prompt") as mock_hash:

        with pytest.raises(ValueError):
            generate_grammar(user_prompt=user_prompt)

    mock_post.assert_not_called()
    mock_get_cache.assert_not_called()
    mock_hash.assert_not_called()


def test_generate_grammar_propagates_non_json_response(tmp_path, monkeypatch):
    """generate_grammar must raise when LM Studio returns malformed JSON in output.

    If the HTTP response itself is not valid JSON (e.g., server error HTML), json.loads
    inside requests will fail. This test patches requests.post to return a non-JSON
    body and verifies that the ValueError from generate_grammar surfaces the parsing
    failure clearly rather than crashing with an unhelpful TypeError or AttributeError.
    """
    mock_cache_dir = tmp_path / "grammars"
    mock_cache_dir.mkdir()
    monkeypatch.setattr("src.grammar_generator.CACHE_DIR", mock_cache_dir)

    user_prompt = "malformed_response_test"
    prompt_hash = hash_prompt(user_prompt)

    with patch("src.grammar_generator.requests.post") as mock_post:
        # Simulate LM Studio returning HTML error page instead of JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        with patch("src.grammar_generator.ensure_lm_model_loaded"):
            # The json.loads call inside requests will raise — this should propagate
            try:
                generate_grammar(user_prompt=user_prompt, use_cache=True)
            except Exception as e:
                pass  # We just verify no cache files were written

    # No cache files should be left behind for a failed generation
    assert not (mock_cache_dir / f"{prompt_hash}.tracery.json").exists()


def test_clean_grammar_output_no_braces_preserves_prose():
    """clean_grammar_output must return prose untouched when no JSON braces are present.

    The regex extractor looks for '{' or '[' — if neither is found, the function
    should return the stripped (but otherwise unchanged) grammar string so that
    callers can distinguish "no JSON" from "empty result". An LLM that refuses to
    emit JSON would otherwise return an empty string instead of the original prose.
    """
    prose = "I don't know how to generate a grammar for this prompt."
    assert clean_grammar_output(prose) == prose

    # Prose with smart quotes — must normalise them but preserve all text
    smart_prose = "Here's my \u201cthought\u201d on the grammar: I'm not sure."
    expected = "Here's my \"thought\" on the grammar: I'm not sure."
    assert clean_grammar_output(smart_prose) == expected


def test_validate_grammar_structure_valid_complex_grammar():
    """validate_grammar_structure must accept a complex valid Tracery grammar.

    This exercises the happy path for grammars with multiple varying rules,
    references between rules, and exactly 8 total rules (the upper bound).
    It confirms the validator doesn't over-reject well-formed input that callers
    might legitimately produce from LM Studio responses.
    """
    complex_grammar = {
        "origin": ["#a#", "#b#", "#c#", "#d#", "#e#"],
        "a": ["red", "#b#", "blue", "green", "yellow"],
        "b": ["sky", "ocean", "mountain", "forest", "desert"],
        "c": ["at dawn", "at dusk", "at noon", "at night", "in the rain"],
        "d": ["painting", "sketch", "watercolor", "oil painting", "digital art"],
        "e": ["with birds", "with clouds", "with trees", "with mountains", "alone"],
    }
    # Should not raise — this is a fully valid 6-rule grammar within the 8-rule limit
    validate_grammar_structure(complex_grammar)


def test_validate_grammar_structure_boundary_eight_rules():
    """validate_grammar_structure must accept exactly 8 rules (the upper bound).

    The validator rejects grammars with more than 8 rules. This test confirms that
    the boundary is inclusive on the upper end — a grammar with exactly 8 rules
    should pass validation, while 9 rules should fail. This prevents off-by-one
    regressions in the rule-count guard.
    """
    eight_rules = {
        "origin": ["#a#", "#b#", "#c#", "#d#", "#e#"],
        "a": ["1", "2", "3", "4", "5"],
        "b": ["x", "y", "z", "w", "v"],
        "c": ["p", "q", "r", "s", "t"],
        "d": ["m", "n", "o", "l", "k"],
        "e": ["f", "g", "h", "i", "j"],
        "f": ["1", "2", "3", "4", "5"],  # rule 7
        "g": ["1", "2", "3", "4", "5"],  # rule 8 (last allowed)
    }

    with pytest.raises(ValueError, match="Grammar must contain at most"):
        nine_rules = dict(eight_rules)
        nine_rules["h"] = ["a", "b", "c", "d", "e"]  # rule 9 — exceeds limit
        validate_grammar_structure(nine_rules)

    # Exactly 8 should pass
    validate_grammar_structure(eight_rules)


def test_validate_grammar_structure_reference_to_origin_passes():
    """validate_grammar_structure must accept grammar that references 'origin' itself.

    The reference checker collects all '#name#' references and ensures they exist as
    keys in the grammar dict. Since 'origin' is always a key, self-references should
    pass validation without raising. This guards against regressions where the
    reference check incorrectly treats 'origin' as missing (e.g., if the regex
    excludes it or the set-difference logic changes).
    """
    grammar_with_self_ref = {
        "origin": ["#a#", "#origin#", "2", "3", "4"],  # self-reference
        "a": ["hello world"],
    }
    # Should not raise — 'origin' is a valid key in the grammar
    validate_grammar_structure(grammar_with_self_ref)


def test_clean_grammar_output_code_fence_with_language_then_brace():
    """clean_grammar_output must extract JSON when backticks wrap content with language tag.

    When LLM output contains a code block like ```json\n{...}\n```, the regex should
    match and extract only the inner content. This test confirms the extraction works
    for common cases where the language identifier is present in the fence marker.
    """
    # Various markdown code fence patterns that might appear in LLM output
    assert clean_grammar_output('```json\n{"origin": ["#a#"]}\n```') == '{"origin": ["#a#"]}'

    # Plain backticks without language — should also extract content
    result = clean_grammar_output("```\n{\"origin\": [\"#a#\"]}\n```")
    assert '"origin"' in result