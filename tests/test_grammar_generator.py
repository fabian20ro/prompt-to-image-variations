import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch, MagicMock
from grammar_generator import (
    _api_root,
    clean_grammar_output,
    get_system_prompt,
    hash_prompt,
    get_cached_grammar,
    cache_grammar,
    CACHE_DIR,
    generate_grammar,
    ensure_lm_model_loaded,
    validate_grammar_structure,
)


class TestCache(unittest.TestCase):
    """Tests for the caching mechanisms."""

    @patch("grammar_generator.CACHE_DIR")
    def test_get_cached_grammar_exists(self, mock_cache_dir):
        # Setup mock for CACHE_DIR / filename
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = '{"origin": "#test#"}'
        mock_cache_dir.__truediv__.return_value = mock_file

        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result == '{"origin": "#test#"}'

    @patch("grammar_generator.CACHE_DIR")
    def test_get_cached_grammar_not_exists(self, mock_cache_dir):
        # Setup mock for CACHE_DIR / filename
        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = False
        mock_cache_dir.__truediv__.return_value = mock_file

        prompt_hash = "abcdef123456"
        result = get_cached_grammar(prompt_hash)
        assert result is None

    @patch("grammar_generator.CACHE_DIR")
    @patch("grammar_generator.Path.write_text")
    @patch("grammar_generator.datetime")
    def test_cache_grammar(self, mock_datetime, mock_write, mock_cache_dir):
        prompt_hash = "abcdef123456"
        grammar = '{"origin": "#prompt#"}'
        raw_response = '<think>...</think>{"origin": "#prompt#"}'
        user_prompt = "a sunset"
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

        result = cache_grammar(prompt_hash, grammar, raw_response, user_prompt)

        assert isinstance(result, tuple)
        assert result[0] == grammar
        assert result[1] is False
        assert result[2] == raw_response


class TestGrammarTools(unittest.TestCase):
    """Tests for grammar utility functions."""

    def test_clean_grammar_output_empty(self):
        self.assertEqual(clean_grammar_output(""), "")
        self.assertEqual(clean_grammar_output("   "), "")

    def test_clean_grammar_output_with_thinking_and_code_blocks(self):
        input_str = '<think>some thought</think>```json\n{"key": "value"}\n```'
        expected = '{"key": "value"}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_with_smart_quotes(self):
        # Test with smart double quotes to ensure valid JSON output
        input_str = '{\u201ckey\u201d: \u201cvalue\u201d}'
        expected = '{"key": "value"}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_normalizes_smart_single_quotes(self):
        # The clean_grammar_output smart-quote normalizer must handle both
        # left/right single quotes so LLM output round-trips as valid Python dict
        # (straight singles are still invalid JSON — the caller must re-quote).
        input_str = "{\u2018key\u2019: \u2018value\u2019}"
        expected = "{'key': 'value'}"
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_extracts_json(self):
        input_str = 'Some noise here {"a": 1} and some noise there'
        expected = '{"a": 1}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_tracery_code_blocks(self):
        # When LM Studio tags the code block as ```tracery instead of ```json,
        # clean_grammar_output must still extract the JSON content.
        input_str = '<think>thinking</think>```tracery\n{"origin":["#subject#"]}\n```'
        expected = '{"origin":["#subject#"]}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_plain_code_block_without_language(self):
        # LM Studio may return a code block tagged with just ``` and no language,
        # so clean_grammar_output must still extract the JSON payload.
        input_str = '```\\n{"origin":["#subject#"]}\\n```'
        expected = '{"origin":["#subject#"]}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_thinking_blocks_followed_by_raw_json(self):
        # When LM Studio emits thinking blocks but no markdown wrapping, the
        # function must still isolate and return the JSON payload.
        input_str = '<think>some thought</think>{"origin":["#subject#"]}'
        expected = '{"origin":["#subject#"]}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_trailing_text_after_json(self):
        # After extracting the first valid JSON, trailing noise after closing ```
        # must not leak into the returned string.
        input_str = '```json\n{"key": "val"}\n```\nSome trailing noise'
        expected = '{"key": "val"}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_thinking_block_only(self):
        # When LM Studio emits a thinking block but no JSON payload at all,
        # clean_grammar_output must strip the block and return empty so callers
        # can distinguish "no grammar" from a malformed one.
        input_str = '<think>some thought</think>'
        expected = ''
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_clean_grammar_output_handles_thinking_block_with_whitespace(self):
        # A thinking block followed by only whitespace must also normalize to empty,
        # not a string of spaces that would confuse JSON parsing downstream.
        input_str = '<think>thought</think>   \n\t  '
        expected = ''
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_hash_prompt(self):
        prompt = "a beautiful sunset"
        h1 = hash_prompt(prompt)
        h2 = hash_prompt(prompt)
        h3 = hash_prompt("another prompt")

        self.assertEqual(len(h1), 12)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_hash_prompt_binds_schema_version(self):
        # The grammar cache key must include the schema version so that when
        # PROMPT_SCHEMA_VERSION changes (e.g. from "ernie-v2" to a future v3),
        # stale grammars are not reused across schemas — preserving correctness
        # of the entire generation pipeline on cache hit.
        prompt = "a cat"

        # Snapshot original schema, then mutate it and confirm the hash changes.
        from grammar_generator import PROMPT_SCHEMA_VERSION as original_schema

        with patch("grammar_generator.PROMPT_SCHEMA_VERSION", new="v99"):
            h_mutated = hash_prompt(prompt)

        self.assertNotEqual(h_mutated, hash_prompt(prompt))
        # Restore so subsequent tests see the real schema.
        with patch("grammar_generator.PROMPT_SCHEMA_VERSION", original_schema):
            h_restored = hash_prompt(prompt)
        self.assertEqual(h_restored, hash_prompt(prompt))

    @patch("grammar_generator.Path.exists")
    @patch("grammar_generator.Path.read_text")
    @patch("grammar_generator.paths")
    def test_get_system_prompt(self, mock_paths, mock_read_text, mock_exists):
        mock_paths.templates_dir = Path("/tmp/templates")
        mock_exists.return_value = True
        mock_read_text.return_value = "system prompt content"

        result = get_system_prompt()
        self.assertEqual(result, "system prompt content")

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.get")
    @patch("grammar_generator.requests.post")
    def test_generate_grammar_uses_native_chat_without_reasoning(
        self, mock_post, mock_get, _mock_prompt
    ):
        mock_get.return_value.json.return_value = {
            "models": [
                {
                    "key": "google/gemma-4-26b-a4b-qat",
                    "loaded_instances": [{"id": "google/gemma-4-26b-a4b-qat"}],
                }
            ]
        }
        grammar_json = (
            '{"origin":["A #subject#."],'
            '"subject":["red cat","blue cat","green cat","black cat","white cat"]}'
        )
        mock_post.return_value.json.return_value = {
            "output": [{"type": "message", "content": grammar_json}]
        }

        grammar, was_cached, raw = generate_grammar(
            "a cat",
            base_url="http://localhost:1234/v1",
            use_cache=False,
        )

        assert grammar == grammar_json
        assert raw == grammar
        assert was_cached is False
        mock_post.assert_called_once_with(
            "http://localhost:1234/api/v1/chat",
            json={
                "model": "google/gemma-4-26b-a4b-qat",
                "input": "a cat",
                "system_prompt": "ERNIE instructions",
                "temperature": 0.7,
                "max_output_tokens": 4096,
                "reasoning": "off",
                "store": False,
            },
            timeout=180.0,
        )
        mock_get.assert_called_once_with("http://localhost:1234/api/v1/models", timeout=180.0)
        mock_get.return_value.raise_for_status.assert_called_once_with()
        mock_post.return_value.raise_for_status.assert_called_once_with()

    @patch("grammar_generator.requests.get")
    @patch("grammar_generator.requests.post")
    def test_ensure_lm_model_loaded_waits_for_native_load(self, mock_post, mock_get):
        mock_get.return_value.json.return_value = {
            "models": [
                {"key": "google/gemma-4-26b-a4b-qat", "loaded_instances": []}
            ]
        }
        mock_post.return_value.json.return_value = {"status": "loaded"}

        ensure_lm_model_loaded("http://localhost:1234/v1")

        mock_post.assert_called_once_with(
            "http://localhost:1234/api/v1/models/load",
            json={"model": "google/gemma-4-26b-a4b-qat", "context_length": 8192},
            timeout=180.0,
        )
        mock_post.return_value.raise_for_status.assert_called_once_with()

    @patch("grammar_generator.time.sleep")
    @patch("grammar_generator.requests.get")
    @patch("grammar_generator.requests.post")
    def test_ensure_lm_model_loaded_retries_transient_failure(
        self, mock_post, mock_get, mock_sleep
    ):
        import requests

        mock_get.return_value.json.return_value = {
            "models": [
                {"key": "google/gemma-4-26b-a4b-qat", "loaded_instances": []}
            ]
        }
        failed = MagicMock()
        failed.raise_for_status.side_effect = requests.HTTPError("load canceled")
        loaded = MagicMock()
        loaded.json.return_value = {"status": "loaded"}
        mock_post.side_effect = [failed, loaded]

        ensure_lm_model_loaded("http://localhost:1234/v1")

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(2.0)


class TestEmptyPromptGuard(unittest.TestCase):
    """Tests for rejecting empty/whitespace-only prompts at the entry point."""

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.post")
    @patch("grammar_generator.requests.get")
    def test_generate_grammar_rejects_empty_string(self, mock_get, mock_post, _mock_prompt):
        with self.assertRaises(ValueError, msg="empty prompt must raise ValueError"):
            generate_grammar("")

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.post")
    @patch("grammar_generator.requests.get")
    def test_generate_grammar_rejects_whitespace_only(self, mock_get, mock_post, _mock_prompt):
        with self.assertRaises(ValueError):
            generate_grammar("   \t\n  ")

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.post")
    @patch("grammar_generator.requests.get")
    def test_generate_grammar_passes_nonempty_prompt(self, mock_get, mock_post, _mock_prompt):
        # Sanity: a real prompt still reaches the cache-hit path (returns cached grammar).
        mock_get.return_value.json.return_value = {
            "models": [
                {"key": "google/gemma-4-26b-a4b-qat", "loaded_instances": [{"id": "x"}]}
            ]
        }
        # Pre-populate cache so generate_grammar returns the cached result without calling LM Studio.
        with patch("grammar_generator.get_cached_grammar", return_value='{"origin":["#s#"]}'):
            grammar, was_cached, raw = generate_grammar("a real prompt")
        assert was_cached is True
        # The cache path must be reached — no POST should have been issued.
        mock_post.assert_not_called()


class TestApiRoot(unittest.TestCase):
    """Tests for _api_root — URL normalization before every LM Studio call."""

    def test_strips_trailing_slash_from_v1_url(self):
        # The config base URL often has a trailing slash; _api_root must strip it.
        self.assertEqual(
            _api_root("http://localhost:1234/v1/"),
            "http://localhost:1234",
        )

    def test_strips_v1_from_url_without_trailing_slash(self):
        # The common case from config — must remove /v1 for LM Studio's server root.
        self.assertEqual(
            _api_root("http://localhost:1234/v1"),
            "http://localhost:1234",
        )

    def test_leaves_url_without_v1_suffix_unchanged(self):
        # A raw host URL without /v1 must be returned as-is; the function should not
        # strip characters from an already-normalized root.
        self.assertEqual(
            _api_root("http://localhost:1234"),
            "http://localhost:1234",
        )

    def test_strips_trailing_slash_before_checking_v1(self):
        # URL like http://host/v1// must not have both /v1 AND a trailing slash leaked.
        self.assertEqual(
            _api_root("http://localhost:1234/v1/"),
            "http://localhost:1234",
        )

    def test_strips_double_slash_in_middle_of_url(self):
        # A URL with // in the middle (e.g. http://host//v1) must still normalize to
        # a single-slash host root; LM Studio rejects double slashes in paths.
        self.assertEqual(
            _api_root("http://localhost:1234//v1"),
            "http://localhost:1234",
        )

    def test_leaves_https_url_without_v1_unchanged(self):
        # An HTTPS URL without /v1 must be returned as-is; the function should not
        # strip https or any other scheme prefix.
        self.assertEqual(
            _api_root("https://lmstudio.example.com"),
            "https://lmstudio.example.com",
        )

    def test_strips_trailing_slash_from_non_v1_url(self):
        # A URL ending with a non-v1 suffix (e.g. /v2/) must still have its trailing
        # slash stripped so callers produce clean LM Studio roots; _api_root treats
        # any trailing slash uniformly regardless of the path segment before it.
        self.assertEqual(
            _api_root("http://localhost:1234/v2/"),
            "http://localhost:1234/v2",
        )

    def test_strips_multiple_trailing_slashes_from_non_v1_url(self):
        # When LM Studio returns a URL with multiple trailing slashes (e.g.
        # http://host/v2//), _api_root must collapse them to a single root so the
        # subsequent /api/v1/... append does not produce double-slash paths.
        self.assertEqual(
            _api_root("http://localhost:1234/v2//"),
            "http://localhost:1234/v2",
        )


class TestEmptyPromptGuardMessages(unittest.TestCase):
    """Tests for exact error messages from empty/whitespace prompt rejection."""

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.post")
    @patch("grammar_generator.requests.get")
    def test_generate_grammar_rejects_empty_string_with_exact_message(
        self, mock_get, mock_post, _mock_prompt
    ):
        # The ValueError message "User prompt must not be empty" is the exact
        # contract communicated to callers; downstream consumers rely on it.
        with self.assertRaisesRegex(ValueError, r"^User prompt must not be empty$"):
            generate_grammar("")

    @patch("grammar_generator.get_system_prompt", return_value="ERNIE instructions")
    @patch("grammar_generator.requests.post")
    @patch("grammar_generator.requests.get")
    def test_generate_grammar_rejects_whitespace_with_exact_message(
        self, mock_get, mock_post, _mock_prompt
    ):
        with self.assertRaisesRegex(ValueError, r"^User prompt must not be empty$"):
            generate_grammar("   \t\n  ")


class TestCacheFileNaming(unittest.TestCase):
    """Tests for cache file naming conventions in get_cached_grammar and cache_grammar."""

    @patch("grammar_generator.CACHE_DIR")
    def test_get_cached_grammar_constructs_correct_filename(self, mock_cache_dir):
        # The filename must follow the {prompt_hash}.tracery.json convention; a wrong
        # extension or naming would silently bypass cache lookups and cause redundant
        # LLM calls on every retry.
        prompt_hash = "abcdef123456"
        get_cached_grammar(prompt_hash)

        mock_cache_dir.__truediv__.assert_called_once_with(
            f"{prompt_hash}.tracery.json"
        )

    def test_cache_grammar_creates_all_three_files(self):
        """Verify cache_grammar writes grammar, raw response, and metadata files.

        The production code calls:
          - CACHE_DIR.mkdir(parents=True, exist_ok=True)
          - (CACHE_DIR / f"{prompt_hash}.tracery.json").write_text(grammar)
          - (CACHE_DIR / f"{prompt_hash}.raw.txt").write_text(raw_response)
          - (CACHE_DIR / f"{prompt_hash}.metaprompt.json").write_text(metadata_json)

        We verify the actual files are created with correct content.
        """
        import json
        from tempfile import TemporaryDirectory

        prompt_hash = "abcdef123456"
        grammar = '{"origin": "#prompt#"}'
        raw_response = '<think>...</think>{"origin": "#prompt#"}'
        user_prompt = "a sunset"

        with TemporaryDirectory() as tmpdir:
            # Patch CACHE_DIR to use a temp directory for this test.
            with patch("grammar_generator.CACHE_DIR", Path(tmpdir)) as patched_cache_dir:
                # Patch datetime to control the timestamp
                with patch("grammar_generator.datetime") as mock_datetime:
                    mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

                    result = cache_grammar(
                        prompt_hash=prompt_hash,
                        grammar=grammar,
                        raw_response=raw_response,
                        user_prompt=user_prompt,
                    )

                # Verify return values (still inside patch context)
                self.assertEqual(result[0], grammar)
                self.assertFalse(result[1])  # was_cached flag
                self.assertEqual(result[2], raw_response)

                # Verify all three files were created in the temp directory
                patched_cache_dir.mkdir(parents=True, exist_ok=True)

                grammar_file = patched_cache_dir / f"{prompt_hash}.tracery.json"
                raw_file = patched_cache_dir / f"{prompt_hash}.raw.txt"
                metadata_file = patched_cache_dir / f"{prompt_hash}.metaprompt.json"

                self.assertTrue(grammar_file.exists())
                self.assertTrue(raw_file.exists())
                self.assertTrue(metadata_file.exists())

                # Verify content
                self.assertEqual(grammar_file.read_text(), grammar)
                self.assertEqual(raw_file.read_text(), raw_response)

                # Verify metadata includes hash and required fields
                metadata = json.loads(metadata_file.read_text())
                self.assertIn("hash", metadata)
                self.assertEqual(metadata["hash"], prompt_hash)
                self.assertIn("user_prompt", metadata)
                self.assertIn("created_at", metadata)

    def test_cache_grammar_metadata_includes_required_fields(self):
        """Verify the metadata JSON contains all required fields for cache validation."""
        import json
        from tempfile import TemporaryDirectory

        prompt_hash = "abcdef123456"
        grammar = '{"origin": "#prompt#"}'
        raw_response = 'raw response'
        user_prompt = "a sunset"

        with TemporaryDirectory() as tmpdir:
            # Patch CACHE_DIR to use a temp directory for this test.
            with patch("grammar_generator.CACHE_DIR", Path(tmpdir)) as patched_cache_dir:
                with patch("grammar_generator.datetime") as mock_datetime:
                    mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

                    cache_grammar(
                        prompt_hash=prompt_hash,
                        grammar=grammar,
                        raw_response=raw_response,
                        user_prompt=user_prompt,
                    )

                # Verify metadata (still inside patch context)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                metadata_file = patched_cache_dir / f"{prompt_hash}.metaprompt.json"
                self.assertTrue(metadata_file.exists())

                metadata = json.loads(metadata_file.read_text())

                # Verify all required fields are present with correct types
                self.assertEqual(type(metadata["hash"]), str)
                self.assertEqual(metadata["hash"], prompt_hash)
                self.assertEqual(type(metadata["user_prompt"]), str)
                self.assertEqual(metadata["user_prompt"], user_prompt)
                self.assertEqual(type(metadata["created_at"]), str)
                self.assertIn("prompt_schema", metadata)
                self.assertEqual(metadata["prompt_schema"], "ernie-v2")
                self.assertIn("lm_model", metadata)


if __name__ == "__main__":
    unittest.main()


class TestGrammarStructureValidation(unittest.TestCase):
    def test_accepts_locked_rules_and_five_to_seven_alternatives(self):
        validate_grammar_structure({
            "origin": ["A #subject# in #light#."],
            "subject": ["fox", "owl", "hare", "badger", "deer"],
            "light": ["dawn", "morning", "noon", "evening", "twilight", "moonlight", "fog"],
        })

    def test_rejects_two_to_four_alternatives(self):
        with self.assertRaisesRegex(ValueError, "5\u20137 alternatives"):
            validate_grammar_structure({
                "origin": ["A #subject#."],
                "subject": ["fox", "owl", "hare", "badger"],
            })

    def test_rejects_missing_referenced_rule(self):
        with self.assertRaisesRegex(ValueError, "missing rules: light"):
            validate_grammar_structure({
                "origin": ["A #subject# in #light#."],
                "subject": ["fox", "owl", "hare", "badger", "deer"],
            })

    def test_rejects_grammar_without_variation(self):
        with self.assertRaisesRegex(ValueError, "at least one varying rule"):
            validate_grammar_structure({"origin": ["A fixed prompt."]})

    def test_rejects_too_many_rules(self):
        # The grammar must contain at most 8 rules; more than that cannot be
        # reliably rendered and risks silent degradation of the generation pipeline.
        with self.assertRaisesRegex(ValueError, "at most 8"):
            validate_grammar_structure({
                "origin": ["#a#"],
                "a": ["x"],
                "b": ["y"],
                "c": ["z"],
                "d": ["w"],
                "e": ["v"],
                "f": ["u"],
                "g": ["t"],
                "h": ["s"],
            })

    def test_rejects_non_dict_grammar(self):
        # validate_grammar_structure must reject any non-dict input (lists,
        # strings, primitives) so the caller cannot accidentally feed a malformed
        # LLM response into cache or rendering without an explicit error.
        with self.assertRaises(ValueError):
            validate_grammar_structure([])

    def test_rejects_dict_without_origin(self):
        # A dict that omits "origin" is structurally incomplete; the validator
        # must refuse it before any further processing occurs.
        with self.assertRaisesRegex(ValueError, '"origin" rule'):
            validate_grammar_structure({"foo": ["bar"]})

    def test_rejects_empty_rule_array(self):
        # An empty array for a rule is rejected because an empty list cannot be
        # sampled — catching this early prevents silent failures downstream.
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            validate_grammar_structure({
                "origin": ["#foo#"],
                "foo": [],
            })

    def test_rejects_duplicate_alternatives(self):
        # Duplicate alternatives in a rule break the randomness guarantee; the
        # validator must surface them so LLM output is rejected before caching.
        with self.assertRaisesRegex(ValueError, "duplicate alternatives"):
            validate_grammar_structure({
                "origin": ["#subject#"],
                "subject": ["fox", "owl", "fox", "hare", "deer"],
            })

    def test_rejects_non_list_rule_value(self):
        # When LM Studio returns a rule value as a string instead of an array,
        # validate_grammar_structure must reject it — otherwise malformed output
        # would reach the cache or rendering pipeline. The error message
        # "non-empty array" is the production contract for this validation branch.
        with self.assertRaisesRegex(ValueError, "must be a non-empty array"):
            validate_grammar_structure({
                "origin": ["#subject#"],
                "subject": "fox",
            })

    def test_rejects_invalid_rule_name_with_spaces(self):
        # Rule names containing spaces (e.g. "#my rule#" references) cannot be
        # resolved by Tracery — validate_grammar_structure must reject them early
        # so the error message is clear instead of confusing "missing rules" noise.
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_grammar_structure({
                "origin": ["A #my rule#."],
                "my rule": ["fox", "owl", "hare", "badger", "deer"],
            })

    def test_rejects_rule_name_starting_with_digit(self):
        # Rule names starting with digits cannot be referenced via Tracery syntax;
        # the validator must reject them to prevent silent failures downstream.
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_grammar_structure({
                "origin": ["A #1subject#."],
                "1subject": ["fox", "owl", "hare", "badger", "deer"],
            })

    def test_rejects_empty_option_strings(self):
        # An option that is empty or whitespace-only after stripping cannot be
        # rendered by Tracery — the validator must surface this so LLM output
        # with trailing whitespace in alternatives is rejected before caching.
        with self.assertRaisesRegex(ValueError, "must contain non-empty strings"):
            validate_grammar_structure({
                "origin": ["#subject#"],
                "subject": ["fox", "", "hare", "badger", "deer"],
            })

    def test_rejects_multiple_missing_references(self):
        # When LM Studio references multiple undefined rules, the validator must
        # report all of them so callers see a complete error instead of one at a
        # time — fixing typos becomes deterministic.
        with self.assertRaisesRegex(ValueError, "missing rules: light, subject"):
            validate_grammar_structure({
                "origin": ["A #subject# in #light#."],
                "color": ["red", "blue", "green", "yellow", "black"],
            })

    def test_rejects_locked_origin_referencing_missing_rule(self):
        # Even when origin is a fixed string (not varying), any #rule# references
        # must resolve to existing rules — the validator walks all options regardless
        # of whether the rule is locked, preventing silent rendering failures.
        with self.assertRaisesRegex(ValueError, "missing rules: subject"):
            validate_grammar_structure({
                "origin": ["A #subject#."],
                "color": ["red", "blue", "green", "yellow", "black"],
            })


if __name__ == "__main__":
    unittest.main()
