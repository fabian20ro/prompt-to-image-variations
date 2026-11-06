import unittest
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch, MagicMock
from grammar_generator import (
    clean_grammar_output, 
    get_system_prompt, 
    hash_prompt, 
    get_cached_grammar, 
    cache_grammar, 
    get_cached_raw_response,
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

    def test_clean_grammar_output_extracts_json(self):
        input_str = 'Some noise here {"a": 1} and some noise there'
        expected = '{"a": 1}'
        self.assertEqual(clean_grammar_output(input_str), expected)

    def test_hash_prompt(self):
        prompt = "a beautiful sunset"
        h1 = hash_prompt(prompt)
        h2 = hash_prompt(prompt)
        h3 = hash_prompt("another prompt")
        
        self.assertEqual(len(h1), 12)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    @patch("grammar_generator.Path.exists")
    @patch("grammar_generator.Path.read_text")
    @patch("grammar_generator.paths")
    def test_get_system_prompt(self, mock_paths, mock_read_text, mock_exists):
        mock_paths.templates_dir = Path("/tmp/templates")
        mock_exists.return_value = True
        mock_read_text.return_value = "system prompt content"
        
        result = get_system_prompt()
        self.assertEqual(result, "system prompt content")

    @patch("grammar_generator.get_system_prompt", return_value="FLUX instructions")
    @patch("grammar_generator.requests.get")
    @patch("grammar_generator.requests.post")
    def test_generate_grammar_uses_native_chat_without_reasoning(
        self, mock_post, mock_get, _mock_prompt
    ):
        mock_get.return_value.json.return_value = {
            "models": [
                {
                    "key": "qwen3.8-27b-splash",
                    "loaded_instances": [{"id": "qwen3.8-27b-splash"}],
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

        assert json.loads(grammar) == json.loads(grammar_json)
        assert raw == grammar_json
        assert was_cached is False
        mock_post.assert_called_once_with(
            "http://localhost:1234/api/v1/chat",
            json={
                "model": "qwen3.8-27b-splash",
                "input": "TARGET CANVAS: 864:1152\na cat",
                "system_prompt": "FLUX instructions",
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
                {"key": "qwen3.8-27b-splash", "loaded_instances": []}
            ]
        }
        mock_post.return_value.json.return_value = {"status": "loaded"}

        ensure_lm_model_loaded("http://localhost:1234/v1")

        mock_post.assert_called_once_with(
            "http://localhost:1234/api/v1/models/load",
            json={"model": "qwen3.8-27b-splash", "context_length": 8192},
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
                {"key": "qwen3.8-27b-splash", "loaded_instances": []}
            ]
        }
        failed = MagicMock()
        error_response = requests.Response()
        error_response.status_code = 409
        failed.raise_for_status.side_effect = requests.HTTPError("load canceled", response=error_response)
        loaded = MagicMock()
        loaded.json.return_value = {"status": "loaded"}
        mock_post.side_effect = [failed, loaded]

        ensure_lm_model_loaded("http://localhost:1234/v1")

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(2.0)


class TestGrammarStructureValidation(unittest.TestCase):
    def test_accepts_locked_rules_and_five_to_seven_alternatives(self):
        validate_grammar_structure({
            "origin": ["A #subject# in #light#."],
            "subject": ["fox", "owl", "hare", "badger", "deer"],
            "light": ["dawn", "morning", "noon", "evening", "twilight", "moonlight", "fog"],
        })

    def test_rejects_two_to_four_alternatives(self):
        with self.assertRaisesRegex(ValueError, "5–7 alternatives"):
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

if __name__ == "__main__":
    unittest.main()
