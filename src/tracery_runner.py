"""Tracery wrapper for generating text from grammars."""

import json
import re
import tracery
from tracery.modifiers import base_english
from grammar_generator import validate_grammar_structure


class TraceryError(Exception):
    """Raised when Tracery encounters an error."""
    pass


def parse_grammar(grammar_json: str) -> dict:
    """
    Parse a JSON grammar string into a dictionary.

    Args:
        grammar_json: JSON string containing Tracery grammar

    Returns:
        Parsed grammar dictionary

    Raises:
        TraceryError: If JSON is invalid
    """
    try:
        result = json.loads(grammar_json)
        validate_grammar_structure(result, require_variation=False)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise TraceryError(f"Invalid JSON grammar: {e}")


def generate_one(grammar_dict: dict, origin: str = "origin") -> str:
    """
    Generate a single text from a Tracery grammar.

    Args:
        grammar_dict: Tracery grammar as a dictionary
        origin: The starting rule (default: "origin")

    Returns:
        Generated text
    """
    grammar = tracery.Grammar(grammar_dict)
    grammar.add_modifiers(base_english)
    text = grammar.flatten(f"#{origin}#")
    # Rule fragments sometimes supply the same sentence-ending dot as origin.
    # Preserve exact visible text inside quotes, including intentional punctuation.
    parts = re.split(r'("(?:\\.|[^"\\])*"|“[^”]*”)', text)
    return "".join(part if index % 2 else re.sub(r"(?<!\.)\.{2}(?!\.)", ".", part)
                   for index, part in enumerate(parts))


def run_tracery(grammar_json: str, count: int = 500, origin: str = "origin") -> list[str]:
    """
    Generate multiple texts from a Tracery grammar.

    Args:
        grammar_json: JSON string containing Tracery grammar
        count: Number of variations to generate
        origin: The starting rule (default: "origin")

    Returns:
        List of generated texts

    Raises:
        TraceryError: If grammar is invalid
    """
    grammar_dict = parse_grammar(grammar_json)

    results = []
    for _ in range(count):
        text = generate_one(grammar_dict, origin)
        results.append(text)

    return results
