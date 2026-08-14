"""Tracery wrapper for generating text from grammars."""

import json
import tracery
from tracery.modifiers import base_english


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
        return json.loads(grammar_json)
    except json.JSONDecodeError as e:
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
    return grammar.flatten(f"#{origin}#")


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
    if count <= 0:
        return []

    grammar_dict = parse_grammar(grammar_json)
    grammar = tracery.Grammar(grammar_dict)
    grammar.add_modifiers(base_english)

    results = []
    for _ in range(count):
        text = grammar.flatten(f"#{origin}#")
        results.append(text)

    return results
