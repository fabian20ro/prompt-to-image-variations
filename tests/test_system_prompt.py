from pathlib import Path
from src.grammar_generator import get_system_prompt, hash_prompt, get_cached_grammar, cache_grammar, get_cached_raw_response
import json

def test_get_system_prompt_default(tmp_path):
    # Setup: create a dummy template file
    template_file = tmp_path / "system_prompt.txt"
    template_file.write_text("generic prompt")
    
    # When
    prompt = get_system_prompt(templates_dir=tmp_path)
    
    # Then
    assert prompt == "generic prompt"

def test_project_system_prompt_is_ernie_specific():
    prompt = get_system_prompt()
    assert "ERNIE-Image-Turbo" in prompt
    assert "Tracery" in prompt
    assert "Subject: exact identity" in prompt
    assert "Description and details" in prompt
    assert "Style and medium" in prompt
    assert "Technical and capture finish" in prompt
    assert "exactly 7 distinct" in prompt
    assert "Never use 2–4 alternatives" in prompt


def test_system_prompt_contains_all_content_type_patterns():
    """Verify the template includes all five documented content-type patterns."""
    prompt = get_system_prompt()
    for pattern in ["Portrait", "Product", "Poster/infographic", "Comic", "Landscape/concept art"]:
        assert f"- {pattern}:" in prompt, f"Missing content-type pattern: {pattern}"


def test_system_prompt_enforces_single_quote_visible_text():
    """Verify the template requires wrapping visible text in single quotes."""
    prompt = get_system_prompt()
    assert "wrap each visible string in single quotes" in prompt


def test_system_prompt_forbids_vague_placeholders():
    """The template explicitly lists forbidden vague placeholders to prevent low-quality prompts.

    These phrases are listed as examples of what the generator must never emit.
    Their presence in the system prompt confirms the quality guardrail is active.
    """
    prompt = get_system_prompt()
    for placeholder in [
        '"some text"',
        '"a button"',
        '"relevant facts"',
        '"and so on"',
    ]:
        assert (
            placeholder in prompt
        ), f"Forbidden vague placeholder guardrail missing: {placeholder}"


def test_system_prompt_forbids_markdown_and_commentary():
    """Verify the template explicitly forbids Markdown formatting, commentary, and reasoning output.

    The generator must return pure JSON — never prose, explanations, or markdown-wrapped blocks.
    These prohibitions are listed in the system prompt to prevent low-quality responses.
    """
    prompt = get_system_prompt()
    for prohibition in [
        "No Markdown",
        "commentary",
        "reasoning",
    ]:
        assert (
            prohibition in prompt
        ), f"Output guardrail missing from system prompt: {prohibition}"


def test_system_prompt_forbids_duplicate_alternatives():
    """Verify the template explicitly forbids duplicating or near-duplicating Tracery alternatives.

    The grammar generator must produce truly distinct options per rule — not inflated counts of
    nearly identical entries. This guardrail prevents wasted variation and low-quality grammars.
    """
    prompt = get_system_prompt()
    for prohibition in [
        "duplicate",
        "rephrase",
    ]:
        assert (
            prohibition in prompt
        ), f"Anti-duplicate guardrail missing from system prompt: {prohibition}"


def test_system_prompt_enforces_output_quality_constraints():
    """Verify the template enforces word count, contradiction avoidance, style consistency, and translation rules.

    These constraints prevent low-quality or inconsistent outputs across series generation.
    """
    prompt = get_system_prompt()
    for constraint in [
        "50–150 words",
        "contradictory",
        "consistent across a series",
        "Do not translate it",
    ]:
        assert (
            constraint in prompt
        ), f"Output quality constraint missing from system prompt: {constraint}"


def test_system_prompt_forbids_inference_metadata():
    """Verify the template forbids embedding inference metadata into prompts.

    The generator must never write guidance scale, quantization info, or numeric resolution
    into the expanded prompt — the application controls those parameters separately.
    """
    prompt = get_system_prompt()
    for forbidden in [
        "guidance scale",
        "quantization",
        "numeric output resolution",
    ]:
        assert (
            forbidden in prompt
        ), f"Inference metadata prohibition missing from system prompt: {forbidden}"


def test_system_prompt_requires_origin_rule():
    """Verify the template requires an "origin" rule in every grammar.

    The origin rule is the entry point for all expansion; without it
    no valid Tracery grammar can be generated. Its presence confirms
    the structural invariant is enforced by the prompt itself.
    """
    prompt = get_system_prompt()
    assert '"origin"' in prompt, "Origin rule requirement missing from system prompt"


def test_system_prompt_requires_hash_references():
    """Verify the template mandates #rule# references and forbids square brackets.

    Tracery grammar validity depends on hash-delimited references;
    square-bracket placeholders would produce invalid grammars.
    This assertion confirms the format contract is enforced upstream.
    """
    prompt = get_system_prompt()
    assert "#rule#" in prompt, "Hash-reference requirement missing from system prompt"
    assert "square-bracket" in prompt or "[placeholder" in prompt, (
        "Square-bracket placeholder prohibition missing from system prompt"
    )


def test_system_prompt_enforces_grammar_rule_format():
    """Verify the template enforces Tracery grammar structural invariants.

    The origin rule is the expansion entry point; without it no valid grammar can be generated.
    Every rule value must be a non-empty array of strings so Tracery can expand them,
    and every #rule# reference must resolve or generation would fail at runtime.
    These are structural guarantees that protect downstream consumers from malformed grammars.
    """
    prompt = get_system_prompt()
    for invariant in [
        "non-empty array of strings",
        "#rule# reference must resolve",
        "\"origin\" rule",
    ]:
        assert (
            invariant in prompt
        ), f"Grammar structural invariant missing from system prompt: {invariant}"


def test_system_prompt_limits_rule_count():
    """Verify the template caps rule count at 8 to keep grammars compact and parseable.

    Tracery grammars with too many rules degrade performance and become hard to debug;
    the limit is enforced explicitly in the system prompt before generation begins.
    """
    prompt = get_system_prompt()
    assert "at most 8 rules" in prompt, (
        "Rule count cap missing from system prompt — grammars could balloon unbounded"
    )


def test_system_prompt_enforces_no_negative_prompt():
    """Verify the template explicitly forbids negative prompts for ERNIE.

    ERNIE-Image-Turbo has no negative-prompt input, so exclusions must be stated
    concretely in the positive prompt itself. This guardrail prevents models from
    emitting invalid negative-prompt syntax into expanded prompts.
    """
    prompt = get_system_prompt()
    assert (
        "no negative-prompt" in prompt
    ), "ERNIE no-negative-prompt constraint missing from system prompt"


def test_system_prompt_enforces_varying_rule_minimum():
    """Verify the template requires at least one varying rule with 7 alternatives.

    The grammar must have variation to produce diverse outputs; a single-option grammar
    defeats the purpose of Tracery expansion. The minimum alternative count (7) prevents
    degenerate grammars that would collapse into identical outputs.
    """
    prompt = get_system_prompt()
    for invariant in [
        "exactly 7",
        "distinct, coherent alternatives",
        "at least one varying rule",
    ]:
        assert (
            invariant in prompt
        ), f"Varying-rule minimum requirement missing from system prompt: {invariant}"


def test_system_prompt_requires_concrete_position_vocabulary():
    """Verify the template mandates concrete spatial language instead of vague placement.

    The model must use explicit directional vocabulary (left/right, foreground/background)
    rather than loose terms like 'around' or 'nearby'. This ensures spatial descriptions
    are precise enough for image generation to interpret correctly.
    """
    prompt = get_system_prompt()
    assert "concrete positions" in prompt, "Concrete position requirement missing from system prompt"
    for direction in ["left/right", "foreground/background"]:
        assert (
            direction in prompt
        ), f"Direction vocabulary requirement missing: {direction}"
