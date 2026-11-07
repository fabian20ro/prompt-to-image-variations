"""Prompt-template requirements for the caption-only FLUX workflow."""

import json
from types import SimpleNamespace

import pytest

from grammar_generator import get_system_prompt, validate_grammar_structure
from tracery_runner import run_tracery


def test_get_system_prompt_default(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("generic prompt")
    assert get_system_prompt(templates_dir=tmp_path) == "generic prompt"


def test_get_system_prompt_default_uses_configured_templates_dir(tmp_path, monkeypatch):
    configured = tmp_path / "configured_templates"
    configured.mkdir()
    (configured / "system_prompt.txt").write_text("configured prompt")
    monkeypatch.setattr("grammar_generator.paths", SimpleNamespace(templates_dir=configured))
    assert get_system_prompt() == "configured prompt"


def test_project_system_prompt_is_flux_specific():
    prompt = get_system_prompt()
    assert "FLUX 3 Image Klein" in prompt
    assert "Tracery" in prompt
    assert "continuous" in prompt
    assert "TARGET CANVAS" in prompt


@pytest.mark.parametrize("requirement", [
    "Return ONE strict JSON object only",
    "No Markdown, explanations, or reasoning",
    'Include an "origin" array',
    "Each rule is a non-empty array of strings",
    "references use #rule#",
    "No modifiers, actions, recursion, or layout JSON",
    "use at most 8 rules total",
    "At least one reachable rule must vary",
    "7 distinct alternatives",
    "5 or 6 are acceptable",
    "Never 2–4",
    "No duplicates or synonym padding",
    "100–350 words",
    "Avoid padding simple requests",
    "Preserve explicit subjects, attributes, counts, positions, style, and exact visible wording",
    "Preserve visible wording in its original script, case, punctuation, and line breaks",
    "wrap it in double quotes escaped for JSON",
    "foreground, midground, background, frame-left and frame-right",
    "posters, pages, grids, and comics",
    "Express exclusions as what is present",
    "Do not include inference steps, guidance, quantization, or model names",
])
def test_template_retains_generation_guardrails(requirement):
    assert requirement in get_system_prompt()


def test_system_prompt_example_expands_valid_captions():
    grammar = json.loads(get_system_prompt().split("Example grammar:\n", 1)[1])
    validate_grammar_structure(grammar)
    assert all(len(options) == 7 for name, options in grammar.items() if name != "origin")
    captions = run_tracery(json.dumps(grammar), 7)
    assert all('"MOUNTAIN TEA"' in caption and "A single closed ceramic tea canister" in caption for caption in captions)


def test_example_explores_open_medium_and_multiple_visual_axes():
    grammar = json.loads(get_system_prompt().split("Example grammar:\n", 1)[1])
    validate_grammar_structure(grammar)
    varying = [name for name, options in grammar.items() if len(options) > 1]
    assert len(varying) >= 4, "An open brief must demonstrate more than scene/finish variation"

    captions = run_tracery(json.dumps(grammar), 256, seed=42)
    for medium in ("photograph", "watercolor", "oil painting", "linocut", "vector", "3D render", "paper-cut"):
        assert any(medium.lower() in caption.lower().split(".", 1)[0] for caption in captions)
    assert all(caption.count('"MOUNTAIN TEA"') == 1 for caption in captions)


def test_template_treats_unspecified_traits_as_creative_axes_not_fixed_defaults():
    prompt = get_system_prompt()
    assert "Unspecified does not mean fixed or omitted" in prompt
    assert "Unspecified medium and style must vary" in prompt
    assert "Only explicit or logically necessary constraints belong in origin" in prompt
    assert "2–5 meaningful variation groups" in prompt
    assert "use at most 8 rules total" in prompt
    assert "Without a requested medium, use an ordinary photograph" not in prompt
    assert "Do not vary style unless requested" not in prompt
    assert 'Specific visual direction stays fixed unless the user asks to vary it' in prompt
