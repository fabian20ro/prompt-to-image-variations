"""LM Studio integration for generating Tracery grammars."""

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from config import settings, paths


# Default LM Studio endpoint (from centralized config)
LM_STUDIO_BASE_URL = settings.lm_studio.base_url

# Cache directory for generated grammars (from centralized paths)
CACHE_DIR = paths.grammars_dir
PROMPT_SCHEMA_VERSION = "flux3-caption-v1"
LM_CONTEXT_LENGTH = 8192
LM_LOAD_ATTEMPTS = 3
logger = logging.getLogger(__name__)


def _api_root(base_url: str) -> str:
    """Convert the OpenAI-compatible base URL to LM Studio's server root."""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def ensure_lm_model_loaded(base_url: str, timeout: float = 180.0) -> None:
    """Synchronously load the configured local prompt model."""
    from image_generator import clear_model_cache
    from image_enhancer import clear_enhancer_cache
    clear_model_cache()
    clear_enhancer_cache()
    root = _api_root(base_url)
    response = requests.get(f"{root}/api/v1/models", timeout=timeout)
    response.raise_for_status()
    models = response.json().get("models", [])
    model = next((item for item in models if item.get("key") == settings.lm_studio.model), None)
    if model is None:
        raise ValueError(f"LM Studio model is not installed: {settings.lm_studio.model}")
    if model.get("loaded_instances"):
        return

    for attempt in range(LM_LOAD_ATTEMPTS):
        try:
            response = requests.post(
                f"{root}/api/v1/models/load",
                json={"model": settings.lm_studio.model, "context_length": LM_CONTEXT_LENGTH},
                timeout=timeout,
            )
            response.raise_for_status()
            if response.json().get("status") == "loaded":
                return
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            transient = isinstance(exc, (requests.ConnectionError, requests.Timeout)) or status in (408, 409, 429, 500, 502, 503, 504)
            if not transient:
                raise
            if attempt == LM_LOAD_ATTEMPTS - 1:
                raise
        if attempt < LM_LOAD_ATTEMPTS - 1:
            time.sleep(2.0)
    raise ValueError(f"LM Studio did not load model: {settings.lm_studio.model}")


def get_system_prompt(templates_dir: Path | None = None) -> str:
    """
    Load the system prompt from the templates directory.

    Args:
    Returns:
        The system prompt content
    """
    templates_dir = templates_dir or paths.templates_dir

    return (templates_dir / "system_prompt.txt").read_text()


def hash_prompt(user_prompt: str, width: int = 864, height: int = 1152, temperature: float = 0.7) -> str:
    """Generate a schema-versioned hash for caching FLUX captions.

    Args:
        user_prompt: The user's image description
    Returns:
        A 12-character hex hash
    """
    key = json.dumps([PROMPT_SCHEMA_VERSION, settings.lm_studio.model,
                      get_system_prompt(), width, height, temperature, user_prompt])
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def get_cached_grammar(prompt_hash: str) -> str | None:
    """
    Check if a grammar exists in cache for the given prompt hash.

    Args:
        prompt_hash: The hash of the user prompt

    Returns:
        The cached grammar content, or None if not cached
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{prompt_hash}.tracery.json"

    if cache_file.exists():
        return cache_file.read_text()
    return None


def cache_grammar(
    prompt_hash: str,
    grammar: str,
    raw_response: str,
    user_prompt: str,
) -> tuple[str, bool, str]:
    """
    Save a grammar to the cache.

    Args:
        prompt_hash: The hash of the user prompt
        grammar: The cleaned grammar content
        raw_response: The raw LLM response (including thinking blocks)
        user_prompt: The original user prompt (for metadata)

    Returns:
        A tuple of (grammar, was_cached, raw_response)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save the cleaned grammar
    cache_file = CACHE_DIR / f"{prompt_hash}.tracery.json"
    cache_file.write_text(grammar)

    # Save the raw LLM response (with thinking blocks etc.)
    raw_file = CACHE_DIR / f"{prompt_hash}.raw.txt"
    raw_file.write_text(raw_response)

    # Save metadata
    metadata_file = CACHE_DIR / f"{prompt_hash}.metaprompt.json"
    metadata = {
        "user_prompt": user_prompt,
        "created_at": datetime.now().isoformat(),
        "hash": prompt_hash,
        "prompt_schema": PROMPT_SCHEMA_VERSION,
        "lm_model": settings.lm_studio.model,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))

    return grammar, False, raw_response


def get_cached_raw_response(prompt_hash: str) -> str | None:
    """
    Get the raw LLM response from cache for the given prompt hash.

    Args:
        prompt_hash: The hash of the user prompt

    Returns:
        The cached raw response content, or None if not cached
    """
    raw_file = CACHE_DIR / f"{prompt_hash}.raw.txt"
    if raw_file.exists():
        return raw_file.read_text()
    return None


def generate_grammar(
    user_prompt: str,
    base_url: str = LM_STUDIO_BASE_URL,
    use_cache: bool = True,
    temperature: float = 0.7,
    width: int = 864,
    height: int = 1152,
) -> tuple[str, bool, str | None]:
    """
    Generate a Dada Engine grammar for the given prompt using LM Studio.

    Args:
        user_prompt: The user's image description
        base_url: LM Studio API base URL
        use_cache: Whether to check/use cached grammars
        temperature: LLM temperature for generation

    Returns:
        Tuple of (grammar content, was_cached, raw_response)
    """
    prompt_hash = hash_prompt(user_prompt, width, height, temperature)

    # Check cache first
    if use_cache:
        cached = get_cached_grammar(prompt_hash)
        if cached:
            try:
                validate_grammar_structure(json.loads(cached))
            except (ValueError, TypeError):
                pass  # Stale/corrupt cache is a miss, never an executable grammar.
            else:
                raw_response = get_cached_raw_response(prompt_hash)
                return cached, True, raw_response

    system_prompt = get_system_prompt()
    api_root = _api_root(base_url)
    ensure_lm_model_loaded(base_url)
    request = {
            "model": settings.lm_studio.model,
            "input": f"TARGET CANVAS: {width}:{height}\n{user_prompt}",
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_output_tokens": 4096,
            "reasoning": "off",
            "store": False,
    }
    for attempt in range(2):
        response = requests.post(f"{api_root}/api/v1/chat", json=request, timeout=180.0)
        response.raise_for_status()
        output = response.json().get("output", [])
        raw_response = "\n".join(
            item["content"] for item in output
            if item.get("type") == "message" and isinstance(item.get("content"), str)
        ).strip()
        grammar = clean_grammar_output(raw_response)
        try:
            parsed = json.loads(grammar)
            # Only exact, explicitly quoted user wording is literalized. Arbitrary
            # model-produced actions/layout arrays remain invalid and are repaired.
            literals = dict.fromkeys(a or b for a, b in re.findall(r'"([^"\n]+)"|“([^”\n]+)”', user_prompt))
            replacements = {}
            for literal in literals:
                if any(char in literal for char in "#[]"):
                    escaped = re.sub(r"([\\#\[\]])", r"\\\1", literal)
                    replacements[literal] = escaped
                    replacements[escaped] = escaped
            # One longest-match pass preserves already escaped and overlapping text.
            literal_pattern = "|".join(re.escape(text) for text in sorted(replacements, key=len, reverse=True))
            if isinstance(parsed, dict):
                for name, options in parsed.items():
                    if isinstance(options, list):
                        for index, option in enumerate(options):
                            if not isinstance(option, str):
                                continue
                            if literal_pattern:
                                option = re.sub(literal_pattern, lambda match: replacements[match.group()], option)
                            parsed[name][index] = option
            validate_grammar_structure(parsed)
            grammar = json.dumps(parsed, ensure_ascii=False)
            break
        except ValueError as exc:
            if attempt:
                raise ValueError(f"Invalid grammar after one repair: {exc}") from exc
            logger.warning("Repairing rejected grammar: %.500s", str(exc))
            request = {**request, "input": (
                f"TARGET CANVAS: {width}:{height}\n{user_prompt}\n"
                f"Repair the previous attempt by generating a new grammar. Validation error: {str(exc)[:500]}\n"
                "Return only corrected JSON, preserving the user's constraints."
            )}

    # Cache the result
    if use_cache:
        cache_grammar(prompt_hash, grammar, raw_response, user_prompt)

    return grammar, False, raw_response


def validate_grammar_structure(grammar: object, require_variation: bool = True) -> None:
    """Reject grammars that cannot provide useful 5–7 option Tracery variation."""
    if not isinstance(grammar, dict) or "origin" not in grammar:
        raise ValueError('Grammar must be a JSON object containing an "origin" rule')
    if len(grammar) > 8:
        raise ValueError("Grammar must contain at most 8 rules")

    varying_rules = 0
    for name, options in grammar.items():
        if not isinstance(name, str) or not isinstance(options, list) or not options:
            raise ValueError(f"Grammar rule {name!r} must be a non-empty array")
        if not all(isinstance(option, str) and option.strip() for option in options):
            raise ValueError(f"Grammar rule {name!r} must contain non-empty strings")
        if len(set(options)) != len(options):
            raise ValueError(f"Grammar rule {name!r} contains duplicate alternatives")
        if len(options) > 1:
            varying_rules += 1
            if require_variation and not 5 <= len(options) <= 7:
                raise ValueError(
                    f"Varying grammar rule {name!r} must contain 5–7 alternatives; "
                    f"received {len(options)}"
                )

    if require_variation and not varying_rules:
        raise ValueError("Grammar must contain at least one varying rule with 5–7 alternatives")

    references_by_rule = {}
    for name, options in grammar.items():
        references_by_rule[name] = set()
        for option in options:
            # Tracery escapes the next character, including another backslash.
            # Mask pairs without joining fragments into a synthetic reference.
            unescaped = re.sub(r"\\[\s\S]", "\0", option)
            tokens = re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)#", unescaped)
            remainder = re.sub(r"#[A-Za-z_][A-Za-z0-9_]*#", "", unescaped)
            if any(char in remainder for char in "#[]"):
                raise ValueError("Use simple #rule# references; escape literal #, [ and ]; actions/modifiers are unsupported")
            references_by_rule[name].update(tokens)
    references = set().union(*references_by_rule.values())
    missing = references - grammar.keys()
    if missing:
        raise ValueError(f"Grammar references missing rules: {', '.join(sorted(missing))}")

    visited, active = set(), set()
    def visit(name):
        if name in active:
            raise ValueError(f"Grammar contains a cycle through {name!r}")
        if name in visited:
            return
        active.add(name)
        for reference in references_by_rule[name]:
            visit(reference)
        active.remove(name)
        visited.add(name)
    visit("origin")
    if require_variation and not any(len(grammar[name]) >= 5 for name in visited):
        raise ValueError("Grammar needs variation reachable from origin")


def clean_grammar_output(grammar: str) -> str:
    """
    Clean LLM output by removing thinking blocks, markdown code blocks, and extra whitespace.

    Args:
        grammar: Raw LLM output

    Returns:
        Cleaned JSON grammar content
    """
    try:
        json.loads(grammar)
        return grammar.strip()
    except json.JSONDecodeError:
        pass
    # Remove <think>...</think> blocks (including multiline)
    grammar = re.sub(r'<think>.*?</think>', '', grammar, flags=re.DOTALL)

    # Remove markdown code blocks and extract content
    # Handle ```json, ```tracery, or plain ``` markers (with or without newline)
    code_block_match = re.search(r'```(?:json|tracery)?\s*(.*?)```', grammar, flags=re.DOTALL | re.IGNORECASE)
    if code_block_match:
        grammar = code_block_match.group(1)

    grammar = grammar.strip()

    # Valid JSON may contain exact user typography; never normalize its string values.
    try:
        json.loads(grammar)
        return grammar
    except json.JSONDecodeError:
        pass

    # Extract valid JSON from prose before attempting delimiter repair. Typography
    # inside valid string values belongs to the user, even with a prose preamble.
    for match in re.finditer(r"\{", grammar):
        try:
            _, end = json.JSONDecoder().raw_decode(grammar[match.start():])
            return grammar[match.start():match.start() + end]
        except json.JSONDecodeError:
            continue

    # Normalize smart/curly quotes to straight ASCII quotes
    quote_replacements = {
        '\u201c': '"',  # U+201C left double quotation mark
        '\u201d': '"',  # U+201D right double quotation mark
        '\u2018': "'",  # U+2018 left single quotation mark
        '\u2019': "'",  # U+2019 right single quotation mark
    }
    for smart, straight in quote_replacements.items():
        grammar = grammar.replace(smart, straight)

    # Try to find the first valid JSON object or array in the text
    match = re.search(r'[\{\[]', grammar)
    if match:
        start_idx = match.start()
        try:
            # Use raw_decode to find the first valid JSON object or array and its end position
            decoder = json.JSONDecoder()
            _, end_pos = decoder.raw_decode(grammar[start_idx:])
            grammar = grammar[start_idx:start_idx + end_pos]
        except (json.JSONDecodeError, ValueError):
            pass

    return grammar
