"""LM Studio integration for generating Tracery grammars."""

import hashlib
import json
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
PROMPT_SCHEMA_VERSION = "ernie-v2"
LM_CONTEXT_LENGTH = 8192
LM_LOAD_ATTEMPTS = 3


def _api_root(base_url: str) -> str:
    """Convert the OpenAI-compatible base URL to LM Studio's server root.

    Strips trailing slashes, ``/v1`` suffix, and any remaining trailing slash so
    callers can safely append ``/api/v1/...`` paths without producing double-slashes
    (e.g. input ``http://host//v1`` must not yield ``http://host/``).
    """
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    return root.rstrip("/")


def ensure_lm_model_loaded(base_url: str, timeout: float = 180.0) -> None:
    """Synchronously load the pinned Gemma model when LM Studio has unloaded it."""
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
        except requests.RequestException:
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


def hash_prompt(user_prompt: str) -> str:
    """Generate a schema-versioned hash for caching ERNIE grammars.

    Args:
        user_prompt: The user's image description
    Returns:
        A 12-character hex hash
    """
    key = f"{PROMPT_SCHEMA_VERSION}:{user_prompt}"
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

    Raises:
        ValueError: If user_prompt is empty or whitespace-only
    """
    if not user_prompt.strip():
        raise ValueError("User prompt must not be empty")

    prompt_hash = hash_prompt(user_prompt)

    # Check cache first
    if use_cache:
        cached = get_cached_grammar(prompt_hash)
        if cached:
            raw_response = get_cached_raw_response(prompt_hash)
            return cached, True, raw_response

    system_prompt = get_system_prompt()
    api_root = _api_root(base_url)
    ensure_lm_model_loaded(base_url)
    response = requests.post(
        f"{api_root}/api/v1/chat",
        json={
            "model": settings.lm_studio.model,
            "input": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_output_tokens": 4096,
            "reasoning": "off",
            "store": False,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    output = response.json().get("output", [])
    raw_response = "\n".join(
        item["content"]
        for item in output
        if item.get("type") == "message" and isinstance(item.get("content"), str)
    ).strip()
    if not raw_response:
        raise ValueError("LM Studio returned no message content")

    # Clean up the grammar (remove markdown code blocks if present)
    grammar = clean_grammar_output(raw_response)

    # Validate JSON and the ERNIE/Tracery structure before caching it.
    try:
        parsed = json.loads(grammar)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON grammar after cleaning: {e}")
    validate_grammar_structure(parsed)

    # Cache the result
    if use_cache:
        cache_grammar(prompt_hash, grammar, raw_response, user_prompt)

    return grammar, False, raw_response


def validate_grammar_structure(grammar: object) -> None:
    """Reject grammars that cannot provide useful 5–7 option Tracery variation."""
    if not isinstance(grammar, dict) or "origin" not in grammar:
        raise ValueError('Grammar must be a JSON object containing an "origin" rule')
    if len(grammar) > 8:
        raise ValueError("Grammar must contain at most 8 rules")

    _RULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    varying_rules = 0
    for name, options in grammar.items():
        if not isinstance(name, str) or not isinstance(options, list) or not options:
            raise ValueError(f"Grammar rule {name!r} must be a non-empty array")
        if not _RULE_NAME_RE.match(name):
            raise ValueError(
                f"Grammar rule name {name!r} is invalid; "
                f"rule names must match [A-Za-z_][A-Za-z0-9_]*"
            )
        if not all(isinstance(option, str) and option.strip() for option in options):
            raise ValueError(f"Grammar rule {name!r} must contain non-empty strings")
        if len(set(options)) != len(options):
            raise ValueError(f"Grammar rule {name!r} contains duplicate alternatives")
        if len(options) > 1:
            varying_rules += 1
            if not 5 <= len(options) <= 7:
                raise ValueError(
                    f"Varying grammar rule {name!r} must contain 5–7 alternatives; "
                    f"received {len(options)}"
                )

    if not varying_rules:
        raise ValueError("Grammar must contain at least one varying rule with 5–7 alternatives")

    references = {
        reference
        for options in grammar.values()
        for option in options
        for reference in re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)#", option)
    }
    missing = references - grammar.keys()
    if missing:
        raise ValueError(f"Grammar references missing rules: {', '.join(sorted(missing))}")


def clean_grammar_output(grammar: str) -> str:
    """
    Clean LLM output by removing thinking blocks, markdown code blocks, and extra whitespace.

    Args:
        grammar: Raw LLM output

    Returns:
        Cleaned JSON grammar content
    """
    # Remove <think>...</think> blocks (including multiline)
    grammar = re.sub(r'<think>.*?</think>', '', grammar, flags=re.DOTALL)

    # Remove markdown code blocks and extract content
    # Handle ```json, ```tracery, or plain ``` markers (with or without newline)
    code_block_match = re.search(r'```(?:json|tracery)?\s*(.*?)```', grammar, flags=re.DOTALL | re.IGNORECASE)
    if code_block_match:
        grammar = code_block_match.group(1)

    grammar = grammar.strip()

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
