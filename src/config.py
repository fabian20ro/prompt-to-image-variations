import math
import os
import platform
import logging
from dataclasses import dataclass, field
from pathlib import Path


def _default_model_path() -> Path:
    """Cross-platform default path for the mflux model."""
    if platform.system() == "Darwin":
        return Path.home() / "Library/Caches/mflux/models/ernie-image-turbo-4bit"
    return Path.home() / ".cache" / "mflux" / "models" / "ernie-image-turbo-4bit"

logger = logging.getLogger(__name__)

def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid value for %s: %s. Using default: %d", key, val, default)
        return default

def _get_env_float(key: str, default: float, negative_allowed: bool = True) -> float:
    val = os.environ.get(key)
    if val is None or not val.strip():
        return default
    try:
        fval = float(val)
        if math.isnan(fval):
            logger.warning("Invalid value for %s: %s (NaN). Using default: %f", key, val, default)
            return default
        if math.isinf(fval):
            logger.warning(
                "Invalid value for %s: %s (Infinity). Using default: %f", key, val, default
            )
            return default
        if not negative_allowed and fval < 0:
            logger.warning("Negative value for %s: %s. Using default: %f", key, val, default)
            return default
        return fval
    except ValueError:
        logger.warning("Invalid value for %s: %s. Using default: %f", key, val, default)
        return default

def _get_env_str(key: str, default: str) -> str:
    val = os.environ.get(key)
    if val is None or not val.strip():
        return default
    return val

@dataclass(frozen=True)
class LMStudioConfig:
    """Configuration for LM Studio connection."""
    base_url: str = "http://localhost:1234/v1"
    model: str = "google/gemma-4-26b-a4b-qat"
    timeout: float = 60.0  # seconds

    def __post_init__(self):
        if math.isnan(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive")

@dataclass(frozen=True)
class ImageGenerationConfig:
    """Default configuration for image generation."""
    default_width: int = 864
    default_height: int = 1152
    seed: int = 0
    model_path: Path = _default_model_path()

    def __post_init__(self):
        if self.default_width <= 0:
            raise ValueError("default_width must be positive")
        if self.default_height <= 0:
            raise ValueError("default_height must be positive")

@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the web server."""
    sse_queue_size: int = 100
    sse_timeout: float = 5.0  # seconds between keepalives
    worker_timeout: float = 300.0  # seconds

    def __post_init__(self):
        if math.isnan(self.sse_timeout) or self.sse_timeout <= 0:
            raise ValueError("sse_timeout must be positive")
        if math.isnan(self.worker_timeout) or self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")

@dataclass(frozen=True)
class EnhancementConfig:
    """Configuration for image enhancement."""
    default_softness: float = 0.5
    default_scale: int = 2

    def __post_init__(self):
        if not (0 <= self.default_softness <= 1):
            raise ValueError("default_softness must be between 0 and 1")
        if self.default_scale < 1:
            raise ValueError("default_scale must be at least 1")

@dataclass(frozen=True)
class PathConfig:
    """Centralized path configuration for the application."""

    @property
    def root_dir(self) -> Path:
        """Project root directory."""
        return Path(__file__).parent.parent

    @property
    def src_dir(self) -> Path:
        """Source code directory."""
        return self.root_dir / "src"

    @property
    def generated_dir(self) -> Path:
        """Directory for all generated output."""
        return self.root_dir / "generated"

    @property
    def grammars_dir(self) -> Path:
        """Directory for cached grammars."""
        return self.generated_dir / "grammars"

    @property
    def prompts_dir(self) -> Path:
        """Directory for prompt output runs."""
        return self.generated_dir / "prompts"

    @property
    def saved_dir(self) -> Path:
        """Directory for archived/backup runs."""
        return self.generated_dir / "saved"

    @property
    def queue_path(self) -> Path:
        """Path to the task queue JSON file."""
        return self.generated_dir / "queue.json"

    @property
    def templates_dir(self) -> Path:
        """Directory for system prompt templates."""
        return self.root_dir / "templates"

    @property
    def env_docs_file(self) -> Path:
        """Path to the generated environment variable documentation file."""
        return self.generated_dir / ".env.example"


# Singleton path configuration instance
paths = PathConfig()


def generate_env_example(env_vars: dict[str, dict[str, str]] | None = None) -> Path:
    """Write formatted env var docs to the project's generated directory.

    Returns the path where the file was written so callers can link or verify it.
    """
    text = format_env_docs(env_vars)
    paths.env_docs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.env_docs_file, "w") as f:
        f.write(text)
    return paths.env_docs_file

@dataclass
class Settings:
    """Application settings, can be overridden via environment variables."""
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with PROMPT_GEN_ prefix."""
        lm_studio = LMStudioConfig(
            base_url=_get_env_str("PROMPT_GEN_LM_STUDIO_URL", LMStudioConfig.base_url),
            model=_get_env_str("PROMPT_GEN_LM_STUDIO_MODEL", LMStudioConfig.model),
            timeout=_get_env_float("PROMPT_GEN_LM_STUDIO_TIMEOUT", LMStudioConfig.timeout),
        )
        image_generation = ImageGenerationConfig(
            default_width=_get_env_int("PROMPT_GEN_DEFAULT_WIDTH", ImageGenerationConfig.default_width),
            default_height=_get_env_int("PROMPT_GEN_DEFAULT_HEIGHT", ImageGenerationConfig.default_height),
            seed=_get_env_int("PROMPT_GEN_IMAGE_SEED", 0),
            model_path=Path(os.environ.get(
                "PROMPT_GEN_ERNIE_MODEL_PATH",
                str(_default_model_path()),
            )),
        )
        server = ServerConfig(
            sse_queue_size=_get_env_int("PROMPT_GEN_SSE_QUEUE_SIZE", ServerConfig.sse_queue_size),
            sse_timeout=_get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout, negative_allowed=False),
            worker_timeout=_get_env_float("PROMPT_GEN_WORKER_TIMEOUT", ServerConfig.worker_timeout, negative_allowed=False),
        )
        enhancement = EnhancementConfig(
            default_softness=_get_env_float("PROMPT_GEN_ENHANCE_SOFTNESS", EnhancementConfig.default_softness, negative_allowed=False),
            default_scale=_get_env_int("PROMPT_GEN_ENHANCE_SCALE", EnhancementConfig.default_scale),
        )
        return cls(
            lm_studio=lm_studio,
            image_generation=image_generation,
            server=server,
            enhancement=enhancement,
        )

# Global settings instance - use from_env() for environment-aware settings
settings = Settings.from_env()

ENV_VAR_DOCS: dict[str, dict[str, str]] = {
    "PROMPT_GEN_LM_STUDIO_URL": {
        "desc": "LM Studio server URL",
        "default": LMStudioConfig.base_url,
        "type": "str",
    },
    "PROMPT_GEN_LM_STUDIO_MODEL": {
        "desc": "Model identifier for LM Studio",
        "default": LMStudioConfig.model,
        "type": "str",
    },
    "PROMPT_GEN_LM_STUDIO_TIMEOUT": {
        "desc": "LM Studio request timeout in seconds",
        "default": str(LMStudioConfig.timeout),
        "type": "float",
    },
    "PROMPT_GEN_DEFAULT_WIDTH": {
        "desc": "Default image width for generation (pixels)",
        "default": str(ImageGenerationConfig.default_width),
        "type": "int",
    },
    "PROMPT_GEN_DEFAULT_HEIGHT": {
        "desc": "Default image height for generation (pixels)",
        "default": str(ImageGenerationConfig.default_height),
        "type": "int",
    },
    "PROMPT_GEN_IMAGE_SEED": {
        "desc": "Random seed for deterministic image generation (0=random)",
        "default": str(ImageGenerationConfig.seed),
        "type": "int",
    },
    "PROMPT_GEN_ERNIE_MODEL_PATH": {
        "desc": "Path to the Ernie image model (cross-platform auto-detected if unset)",
        "default": str(_default_model_path()),
        "type": "str",
    },
    "PROMPT_GEN_SSE_QUEUE_SIZE": {
        "desc": "Maximum SSE event queue size per client",
        "default": str(ServerConfig.sse_queue_size),
        "type": "int",
    },
    "PROMPT_GEN_SSE_TIMEOUT": {
        "desc": "SSE keepalive interval in seconds (must be positive)",
        "default": str(ServerConfig.sse_timeout),
        "type": "float",
    },
    "PROMPT_GEN_WORKER_TIMEOUT": {
        "desc": "Worker task timeout in seconds (must be positive)",
        "default": str(ServerConfig.worker_timeout),
        "type": "float",
    },
    "PROMPT_GEN_ENHANCE_SOFTNESS": {
        "desc": "Default enhancement softness factor [0.0, 1.0] (must be non-negative)",
        "default": str(EnhancementConfig.default_softness),
        "type": "float",
    },
    "PROMPT_GEN_ENHANCE_SCALE": {
        "desc": "Default enhancement scale factor (must be >= 1)",
        "default": str(EnhancementConfig.default_scale),
        "type": "int",
    },
}


def format_env_docs(env_vars: dict[str, dict[str, str]] | None = None) -> str:
    """Format ENV_VAR_DOCS as a CLI-help-ready multi-line string."""
    if env_vars is None:
        env_vars = ENV_VAR_DOCS
    lines = ["# Environment Variables", ""]
    for key in sorted(env_vars):
        info = env_vars[key]
        lines.append(
            f"# {key}  ({info['type']})"
        )
        lines.append(f"#   Description: {info['desc']}")
        lines.append(f"#   Default: {info['default']}")
        lines.append("")
    return "\n".join(lines) + "\n\n"
