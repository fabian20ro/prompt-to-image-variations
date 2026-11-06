import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

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

def _get_env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid value for %s: %s. Using default: %f", key, val, default)
        return default

@dataclass(frozen=True)
class LMStudioConfig:
    """Configuration for LM Studio connection."""
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3.8-27b-splash"

@dataclass(frozen=True)
class ImageGenerationConfig:
    """Default configuration for image generation."""
    default_width: int = 864
    default_height: int = 1152
    model_path: Path = Path.home() / "Library/Caches/mflux/models/flux3-klein-q4"

@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the web server."""
    sse_queue_size: int = 100
    sse_timeout: float = 5.0  # seconds between keepalives
    worker_timeout: float = 300.0  # seconds

@dataclass(frozen=True)
class EnhancementConfig:
    """Configuration for image enhancement."""
    default_softness: float = 0.5
    default_scale: int = 2

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

# Singleton path configuration instance
paths = PathConfig()

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
            base_url=os.environ.get("PROMPT_GEN_LM_STUDIO_URL", LMStudioConfig.base_url),
            model=os.environ.get("PROMPT_GEN_LM_STUDIO_MODEL", LMStudioConfig.model),
        )
        image_generation = ImageGenerationConfig(
            default_width=_get_env_int("PROMPT_GEN_DEFAULT_WIDTH", ImageGenerationConfig.default_width),
            default_height=_get_env_int("PROMPT_GEN_DEFAULT_HEIGHT", ImageGenerationConfig.default_height),
            model_path=Path(os.environ.get(
                "PROMPT_GEN_IMAGE_MODEL_PATH",
                str(ImageGenerationConfig.model_path),
            )),
        )
        server = ServerConfig(
            sse_queue_size=_get_env_int("PROMPT_GEN_SSE_QUEUE_SIZE", ServerConfig.sse_queue_size),
            sse_timeout=_get_env_float("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout),
            worker_timeout=_get_env_float("PROMPT_GEN_WORKER_TIMEOUT", ServerConfig.worker_timeout),
        )
        enhancement = EnhancementConfig(
            default_softness=_get_env_float("PROMPT_GEN_ENHANCE_SOFTNESS", EnhancementConfig.default_softness),
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
