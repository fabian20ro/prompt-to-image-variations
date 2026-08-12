"""ERNIE-Image-Turbo generation through mflux on Apple Silicon."""

import gc
import logging
import random
from pathlib import Path

from config import settings
from lm_studio import unload_all_models

logger = logging.getLogger(__name__)

MODEL_NAME = "ernie-image-turbo"
INFERENCE_STEPS = 8
GUIDANCE = 1.0
QUANTIZATION = 4

_model_cache: dict[bool, object] = {}


def clear_model_cache() -> None:
    """Release the cached ERNIE model."""
    _model_cache.clear()
    gc.collect()


def _get_model(tiled_vae: bool = False):
    """Load the fixed local ERNIE-Image-Turbo q4 model."""
    if tiled_vae in _model_cache:
        return _model_cache[tiled_vae]

    model_path = settings.image_generation.model_path.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"ERNIE q4 model not found: {model_path}. "
            "Provision it with mflux-save before generating images."
        )

    logger.info("Loading ERNIE-Image-Turbo q4 from %s", model_path)

    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.ernie_image import ErnieImage
    except ImportError as exc:
        raise ImportError(
            "mflux 0.18.0+ is required for ERNIE image generation. "
            "Install with: uv sync --extra images"
        ) from exc

    instance = ErnieImage(
        model_config=ModelConfig.ernie_image_turbo(),
        model_path=str(model_path),
        quantize=None,
    )

    if tiled_vae:
        from mflux.models.common.vae.tiling_config import TilingConfig

        instance.tiling_config = TilingConfig()

    _model_cache[tiled_vae] = instance
    return instance


def generate_image(
    prompt: str,
    output_path: Path,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    tiled_vae: bool = False,
) -> Path:
    """Generate one image with fixed ERNIE-Image-Turbo q4 settings."""
    width = width if width is not None else settings.image_generation.default_width
    height = height if height is not None else settings.image_generation.default_height

    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}.")
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height}.")
    if width % 8 != 0:
        raise ValueError(f"Width must be a multiple of 8, got {width}.")
    if height % 8 != 0:
        raise ValueError(f"Height must be a multiple of 8, got {height}.")

    if seed is not None and seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    logger.info("Generating image with seed %d", seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unload_all_models()
    model = _get_model(tiled_vae)
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=INFERENCE_STEPS,
        guidance=GUIDANCE,
        height=height,
        width=width,
    )
    image.save(str(output_path))
    return output_path
