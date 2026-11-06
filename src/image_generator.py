"""Local distilled FLUX 3 Image Klein q4 generation."""

import gc
import json
import logging
import random
import sys
from pathlib import Path
from importlib import metadata

from config import settings
from lm_studio import unload_all_models

logger = logging.getLogger(__name__)

MODEL_NAME = "flux3-image-klein"
INFERENCE_STEPS = 16
GUIDANCE = None
QUANTIZATION = 4

_model_cache: dict[Path, object] = {}


def generation_settings() -> dict:
    """Renderer provenance shared by run metadata and individual image sidecars."""
    path = settings.image_generation.model_path.expanduser()
    try:
        mflux_version = metadata.version("mflux")
    except metadata.PackageNotFoundError:
        mflux_version = None
    provenance = path / "provenance.json"
    source = json.loads(provenance.read_text()) if provenance.is_file() else {}
    return {
        "model": MODEL_NAME, "steps": INFERENCE_STEPS, "guidance": GUIDANCE,
        "quantize": QUANTIZATION, "variant": "distilled-noCFG",
        "checkpoint": str(path), "mflux_version": mflux_version,
        "source_revision": source.get("source_revision"),
    }


def validate_dimensions(width: int, height: int) -> None:
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise ValueError("Width and height must be between 64 and 4096.")
    if width % 16 or height % 16:
        raise ValueError("Width and height must be multiples of 16.")


def validate_checkpoint() -> Path:
    """Fail before MLX import when the saved local q4 checkpoint is incomplete."""
    path = settings.image_generation.model_path.expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"FLUX 3 q4 model not found: {path}. Provision with mflux-save.")
    try:
        for component in ("transformer", "text_encoder", "vae"):
            directory = path / component
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            if str(index["metadata"]["quantization_level"]) != "4" or not index["weight_map"]:
                raise ValueError("Expected quantization_level 4 and nonempty weights")
            for shard in set(index["weight_map"].values()):
                file = (directory / shard).resolve()
                if not file.is_relative_to(path.resolve()) or not file.is_file() or not file.stat().st_size:
                    raise ValueError(f"Missing shard {file}")
        if not (path / "text_encoder" / "tokenizer_config.json").is_file():
            raise ValueError("Missing tokenizer configuration")
        tokenizer = path / "text_encoder"
        if not (tokenizer / "tokenizer.json").is_file() and not all(
            (tokenizer / name).is_file() for name in ("vocab.json", "merges.txt")
        ):
            raise ValueError("Missing tokenizer data")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid FLUX 3 q4 checkpoint at {path}: {exc}") from exc
    return path


def clear_model_cache() -> None:
    """Release the cached image model."""
    _model_cache.clear()
    gc.collect()
    if mx := sys.modules.get("mlx.core"):
        mx.clear_cache()


def _get_model():
    """Load the fixed local FLUX 3 q4 model."""
    model_path = validate_checkpoint()
    if model_path in _model_cache:
        return _model_cache[model_path]

    logger.info("Loading FLUX 3 Image Klein q4 from %s", model_path)

    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux3.variants import Flux3Klein
    except ImportError as exc:
        raise ImportError(
            "mflux 0.24.0 is required for FLUX 3 image generation. "
            "Install with: uv sync --extra images"
        ) from exc

    instance = Flux3Klein(
        model_config=ModelConfig.flux3_image_klein(),
        model_path=str(model_path),
        quantize=None,
    )

    _model_cache[model_path] = instance
    return instance


def generate_image(
    prompt: str,
    output_path: Path,
    seed: int | None = None,
    width: int = 864,
    height: int = 1152,
) -> Path:
    """Generate one image with fixed distilled FLUX 3 q4 settings."""
    validate_dimensions(width, height)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    validate_checkpoint()
    from image_enhancer import clear_enhancer_cache
    clear_enhancer_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unload_all_models()
    model = _get_model()
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=INFERENCE_STEPS,
        height=height,
        width=width,
    )
    image.save(path=str(output_path), overwrite=True)
    output_path.with_suffix(".generation.json").write_text(json.dumps({
        **generation_settings(), "seed": seed, "prompt": prompt,
        "width": width, "height": height,
    }, indent=2, ensure_ascii=False))
    return output_path
