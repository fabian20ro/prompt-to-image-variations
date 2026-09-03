"""SeedVR2 image enhancement using mflux (MLX-based for Apple Silicon)."""

import random
from pathlib import Path

from lm_studio import unload_all_models


# Cache for loaded enhancer model (expensive to load)
_enhancer_cache: dict = {}


def clear_enhancer_cache():
    """Clear the enhancer cache and free memory."""
    global _enhancer_cache
    _enhancer_cache.clear()
    import gc
    gc.collect()


def enhancer_is_loaded(tiled_vae: bool = False) -> bool:
    """Check whether the SeedVR2 model is already loaded in unified memory.

    Returns True only when the enhancer cache holds an entry for the
    given tiling key, so the server UI can tell users whether the next
    enhancement request will start immediately or wait for a cold model
    load.
    """
    return tiled_vae in _enhancer_cache


def _get_enhancer(tiled_vae: bool = False):
    """Get or create a cached SeedVR2 enhancer instance."""
    cache_key = tiled_vae
    if cache_key in _enhancer_cache:
        return _enhancer_cache[cache_key]

    try:
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    except ImportError as e:
        raise ImportError(
            "mflux is required for image enhancement. "
            "Install with: uv sync --extra images\n"
            "Note: mflux requires macOS with Apple Silicon (M1/M2/M3/M4)."
        ) from e

    instance = SeedVR2(quantize=8)

    # SeedVR2 has tiling enabled by default; disable if requested
    if not tiled_vae:
        instance.tiling_config = None

    _enhancer_cache[cache_key] = instance
    return instance


def enhance_image(
    image_path: Path,
    output_path: Path,
    softness: float = 0.5,
    seed: int | None = None,
    tiled_vae: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """
    Enhance a single image using SeedVR2 2x upscaling.

    Args:
        image_path: Path to the source image
        output_path: Path where the enhanced image will be saved
        softness: Enhancement softness (0.0-1.0, default 0.5)
        seed: Random seed for reproducibility (None for random)
        tiled_vae: Enable tiled VAE decoding to reduce memory (default: False)
        width: Override output width in pixels (must be positive multiple of 8; None = model default)
        height: Override output height in pixels (must be positive multiple of 8; None = model default)

    Returns:
        Path to the enhanced image file

    Raises:
        ImportError: If mflux is not installed
        FileNotFoundError: If image_path does not exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not (0.0 <= softness <= 1.0):
        raise ValueError("softness must be between 0.0 and 1.0")

    MAX_DIMENSION = 16384

    if width is not None:
        if width <= 0:
            raise ValueError("Width must be positive.")
        if width > MAX_DIMENSION:
            raise ValueError(f"Width must not exceed {MAX_DIMENSION}.")
        if width % 8 != 0:
            raise ValueError("Width must be a multiple of 8.")

    if height is not None:
        if height <= 0:
            raise ValueError("Height must be positive.")
        if height > MAX_DIMENSION:
            raise ValueError(f"Height must not exceed {MAX_DIMENSION}.")
        if height % 8 != 0:
            raise ValueError("Height must be a multiple of 8.")

    # Import ScaleFactor for 2x upscaling
    try:
        from mflux.utils.scale_factor import ScaleFactor
    except ImportError as e:
        raise ImportError(
            "mflux is required for image enhancement. "
            "Install with: uv sync --extra images"
        ) from e

    # Generate random seed if not specified
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reclaim LM Studio memory before every SeedVR2 operation, including cache hits.
    unload_all_models()
    enhancer = _get_enhancer(tiled_vae)

    # Enhance the image with 2x upscaling
    kwargs = dict(
        seed=seed,
        image_path=str(image_path),
        resolution=ScaleFactor(2),
        softness=softness,
    )
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    result = enhancer.generate_image(**kwargs)

    # Save the enhanced image (overwrite if replacing original)
    result.save(path=str(output_path), overwrite=True)

    return output_path


def collect_images(path_spec: str) -> list[Path]:
    """
    Collect image paths from a file, directory, or glob pattern.

    Args:
        path_spec: Path to file, directory, or glob pattern

    Returns:
        List of image paths (sorted)

    Raises:
        ValueError: If no images found
    """
    from glob import glob

    path = Path(path_spec)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    if path.is_file():
        # Single file
        if path.suffix.lower() in image_extensions:
            return [path]
        raise ValueError(f"Not an image file: {path}")

    elif path.is_dir():
        # Directory - find all images, recursing into symlinked subdirectories
        images = []
        for ext in image_extensions:
            images.extend(path.rglob(f"*{ext}"))
            images.extend(path.rglob(f"*{ext.upper()}"))
        images = sorted(set(images))
        if not images:
            raise ValueError(f"No images found in directory: {path}")
        return images

    else:
        # Treat as glob pattern
        matches = glob(path_spec)
        images = [Path(m) for m in matches if Path(m).suffix.lower() in image_extensions]
        images = sorted(set(images))
        if not images:
            raise ValueError(f"No images found matching pattern: {path_spec}")
        return images
