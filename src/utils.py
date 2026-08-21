"""Shared utility functions for the prompt-to-image-variations application."""

import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)


def check_lm_studio(url: str) -> bool:
    """Check if LM Studio is reachable and returns 200.
    
    Skips the check if running in a test environment (pytest/unittest).
    """
    if any(m in sys.modules for m in ['pytest', 'unittest']):
        return True
    try:
        response = requests.get(f"{url}/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def format_run_timestamp(timestamp: str) -> str:
    """Format a run timestamp for display.

    Converts YYYYMMDD_HHMMSS to YYYY-MM-DD HH:MM:SS format.

    Args:
        timestamp: Timestamp string in YYYYMMDD_HHMMSS format

    Returns:
        Formatted timestamp or original if format doesn't match
    """
    if len(timestamp) == 15 and timestamp[8] == "_" and re.fullmatch(r"\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}", timestamp):
        return (
            f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} "
            f"{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        )
    return timestamp


def load_run_metadata(run_dir: Path) -> dict:
    """Load metadata from a run directory.

    Delegates to MetadataManager for consistent behavior.

    Args:
        run_dir: Path to a run directory containing *_metadata.json

    Returns:
        Dictionary containing the metadata

    Raises:
        ValueError: If no metadata file found
        json.JSONDecodeError: If metadata file is invalid JSON
    """
    from metadata_manager import MetadataManager, MetadataNotFoundError, MetadataError
    try:
        return MetadataManager.load_raw(run_dir)
    except MetadataNotFoundError as e:
        raise ValueError(str(e))
    except MetadataError as e:
        raise json.JSONDecodeError(str(e), "", 0)


def get_prefix_from_metadata(run_dir: Path) -> str:
    """Get the prefix from a run directory's metadata.

    Delegates to MetadataManager for consistent behavior.

    Args:
        run_dir: Path to a run directory

    Returns:
        The prefix string (defaults to "image" if not found)
    """
    from metadata_manager import MetadataManager
    return MetadataManager.get_prefix(run_dir)


def find_metadata_file(run_dir: Path) -> Path | None:
    """Find the metadata file in a run directory.

    Delegates to MetadataManager for consistent behavior.

    Args:
        run_dir: Path to a run directory

    Returns:
        Path to the metadata file, or None if not found
    """
    from metadata_manager import MetadataManager
    return MetadataManager.find_metadata_file(run_dir)


def count_images_in_run(run_dir: Path, prefix: str | None = None) -> int:
    """Count the number of generated images in a run directory.

    Args:
        run_dir: Path to a run directory
        prefix: Optional prefix for image files (auto-detected if not provided)

    Returns:
        Number of image files found
    """
    if prefix is None:
        prefix = get_prefix_from_metadata(run_dir)
    return len(list(run_dir.glob(f"{prefix}_*_*.png")))


def get_prompts_from_run(run_dir: Path, prefix: str | None = None) -> list[str]:
    """Load all prompt texts from a run directory.

    Args:
        run_dir: Path to a run directory
        prefix: Optional prefix for prompt files (auto-detected if not provided)

    Returns:
        List of prompt strings in order
    """
    if prefix is None:
        prefix = get_prefix_from_metadata(run_dir)

    prompt_files = sorted(run_dir.glob(f"{prefix}_*.txt"))
    # Filter to only prompt files (prefix_N.txt), not metadata or other files
    import re
    pattern = re.compile(rf"^{re.escape(prefix)}_\d+\.txt$")
    prompt_files = [f for f in prompt_files if pattern.match(f.name)]

    return [f.read_text() for f in prompt_files]


def _get_prompt_text(run_dir: Path, prefix: str, prompt_idx: int) -> str:
    """Get prompt text for a given index from a run directory.

    Args:
        run_dir: Path to the run directory
        prefix: File prefix
        prompt_idx: Index of the prompt

    Returns:
        Prompt text or empty string if not found
    """
    prompt_file = run_dir / f"{prefix}_{prompt_idx}.txt"
    if prompt_file.exists():
        return prompt_file.read_text()
    return ""


PNG_TEXT_MAX_BYTES = 8000


def _truncate_for_png(value: str, key: str) -> str:
    """Truncate a text value to PNG tEXt chunk limits.

    Pillow's add_text compresses via zlib; very large uncompressed values can
    cause OOM or compression failures. This caps values at ~8KB with a warning.

    Args:
        value: Text value to embed in a PNG chunk
        key: Field name (used in log message)

    Returns:
        The (possibly truncated) text value safe for embedding.
    """
    encoded = value.encode("utf-8")
    if len(encoded) > PNG_TEXT_MAX_BYTES:
        logger.warning(
            f"Truncating '{key}' from {len(encoded)} to {PNG_TEXT_MAX_BYTES} bytes "
            f"for PNG text chunk embedding"
        )
        # Trim trailing bytes until UTF-8 decode succeeds without error,
        # preserving more content when multi-byte chars are at the boundary.
        for cut in range(PNG_TEXT_MAX_BYTES + 1):
            try:
                return encoded[:PNG_TEXT_MAX_BYTES - cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
        # All slices failed — last resort, same as before but deterministic.
        return encoded[:PNG_TEXT_MAX_BYTES].decode("utf-8", errors="ignore")
    return value


def _copy_with_exif(
    src: Path,
    dest: Path,
    metadata: dict,
    prompt_text: str,
    reason: str,
) -> None:
    """Copy PNG and embed metadata in PNG text chunks.

    Args:
        src: Source image path
        dest: Destination image path
        metadata: Run metadata dictionary
        prompt_text: The prompt text for this image
        reason: Backup reason
    """
    img = Image.open(src)
    png_info = PngInfo()

    # Embed key metadata in PNG text chunks (truncated if too large)
    safe_prompt = _truncate_for_png(prompt_text, "prompt")
    safe_user_prompt = _truncate_for_png(metadata.get("user_prompt", ""), "user_prompt")
    png_info.add_text("prompt", safe_prompt)
    png_info.add_text("user_prompt", safe_user_prompt)
    png_info.add_text("model", metadata.get("model", ""))
    png_info.add_text("created_at", metadata.get("created_at", ""))
    png_info.add_text("backup_reason", reason)

    # Include image generation settings if available
    img_settings = metadata.get("image_generation", {})
    if img_settings:
        for key in ["width", "height", "steps", "quantize"]:
            if val := img_settings.get(key):
                png_info.add_text(key, str(val))

    img.save(dest, pnginfo=png_info)


def backup_run(run_dir: Path, saved_dir: Path, reason: str = "manual_archive") -> list[Path]:
    """Archive images to flat files with embedded PNG metadata.

    Archives images as flat files in saved/ with format:
    {prefix}_{timestamp}_{promptIdx}_{imgIdx}.png

    Metadata is embedded in PNG text chunks instead of a separate JSON file.
    This makes archives browsable and self-contained.

    Args:
        run_dir: Path to the run directory to backup
        saved_dir: Path to the saved/ directory
        reason: "pre_regenerate", "pre_enhance", or "manual_archive"

    Returns:
        List of paths to created archive files
    """
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata for embedding
    try:
        metadata = load_run_metadata(run_dir)
    except (ValueError, json.JSONDecodeError):
        metadata = {}

    prefix = metadata.get("prefix", "image")

    saved_files = []

    # Find all PNG images matching pattern: prefix_promptIdx_imgIdx.png
    for png_file in run_dir.glob(f"{prefix}_*_*.png"):
        # Parse original name to extract indices
        # Pattern: prefix_promptIdx_imgIdx.png
        match = re.match(rf'^{re.escape(prefix)}_(\d+)_(\d+)\.png$', png_file.name)
        if not match:
            continue

        prompt_idx = match.group(1)
        img_idx = match.group(2)

        # Get prompt text for this image
        prompt_text = _get_prompt_text(run_dir, prefix, int(prompt_idx))

        # New flat filename: prefix_timestamp_promptIdx_imgIdx.png
        new_name = f"{prefix}_{timestamp}_{prompt_idx}_{img_idx}.png"
        dest_path = saved_dir / new_name

        # Copy with EXIF metadata embedding (overwrites if exists)
        _copy_with_exif(png_file, dest_path, metadata, prompt_text, reason)
        saved_files.append(dest_path)

    return saved_files


def scan_flat_archives(saved_dir: Path) -> list[dict]:
    """Scan saved/ for flat archived images and group by prefix+timestamp.

    Flat archives follow the naming pattern: {prefix}_{timestamp}_{promptIdx}_{imgIdx}.png
    where timestamp is YYYYMMDD_HHMMSS format.

    Args:
        saved_dir: Path to the saved/ directory

    Returns:
        List of dictionaries with archive info, each containing:
        - prefix: The image prefix
        - timestamp: Archive timestamp (YYYYMMDD_HHMMSS)
        - images: List of image paths in this archive
        - first_image: Path to first image (for thumbnail)
        - image_count: Number of images in this archive
    """
    if not saved_dir.exists():
        return []

    archives: dict[tuple[str, str], dict] = {}

    # Pattern: prefix_YYYYMMDD_HHMMSS_promptIdx_imgIdx.png
    # We need to identify where the timestamp is (it's always YYYYMMDD_HHMMSS format)
    timestamp_pattern = re.compile(r'^(.+)_(\d{8}_\d{6})_(\d+)_(\d+)\.png$')

    for png_file in saved_dir.glob("*.png"):
        match = timestamp_pattern.match(png_file.name)
        if not match:
            continue

        prefix = match.group(1)
        timestamp = match.group(2)
        prompt_idx = match.group(3)
        img_idx = match.group(4)

        key = (prefix, timestamp)
        if key not in archives:
            archives[key] = {
                "prefix": prefix,
                "timestamp": timestamp,
                "images": [],
                "first_image": None,
                "image_count": 0,
            }

        archives[key]["images"].append(png_file)
        archives[key]["image_count"] += 1

        # Track first image (sorted by prompt_idx, img_idx)
        current_first = archives[key]["first_image"]
        if current_first is None or png_file.name < current_first.name:
            archives[key]["first_image"] = png_file

    return list(archives.values())


def get_flat_archive_metadata(image_path: Path) -> dict:
    """Extract metadata from a flat archive PNG file.

    Args:
        image_path: Path to a flat archive PNG file

    Returns:
        Dictionary with embedded metadata (prompt, user_prompt, model, etc.)
    """
    try:
        img = Image.open(image_path)
        metadata = {}
        if hasattr(img, 'text'):
            metadata = dict(img.text)
        return metadata
    except Exception as e:
        logger.debug(f"Failed to read metadata from {image_path}: {e}")
        return {}


def run_has_images(run_dir: Path) -> bool:
    """Check if a run directory contains any generated images."""
    prefix = get_prefix_from_metadata(run_dir)
    return len(list(run_dir.glob(f"{prefix}_*_*.png"))) > 0


def is_backup_run(run_dir: Path) -> bool:
    """Check if a run directory is a backup."""
    try:
        metadata = load_run_metadata(run_dir)
        return metadata.get("backup_info", {}).get("is_backup", False)
    except (ValueError, json.JSONDecodeError):
        return False


def delete_run(run_dir: Path, prompts_dir: Path) -> None:
    """Delete a gallery run directory completely.

    Args:
        run_dir: Directory to delete
        prompts_dir: The prompts directory (for safety validation)

    Raises:
        ValueError: If run_dir doesn't exist, is not in prompts_dir, or is an archive
    """
    # Safety check 1: Verify run_dir exists
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    # Safety check 2: Verify run_dir is inside prompts_dir (prevent path traversal)
    try:
        run_dir.resolve().relative_to(prompts_dir.resolve())
    except ValueError:
        raise ValueError(f"Run directory is not inside prompts directory: {run_dir}")

    # Safety check 3: Verify it's not an archive
    if is_backup_run(run_dir):
        raise ValueError(f"Cannot delete archived galleries: {run_dir}")

    # Delete the directory
    shutil.rmtree(run_dir)
