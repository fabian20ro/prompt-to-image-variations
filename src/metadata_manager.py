"""Centralized metadata management for run directories.

This module provides a unified interface for loading, saving, and updating
metadata files in run directories, replacing the duplicated pattern across
the codebase.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetadataError(Exception):
    """Raised when metadata operations fail."""
    pass


class MetadataNotFoundError(MetadataError):
    """Raised when no metadata file is found in a directory."""
    pass


@dataclass
class RunMetadata:
    """Structured representation of run metadata."""

    prefix: str = "image"
    count: int = 0
    user_prompt: str = ""
    model: str = "ernie-image-turbo"
    created_at: str = ""
    grammar_cached: bool = False
    image_generation: dict = field(default_factory=dict)
    source: str | None = None
    grammar_path: str | None = None
    regenerated_at: str | None = None
    gallery_layout: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.count < 0:
            raise ValueError("count must be non-negative")

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        """Create RunMetadata from a dictionary."""
        return cls(
            prefix=data.get("prefix", "image"),
            count=data.get("count", 0),
            user_prompt=data.get("user_prompt", ""),
            model=data.get("model", "ernie-image-turbo"),
            created_at=data.get("created_at", ""),
            grammar_cached=data.get("grammar_cached", False),
            image_generation=data.get("image_generation", {}),
            source=data.get("source"),
            grammar_path=data.get("grammar_path"),
            regenerated_at=data.get("regenerated_at"),
            gallery_layout=data.get("gallery_layout", {}),
            _raw=data,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "prefix": self.prefix,
            "count": self.count,
            "user_prompt": self.user_prompt,
            "model": self.model,
            "created_at": self.created_at,
            "grammar_cached": self.grammar_cached,
        }

        if self.image_generation:
            result["image_generation"] = self.image_generation
        if self.source:
            result["source"] = self.source
        if self.grammar_path:
            result["grammar_path"] = self.grammar_path
        if self.regenerated_at:
            result["regenerated_at"] = self.regenerated_at
        if self.gallery_layout:
            result["gallery_layout"] = self.gallery_layout

        # Preserve any extra fields from raw data
        for key, value in self._raw.items():
            if key not in result:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from raw data, for backwards compatibility."""
        return self._raw.get(key, default)


def resolve_gallery_layout(metadata: RunMetadata | dict, prompt_count: int | None = None) -> dict:
    """Resolve persisted gallery layout with fallbacks for older runs."""
    if isinstance(metadata, RunMetadata):
        gallery_layout = metadata.gallery_layout or {}
        image_generation = metadata.image_generation or {}
    else:
        gallery_layout = metadata.get("gallery_layout", {}) or {}
        image_generation = metadata.get("image_generation", {}) or {}

    images_per_prompt = gallery_layout.get("images_per_prompt")
    if images_per_prompt is None:
        images_per_prompt = image_generation.get("images_per_prompt", 1)

    max_prompts = gallery_layout.get("max_prompts")
    if max_prompts is None and "max_prompts" not in gallery_layout:
        max_prompts = image_generation.get("max_prompts")

    if images_per_prompt is None:
        images_per_prompt = 1
    images_per_prompt = max(0, int(images_per_prompt))
    if max_prompts is not None:
        max_prompts = max(1, int(max_prompts))
        if prompt_count is not None:
            max_prompts = min(max_prompts, prompt_count)

    return {
        "images_per_prompt": images_per_prompt,
        "max_prompts": max_prompts,
    }


class MetadataManager:
    """Manager for run directory metadata operations."""

    @staticmethod
    def find_metadata_file(run_dir: Path) -> Path | None:
        """Find the metadata file in a run directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            Path to the metadata file, or None if not found
        """
        # Search for both the new .metaprompt.json and old _metadata.json patterns
        for pattern in ["*.metaprompt.json", "*_metadata.json"]:
            meta_files = list(run_dir.glob(pattern))
            if meta_files:
                return meta_files[0]
        return None

    @classmethod
    def load(cls, run_dir: Path) -> RunMetadata:
        """Load metadata from a run directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            RunMetadata object

        Raises:
            MetadataNotFoundError: If no metadata file found
            MetadataError: If metadata file cannot be read or parsed
        """
        meta_file = cls.find_metadata_file(run_dir)
        if not meta_file:
            raise MetadataNotFoundError(f"No metadata file found in {run_dir}")

        try:
            data = json.loads(meta_file.read_text())
            return RunMetadata.from_dict(data)
        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON in metadata file: {e}")
        except OSError as e:
            raise MetadataError(f"Failed to read metadata file: {e}")

    @classmethod
    def load_raw(cls, run_dir: Path) -> dict:
        """Load raw metadata dictionary from a run directory.

        This is provided for backwards compatibility with code that
        expects a plain dictionary.

        Args:
            run_dir: Path to the run directory

        Returns:
            Metadata dictionary

        Raises:
            MetadataNotFoundError: If no metadata file found
            MetadataError: If metadata file cannot be read or parsed
        """
        meta_file = cls.find_metadata_file(run_dir)
        if not meta_file:
            raise MetadataNotFoundError(f"No metadata file found in {run_dir}")

        try:
            return json.loads(meta_file.read_text())
        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON in metadata file: {e}")
        except OSError as e:
            raise MetadataError(f"Failed to read metadata file: {e}")

    @classmethod
    def save(cls, run_dir: Path, metadata: RunMetadata | dict, prefix: str | None = None) -> Path:
        """Save metadata to a run directory.

        Args:
            run_dir: Path to the run directory
            metadata: Metadata to save (RunMetadata or dict)
            prefix: File prefix (defaults to metadata prefix or "image")

        Returns:
            Path to the saved metadata file

        Raises:
            MetadataError: If metadata cannot be saved
        """
        if isinstance(metadata, RunMetadata):
            data = metadata.to_dict()
            prefix = prefix or metadata.prefix
        else:
            data = metadata
            prefix = prefix or data.get("prefix", "image")

        meta_file = run_dir / f"{prefix}.metaprompt.json"

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=f".{meta_file.name}.", dir=str(run_dir))
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, meta_file)
            except BaseException:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
                raise
            return meta_file
        except OSError as e:
            raise MetadataError(f"Failed to save metadata file: {e}")

    @classmethod
    def update(cls, run_dir: Path, **updates) -> RunMetadata:
        """Update specific fields in metadata.

        Args:
            run_dir: Path to the run directory
            **updates: Fields to update

        Returns:
            Updated RunMetadata object

        Raises:
            MetadataNotFoundError: If no metadata file found
            MetadataError: If update fails
        """
        # Load existing metadata
        meta_file = cls.find_metadata_file(run_dir)
        if not meta_file:
            raise MetadataNotFoundError(f"No metadata file found in {run_dir}")

        try:
            data = json.loads(meta_file.read_text())

            # Apply updates
            for key, value in updates.items():
                if value is not None:
                    existing = data.get(key)
                    if isinstance(existing, dict) and isinstance(value, dict):
                        merged = {**existing, **value}
                        data[key] = merged
                    else:
                        data[key] = value
                elif key in data:
                    del data[key]

            # Save back atomically (tempfile + os.replace) — matches save() pattern
            fd, tmp_path = tempfile.mkstemp(prefix=f".{meta_file.name}.", dir=str(run_dir))
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, meta_file)
            except BaseException:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
                raise
            return RunMetadata.from_dict(data)

        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON in metadata file: {e}")
        except OSError as e:
            raise MetadataError(f"Failed to update metadata file: {e}")

    @classmethod
    def exists(cls, run_dir: Path) -> bool:
        """Check if metadata exists in a run directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            True if metadata file exists
        """
        return cls.find_metadata_file(run_dir) is not None

    @classmethod
    def get_prefix(cls, run_dir: Path, default: str = "image") -> str:
        """Get the prefix from metadata, with a default fallback.

        Args:
            run_dir: Path to the run directory
            default: Default prefix if not found

        Returns:
            The prefix string
        """
        try:
            metadata = cls.load(run_dir)
            return metadata.prefix
        except MetadataError:
            return default

    @classmethod
    def get_image_settings(cls, run_dir: Path) -> dict:
        """Get image generation settings from metadata.

        Args:
            run_dir: Path to the run directory

        Returns:
            Image generation settings dictionary, or empty dict if not found
        """
        try:
            metadata = cls.load(run_dir)
            return metadata.image_generation
        except MetadataError:
            return {}

    @classmethod
    def get_gallery_layout(cls, run_dir: Path, prompt_count: int | None = None) -> dict:
        """Get gallery layout settings with fallbacks for older runs."""
        try:
            metadata = cls.load(run_dir)
            return resolve_gallery_layout(metadata, prompt_count=prompt_count)
        except MetadataError:
            return resolve_gallery_layout({}, prompt_count=prompt_count)


# Convenience functions for backwards compatibility
def load_metadata(run_dir: Path) -> dict:
    """Load metadata from a run directory (convenience function).

    Args:
        run_dir: Path to the run directory

    Returns:
        Metadata dictionary, or empty dict if not found
    """
    try:
        return MetadataManager.load_raw(run_dir)
    except MetadataError:
        return {}


def save_metadata(run_dir: Path, metadata: dict, prefix: str = "image") -> Path | None:
    """Save metadata to a run directory (convenience function).

    Args:
        run_dir: Path to the run directory
        metadata: Metadata dictionary to save
        prefix: File prefix

    Returns:
        Path to saved file, or None if save failed
    """
    try:
        return MetadataManager.save(run_dir, metadata, prefix)
    except MetadataError as e:
        logger.error(f"Failed to save metadata: {e}")
        return None


def get_metadata_prefix(run_dir: Path, default: str = "image") -> str:
    """Get the prefix from metadata (convenience function).

    Args:
        run_dir: Path to the run directory
        default: Default prefix if not found

    Returns:
        The prefix string
    """
    return MetadataManager.get_prefix(run_dir, default)
