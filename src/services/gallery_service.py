"""Gallery service for managing gallery operations.

Consolidates repeated patterns from routes.py for better maintainability.
"""

import json
from pathlib import Path


class GalleryNotFoundError(Exception):
    """Gallery does not exist."""
    pass


class MetadataNotFoundError(Exception):
    """Metadata file not found in gallery."""
    pass


class GalleryService:
    """Service for gallery operations."""

    def __init__(self, prompts_dir: Path, saved_dir: Path):
        """Initialize the service.

        Args:
            prompts_dir: Path to prompts/ directory
            saved_dir: Path to saved/ directory
        """
        self.prompts_dir = prompts_dir
        self.saved_dir = saved_dir

    def get_run_directory(self, run_id: str, is_archive: bool = False) -> Path:
        """Get and validate run directory.

        Args:
            run_id: The run directory name
            is_archive: Whether to look in saved/ instead of prompts/

        Returns:
            Path to the run directory

        Raises:
            GalleryNotFoundError: If directory doesn't exist
        """
        base_dir = self.saved_dir if is_archive else self.prompts_dir
        run_dir = base_dir / run_id

        # Reject paths that traverse outside the gallery directory (e.g. ".." or symlinks).
        try:
            file_path = run_dir.resolve()
            file_path.relative_to(base_dir.resolve())
        except ValueError:
            location = "Archive" if is_archive else "Gallery"
            raise GalleryNotFoundError(f"{location} not found: {run_id}")

        if not run_dir.exists():
            location = "Archive" if is_archive else "Gallery"
            raise GalleryNotFoundError(f"{location} not found: {run_id}")

        return run_dir

    def load_metadata(self, run_dir: Path) -> dict:
        """Load metadata from a run directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            Metadata dictionary

        Raises:
            MetadataNotFoundError: If no metadata file found
        """
        meta_file = self.get_metadata_file(run_dir)
        return json.loads(meta_file.read_text())

    def get_metadata_file(self, run_dir: Path) -> Path:
        """Get the metadata file path from a run directory.

        Args:
            run_dir: Path to the run directory

        Returns:
            Path to the metadata file

        Raises:
            MetadataNotFoundError: If no metadata file found
        """
        for pattern in ["*.metaprompt.json", "*_metadata.json"]:
            meta_files = list(run_dir.glob(pattern))
            if meta_files:
                return meta_files[0]
        raise MetadataNotFoundError(f"No metadata file found in {run_dir}")

    def get_prefix(self, run_dir: Path) -> str:
        """Get prefix from metadata.

        Args:
            run_dir: Path to the run directory

        Returns:
            The prefix string (defaults to "image" if not found)
        """
        try:
            metadata = self.load_metadata(run_dir)
            return metadata.get("prefix", "image")
        except MetadataNotFoundError:
            return "image"

    def get_grammar_file(self, run_dir: Path, prefix: str | None = None) -> Path:
        """Get the grammar file path.

        Args:
            run_dir: Path to the run directory
            prefix: Optional prefix (auto-detected if not provided)

        Returns:
            Path to the grammar file (may not exist)
        """
        if prefix is None:
            prefix = self.get_prefix(run_dir)

        return run_dir / f"{prefix}_grammar.json"

    def load_grammar(self, run_dir: Path, prefix: str | None = None) -> str | None:
        """Load grammar content from a run directory.

        Args:
            run_dir: Path to the run directory
            prefix: Optional prefix (auto-detected if not provided)

        Returns:
            Grammar JSON string or None if not found
        """
        grammar_file = self.get_grammar_file(run_dir, prefix)
        if grammar_file.exists():
            return grammar_file.read_text()
        return None

    def load_prompts(self, run_dir: Path, prefix: str | None = None) -> list[str]:
        """Load all prompt texts from a run directory.

        Args:
            run_dir: Path to the run directory
            prefix: Optional prefix (auto-detected if not provided)

        Returns:
            List of prompt strings in order
        """
        if prefix is None:
            prefix = self.get_prefix(run_dir)

        prompt_files = sorted(run_dir.glob(f"{prefix}_*.txt"))
        # Filter to only prompt files (prefix_N.txt), not other files
        prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

        return [f.read_text() for f in prompt_files]

    def validate_file_access(self, file_path: Path, base_dir: Path) -> bool:
        """Validate file path for security (prevents path traversal).

        Args:
            file_path: Path to validate
            base_dir: Base directory the file should be within

        Returns:
            True if access is allowed

        Raises:
            ValueError: If file is outside base directory
        """
        try:
            file_path.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            raise ValueError(f"Access denied: {file_path} is outside {base_dir}")

    def is_backup_run(self, run_dir: Path) -> bool:
        """Check if a run directory is a backup.

        Args:
            run_dir: Path to the run directory

        Returns:
            True if this is a backup directory
        """
        try:
            metadata = self.load_metadata(run_dir)
            return metadata.get("backup_info", {}).get("is_backup", False)
        except MetadataNotFoundError:
            return False

    def count_images(self, run_dir: Path, prefix: str | None = None) -> int:
        """Count images in a run directory.

        Args:
            run_dir: Path to the run directory
            prefix: Optional prefix (auto-detected if not provided)

        Returns:
            Number of image files
        """
        if prefix is None:
            prefix = self.get_prefix(run_dir)

        return len(list(run_dir.glob(f"{prefix}_*_*.png")))

    def list_images(self, run_dir: Path, prefix: str | None = None) -> list[Path]:
        """List all image files in a run directory.

        Args:
            run_dir: Path to the run directory
            prefix: Optional prefix (auto-detected if not provided)

        Returns:
            Sorted list of image paths
        """
        if prefix is None:
            prefix = self.get_prefix(run_dir)

        return sorted(run_dir.glob(f"{prefix}_*_*.png"))

    def get_run_summary(self, run_id: str) -> dict:
        """Get a summary of a gallery run.

        Args:
            run_id: The run directory name

        Returns:
            Dictionary with prefix, image_count, prompt_count, and is_backup keys

        Raises:
            GalleryNotFoundError: If directory doesn't exist
        """
        run_dir = self.get_run_directory(run_id)
        return {
            "prefix": self.get_prefix(run_dir),
            "image_count": self.count_images(run_dir),
            "prompt_count": len(self.load_prompts(run_dir)),
            "is_backup": self.is_backup_run(run_dir),
        }
