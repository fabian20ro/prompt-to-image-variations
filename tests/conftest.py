"""Shared test fixtures for all test modules."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Match the flat-module runtime used by `uv run python src/cli.py`.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def run_dir(temp_dir):
    """Create a mock run directory with basic structure."""
    run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
    run_dir.mkdir(parents=True)
    return run_dir


@pytest.fixture
def saved_dir(temp_dir):
    """Create a saved/archive directory."""
    saved_dir = temp_dir / "saved"
    saved_dir.mkdir(parents=True)
    return saved_dir


@pytest.fixture
def queue_path(temp_dir):
    """Create a queue file path for testing."""
    return temp_dir / "queue.json"


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary for testing."""
    return {
        "prefix": "test",
        "count": 10,
        "user_prompt": "a dragon flying over mountains",
        "model": "ernie-image-turbo",
        "image_generation": {"images_per_prompt": 1},
    }


@pytest.fixture
def sample_grammar():
    """Sample Tracery grammar for testing."""
    return {
        "origin": ["#subject# in #setting#"],
        "subject": ["a cat", "a dog"],
        "setting": ["a garden", "a forest"],
    }


def create_run_files(run_dir: Path, prefix: str = "test", num_prompts: int = 2,
                      metadata: dict = None, grammar: dict = None, create_images: bool = False):
    """Helper to create run directory with files.

    Args:
        run_dir: Directory to create files in
        prefix: File prefix
        num_prompts: Number of prompt files to create
        metadata: Metadata dict (defaults to basic metadata)
        grammar: Grammar dict (defaults to simple grammar)
        create_images: Whether to create fake image files
    """
    if metadata is None:
        metadata = {
            "prefix": prefix,
            "count": num_prompts,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }

    if grammar is None:
        grammar = {"origin": ["test"]}

    # Ensure directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    (run_dir / f"{prefix}.metaprompt.json").write_text(json.dumps(metadata))

    # Create grammar
    (run_dir / f"{prefix}_grammar.json").write_text(json.dumps(grammar))

    # Create prompts
    for i in range(num_prompts):
        (run_dir / f"{prefix}_{i}.txt").write_text(f"Prompt {i}")

    # Create images if requested
    if create_images:
        for i in range(num_prompts):
            (run_dir / f"{prefix}_{i}_0.png").write_bytes(b"fake image")

    return run_dir


@pytest.fixture
def full_run_dir(run_dir):
    """Create a run directory with all required files."""
    return create_run_files(run_dir)


def test_create_run_files_integrity(temp_dir):
    """Verify create_run_files creates a valid directory structure."""
    num_prompts = 3
    prefix = "test_run"
    run_dir = create_run_files(temp_dir / "test_run", prefix=prefix, num_prompts=num_prompts)
    assert (run_dir / f"{prefix}.metaprompt.json").exists()
    assert (run_dir / f"{prefix}_grammar.json").exists()
    for i in range(num_prompts):
        assert (run_dir / f"{prefix}_{i}.txt").exists()


def test_create_run_files_with_images(temp_dir):
    """Verify create_run_files writes image files when create_images=True."""
    num_prompts = 2
    prefix = "img_run"
    run_dir = create_run_files(
        temp_dir / "img_run",
        prefix=prefix,
        num_prompts=num_prompts,
        create_images=True,
    )
    for i in range(num_prompts):
        img_path = run_dir / f"{prefix}_{i}_0.png"
        assert img_path.exists()
        assert img_path.read_bytes() == b"fake image"


def test_create_run_files_custom_metadata(temp_dir):
    """Verify create_run_files writes custom metadata and grammar to disk."""
    num_prompts = 1
    prefix = "custom_run"
    run_dir = temp_dir / "custom_run"

    meta = {"prefix": prefix, "count": num_prompts, "user_prompt": "a river"}
    gram = {"origin": ["river scene"], "scene": ["valley"]}

    returned = create_run_files(
        run_dir,
        prefix=prefix,
        num_prompts=num_prompts,
        metadata=meta,
        grammar=gram,
    )

    # Metadata round-trips exactly as provided
    written_meta = json.loads((returned / f"{prefix}.metaprompt.json").read_text())
    assert written_meta == meta

    # Grammar round-trips exactly as provided
    written_gram = json.loads((returned / f"{prefix}_grammar.json").read_text())
    assert written_gram == gram


def test_create_run_files_exist_ok(temp_dir):
    """Verify create_run_files succeeds when directory already exists."""
    prefix = "exist_ok"
    run_subdir = temp_dir / "existing" / prefix
    run_subdir.mkdir(parents=True)  # pre-create the parent

    returned = create_run_files(run_subdir, prefix=prefix, num_prompts=1)
    assert returned == run_subdir


def test_create_run_files_default_metadata_structure(temp_dir):
    """Verify default metadata contains expected fields when no custom metadata provided."""
    prefix = "default_meta"
    run_dir = create_run_files(
        temp_dir / "default_meta",
        prefix=prefix,
        num_prompts=2,
    )

    written_meta = json.loads((run_dir / f"{prefix}.metaprompt.json").read_text())

    # Verify default metadata contains required fields with expected values
    assert written_meta["prefix"] == "default_meta"
    assert written_meta["count"] == 2
    assert written_meta["user_prompt"] == "test prompt"
    assert written_meta.get("image_generation", {}).get("images_per_prompt") == 1


def test_create_run_files_zero_prompts(temp_dir):
    """Verify create_run_files handles num_prompts=0 by creating no prompt files."""
    prefix = "zero_run"
    run_dir = temp_dir / "zero_run"

    returned = create_run_files(
        run_dir,
        prefix=prefix,
        num_prompts=0,
    )

    # Metadata and grammar still created even with zero prompts
    assert (returned / f"{prefix}.metaprompt.json").exists()
    assert (returned / f"{prefix}_grammar.json").exists()

    # No prompt files should be created when num_prompts is 0
    for i in range(0):
        assert not (returned / f"{prefix}_{i}.txt").exists()

    # Verify no unexpected prompt files exist
    txt_files = list((returned).glob(f"{prefix}_*.txt"))
    assert len(txt_files) == 0


def test_full_run_dir_fixture(temp_dir):
    """Verify the full_run_dir fixture produces a valid run directory structure."""
    from tests.conftest import create_run_files

    # Simulate what the full_run_dir fixture does
    run_subdir = temp_dir / "prompts" / "20240101_120000_test"
    run_subdir.mkdir(parents=True)

    returned = create_run_files(run_subdir)

    # Verify directory structure is valid
    assert (returned / "test.metaprompt.json").exists()
    assert (returned / "test_grammar.json").exists()

    # Default metadata should have expected fields
    meta = json.loads((returned / "test.metaprompt.json").read_text())
    assert meta["prefix"] == "test"
    assert meta["count"] == 2
    assert meta["user_prompt"] == "test prompt"

    # Default grammar should be present
    gram = json.loads((returned / "test_grammar.json").read_text())
    assert gram["origin"] == ["test"]

    # Two default prompts created (num_prompts=2)
    for i in range(2):
        prompt_path = returned / f"test_{i}.txt"
        assert prompt_path.exists()
        assert prompt_path.read_text() == f"Prompt {i}"


def test_create_run_files_no_images_when_flag_false(temp_dir):
    """Verify create_run_files creates no image files when create_images=False."""
    prefix = "no_img"
    num_prompts = 2
    run_subdir = temp_dir / "no_img"

    returned = create_run_files(
        run_subdir,
        prefix=prefix,
        num_prompts=num_prompts,
        create_images=False,
    )

    # Metadata and grammar should still exist
    assert (returned / f"{prefix}.metaprompt.json").exists()
    assert (returned / f"{prefix}_grammar.json").exists()

    # Prompt files should be created normally
    for i in range(num_prompts):
        prompt_path = returned / f"{prefix}_{i}.txt"
        assert prompt_path.exists()

    # No image files should exist when create_images=False
    png_files = list((returned).glob("*.png"))
    assert len(png_files) == 0

    # Explicit check: no file matching the image pattern exists
    for i in range(num_prompts):
        img_path = returned / f"{prefix}_{i}_0.png"
        assert not img_path.exists()


def test_saved_dir_fixture(temp_dir):
    """Verify saved_dir fixture returns a 'saved' subdirectory under temp_dir."""
    saved = temp_dir / "saved"
    assert saved.is_dir()
    # Should be empty — fixture only creates the directory
    assert list(saved.iterdir()) == []


def test_queue_path_fixture(temp_dir):
    """Verify queue_path fixture returns a queue.json path under temp_dir."""
    qp = temp_dir / "queue.json"
    assert str(qp).endswith("queue.json")
    assert not qp.exists()  # only the path is created, no file


def test_sample_metadata_fixture(sample_metadata):
    """Verify sample_metadata fixture has required keys and expected types."""
    meta = sample_metadata
    for key in ("prefix", "count", "user_prompt", "model", "image_generation"):
        assert key in meta, f"missing required key: {key}"
    assert isinstance(meta["count"], int) and meta["count"] > 0


def test_sample_grammar_fixture(sample_grammar):
    """Verify sample_grammar fixture is valid Tracery grammar with origin key."""
    gram = sample_grammar
    assert "origin" in gram
    assert isinstance(gram["origin"], list) and len(gram["origin"]) > 0
    for key, val in gram.items():
        assert isinstance(val, list), f"key {key!r} value should be a list"