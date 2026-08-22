#!/usr/bin/env python3
"""Worker subprocess for executing generation tasks.

This script is spawned by the main server to handle heavy operations
in isolation. It reads task configuration from a JSON file, executes
the appropriate operation, and writes progress to stdout as JSON lines.

Usage:
    python worker_subprocess.py <task.json>
"""

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

from config import paths
from pipeline import PipelineExecutor, PipelineResult
from grammar_generator import hash_prompt
from utils import delete_run
from gallery_index import generate_master_index
from metadata_manager import MetadataManager, MetadataError, MetadataNotFoundError

# Global log file handle (set when we know the output directory)
_log_file = None


def set_log_file(log_path: Path) -> None:
    """Set the log file path for this task."""
    global _log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, 'a', buffering=1)  # Line-buffered
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Task started at {datetime.now().isoformat()}\n")
    _log_file.write(f"{'='*60}\n")


def close_log_file() -> None:
    """Close the log file."""
    global _log_file
    if _log_file:
        _log_file.write(f"\n{'='*60}\n")
        _log_file.write(f"Task ended at {datetime.now().isoformat()}\n")
        _log_file.write(f"{'='*60}\n")
        _log_file.close()
        _log_file = None


def log_to_file(message: str) -> None:
    """Write a message to the log file if set."""
    if _log_file:
        timestamp = datetime.now().strftime("%H:%M:%S")
        _log_file.write(f"[{timestamp}] {message}\n")


def emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
    """Emit progress update to stdout."""
    data = {
        "type": "progress",
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
    }
    print(json.dumps(data), flush=True)
    if message:
        log_to_file(message)


def emit_result(success: bool, data: dict | None = None, error: str | None = None):
    """Emit final result to stdout."""
    result = {
        "type": "result",
        "success": success,
        "data": data,
        "error": error,
    }
    print(json.dumps(result), flush=True)
    if success:
        log_to_file(f"Task completed successfully: {data}")
    else:
        log_to_file(f"Task failed: {error}")


def emit_image_ready(run_id: str, path: str):
    """Emit notification that an image is ready."""
    data = {
        "type": "image_ready",
        "run_id": run_id,
        "path": path,
    }
    print(json.dumps(data), flush=True)


class Heartbeat:
    """Context manager for emitting periodic heartbeats during long operations."""

    def __init__(self, message: str = "Working...", interval: int = 30):
        """Initialize heartbeat.

        Args:
            message: Message to emit with heartbeat
            interval: Seconds between heartbeats
        """
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def _heartbeat_loop(self):
        """Emit heartbeat every interval seconds until stopped."""
        while not self._stop_event.wait(timeout=self.interval):
            try:
                emit_progress("heartbeat", 0, 0, self.message)
            except Exception as exc:
                log_to_file(
                    "Heartbeat stopped after progress emission failed: "
                    f"{type(exc).__name__}"
                )
                self._stop_event.set()
                break

    def __enter__(self):
        """Start the heartbeat thread."""
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        return False


def create_executor() -> PipelineExecutor:
    """Create a PipelineExecutor with JSON progress callbacks."""
    return PipelineExecutor(
        on_progress=emit_progress,
        on_image_ready=emit_image_ready,
    )


def run_generate_pipeline(params: dict):
    """Run full generation pipeline: LLM -> Tracery -> Images."""
    prompt = params["prompt"]
    count = params.get("count", 50)
    prefix = params.get("prefix", "image")
    temperature = params.get("temperature", 0.7)
    no_cache = params.get("no_cache", False)
    generate_images = params.get("generate_images", False)
    images_per_prompt = params.get("images_per_prompt", 1)
    width = params.get("width", 864)
    height = params.get("height", 1152)
    seed = params.get("seed")
    max_prompts = params.get("max_prompts")
    tiled_vae = params.get("tiled_vae", False)
    enhance = params.get("enhance", False)
    enhance_softness = params.get("enhance_softness", 0.5)
    enhance_after = params.get("enhance_after", False)

    # Precompute output directory so logs include full pipeline execution.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = paths.prompts_dir / f"{run_timestamp}_{hash_prompt(prompt)}"
    set_log_file(run_output_dir / f"{prefix}_worker.log")
    log_to_file(f"Starting generation pipeline for: {prompt[:100]}...")

    executor = create_executor()

    with Heartbeat(f"Generating pipeline for: {prompt[:30]}..."):
        result = executor.run_full_pipeline(
            prompt=prompt,
            count=count,
            prefix=prefix,
            temperature=temperature,
            no_cache=no_cache,
            generate_images=generate_images,
            images_per_prompt=images_per_prompt,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            tiled_vae=tiled_vae,
            enhance=enhance,
            enhance_softness=enhance_softness,
            enhance_after=enhance_after,
            output_dir=run_output_dir,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "prompt_count": result.prompt_count,
            "output_dir": str(result.output_dir),
        })
    else:
        emit_result(False, error=result.error)


def run_generate_from_grammar(params: dict):
    """Create a new gallery directly from pasted Tracery grammar."""
    grammar = params["grammar"]
    title = params.get("title") or "Grammar import"
    prefix = params.get("prefix", "image")
    count = params.get("count", 50)
    images_per_prompt = params.get("images_per_prompt", 1)
    max_prompts = params.get("max_prompts")
    width = params.get("width", 864)
    height = params.get("height", 1152)
    seed = params.get("seed")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = paths.prompts_dir / f"{run_timestamp}_{hash_prompt(grammar)}"
    set_log_file(run_output_dir / f"{prefix}_worker.log")
    log_to_file(f"Starting grammar import for: {title[:100]}")

    executor = create_executor()

    with Heartbeat(f"Creating gallery from grammar: {title[:30]}..."):
        result = executor.run_from_grammar_text(
            grammar=grammar,
            count=count,
            prefix=prefix,
            images_per_prompt=images_per_prompt,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            output_dir=run_output_dir,
            user_prompt=title,
            display_title=title,
            generate_images=False,
            source="from_grammar_text",
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "prompt_count": result.prompt_count,
            "output_dir": str(result.output_dir),
        })
    else:
        emit_result(False, error=result.error)


def run_regenerate_prompts(params: dict):
    """Regenerate prompts from edited grammar."""
    run_id = params["run_id"]
    grammar = params["grammar"]
    count = params.get("count")
    images_per_prompt = params.get("images_per_prompt")
    max_prompts = params.get("max_prompts")

    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    # Load metadata for prefix using MetadataManager for proper error handling
    try:
        metadata = MetadataManager.load(output_dir)
        prefix = metadata.prefix
        set_log_file(output_dir / f"{prefix}_worker.log")
        log_to_file(f"Regenerating prompts: count={count}")
    except MetadataError as e:
        log_to_file(f"Warning: Could not load metadata: {e}")
        prefix = "image"
        set_log_file(output_dir / f"{prefix}_worker.log")

    executor = create_executor()

    with Heartbeat(f"Regenerating prompts for: {run_id}..."):
        result = executor.regenerate_prompts(
            run_id=run_id,
            grammar=grammar,
            count=count,
            images_per_prompt=images_per_prompt,
            max_prompts=max_prompts,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "prompt_count": result.prompt_count,
            "task_type": "regenerate_prompts",
        })
    else:
        emit_result(False, error=result.error)


def run_generate_image(params: dict):
    """Generate a single image."""
    run_id = params["run_id"]
    prompt_idx = params["prompt_idx"]
    image_idx = params.get("image_idx", 0)

    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    # Load metadata for logging using MetadataManager for proper error handling
    try:
        metadata = MetadataManager.load(output_dir)
        prefix = metadata.prefix
        set_log_file(output_dir / f"{prefix}_worker.log")
        log_to_file(f"Generating single image: prompt_idx={prompt_idx}, image_idx={image_idx}")
    except MetadataError as e:
        log_to_file(f"Warning: Could not load metadata: {e}")
        prefix = "image"
        set_log_file(output_dir / f"{prefix}_worker.log")

    executor = create_executor()

    with Heartbeat(f"Generating image {prompt_idx}_{image_idx}..."):
        result = executor.generate_single_image(
            run_id=run_id,
            prompt_idx=prompt_idx,
            image_idx=image_idx,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "image_path": result.data.get("image_path"),
        })
    else:
        emit_result(False, error=result.error)


def run_enhance_image(params: dict):
    """Enhance a single image."""
    run_id = params["run_id"]
    prompt_idx = params["prompt_idx"]
    image_idx = params.get("image_idx", 0)
    softness = params.get("softness", 0.5)

    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    # Load metadata for logging using MetadataManager for proper error handling
    try:
        metadata = MetadataManager.load(output_dir)
        prefix = metadata.prefix
        set_log_file(output_dir / f"{prefix}_worker.log")
        log_to_file(f"Enhancing single image: prompt_idx={prompt_idx}, image_idx={image_idx}")
    except MetadataError as e:
        log_to_file(f"Warning: Could not load metadata: {e}")
        prefix = "image"
        set_log_file(output_dir / f"{prefix}_worker.log")

    executor = create_executor()

    with Heartbeat(f"Enhancing image {prompt_idx}_{image_idx}..."):
        result = executor.enhance_single_image(
            run_id=run_id,
            prompt_idx=prompt_idx,
            image_idx=image_idx,
            softness=softness,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "image_path": result.data.get("image_path"),
        })
    else:
        emit_result(False, error=result.error)


def run_generate_all_images(params: dict):
    """Generate all images for a gallery."""
    run_id = params["run_id"]
    images_per_prompt = params.get("images_per_prompt", 1)
    resume = params.get("resume", True)

    # Optional image settings (passed from gallery form)
    width = params.get("width")
    height = params.get("height")
    seed = params.get("seed")
    max_prompts = params.get("max_prompts")
    enhance = params.get("enhance", False)
    enhance_softness = params.get("enhance_softness", 0.5)

    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    # Load metadata for logging using MetadataManager for proper error handling
    try:
        metadata = MetadataManager.load(output_dir)
        prefix = metadata.prefix
        set_log_file(output_dir / f"{prefix}_worker.log")
        log_to_file(f"Generating all images: images_per_prompt={images_per_prompt}, resume={resume}, "
                    f"width={width}, height={height}, enhance={enhance}")
    except MetadataError as e:
        log_to_file(f"Warning: Could not load metadata: {e}")
        prefix = "image"
        set_log_file(output_dir / f"{prefix}_worker.log")

    executor = create_executor()

    with Heartbeat(f"Generating all images for: {run_id}..."):
        result = executor.generate_all_images(
            run_id=run_id,
            images_per_prompt=images_per_prompt,
            resume=resume,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            enhance=enhance,
            enhance_softness=enhance_softness,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "generated": result.image_count,
            "skipped": result.skipped_count,
        })
    else:
        emit_result(False, error=result.error)


def run_enhance_all_images(params: dict):
    """Enhance all images for a gallery."""
    run_id = params["run_id"]
    softness = params.get("softness", 0.5)

    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    # Load metadata for logging using MetadataManager for proper error handling
    try:
        metadata = MetadataManager.load(output_dir)
        prefix = metadata.prefix
        set_log_file(output_dir / f"{prefix}_worker.log")
        log_to_file(f"Enhancing all images: softness={softness}")
    except MetadataError as e:
        log_to_file(f"Warning: Could not load metadata: {e}")
        prefix = "image"
        set_log_file(output_dir / f"{prefix}_worker.log")

    executor = create_executor()

    with Heartbeat(f"Enhancing all images for: {run_id}..."):
        result = executor.enhance_all_images(
            run_id=run_id,
            softness=softness,
        )

    if result.success:
        emit_result(True, data={
            "run_id": result.run_id,
            "enhanced": result.image_count,
        })
    else:
        emit_result(False, error=result.error)


def run_delete_gallery(params: dict):
    """Delete a gallery directory."""
    run_id = params["run_id"]
    output_dir = paths.prompts_dir / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Gallery not found: {run_id}")
        return

    with Heartbeat(f"Deleting gallery: {run_id}"):
        try:
            # Delete the directory
            delete_run(output_dir, paths.prompts_dir)
        except ValueError as e:
            # delete_run raises ValueError for validation errors (not in prompts_dir, is archive, etc.)
            emit_result(False, error=str(e))
            sys.exit(1)
        except OSError as e:
            emit_result(False, error=f"Failed to delete gallery: {e}")
            sys.exit(1)

    # Regenerate master index - best effort, don't fail if this fails
    try:
        generate_master_index(paths.generated_dir)
    except Exception as e:
        log_to_file(f"Warning: Failed to regenerate master index: {e}")

    emit_result(True, data={"run_id": run_id, "deleted": True})


TASK_HANDLERS = {
    "generate_pipeline": run_generate_pipeline,
    "generate_from_grammar": run_generate_from_grammar,
    "regenerate_prompts": run_regenerate_prompts,
    "generate_image": run_generate_image,
    "enhance_image": run_enhance_image,
    "generate_all_images": run_generate_all_images,
    "enhance_all_images": run_enhance_all_images,
    "delete_gallery": run_delete_gallery,
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_subprocess.py <task.json>", file=sys.stderr)
        sys.exit(1)

    task_file = Path(sys.argv[1])
    if not task_file.exists():
        print(f"Task file not found: {task_file}", file=sys.stderr)
        sys.exit(1)

    try:
        task = json.loads(task_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in task file: {e}", file=sys.stderr)
        sys.exit(1)

    task_type = task.get("type")
    params = task.get("params", {})

    if task_type is None:
        emit_result(False, error="Missing 'type' field in task JSON")
        sys.exit(1)

    handler = TASK_HANDLERS.get(task_type)
    if handler is None:
        emit_result(False, error=f"Unknown task type: {task_type}")
        sys.exit(1)

    try:
        handler(params)
    except Exception as e:
        import traceback
        log_to_file(f"Unhandled exception: {traceback.format_exc()}")
        emit_result(False, error=str(e))
        sys.exit(1)
    finally:
        close_log_file()


if __name__ == "__main__":
    main()
