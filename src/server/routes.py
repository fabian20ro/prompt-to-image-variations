"""API routes for the web server."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from .app import get_queue_manager, get_worker, get_shutdown_event
from .models import (
    GenerateRequest,
    GenerateFromGrammarRequest,
    RegeneratePromptsApiRequest,
    GrammarUpdateRequest,
    GenerateImageRequest,
    EnhanceImageRequest,
    GenerateAllImagesRequest,
    EnhanceAllImagesRequest,
    GalleryLayoutUpdateRequest,
    StatusResponse,
    TaskResponse,
    TaskType,
    GalleryInfo,
    GalleryDetailResponse,
)

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import paths
from grammar_history import append_grammar_revision, load_grammar_history
from gallery_index import generate_master_index, _extract_run_info
from metadata_manager import MetadataManager
from utils import backup_run, is_backup_run
from services.gallery_service import GalleryService, GalleryNotFoundError, MetadataNotFoundError


from functools import lru_cache


@lru_cache(maxsize=1)
def get_gallery_service() -> GalleryService:
    """FastAPI dependency to get the GalleryService singleton.

    Uses lru_cache for singleton behavior while being compatible
    with FastAPI's Depends() mechanism.
    """
    return GalleryService(paths.prompts_dir, paths.saved_dir)


router = APIRouter()


# ----------------------------------------------------------------------------
# Global Endpoints
# ----------------------------------------------------------------------------

@router.get("/index", response_class=HTMLResponse)
async def get_index():
    """Serve the master index page with generation form."""
    # Regenerate index with interactive mode for the web UI
    index_path = generate_master_index(paths.generated_dir, interactive=True)

    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(content="<h1>No galleries yet</h1>")


@router.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current queue status."""
    qm = get_queue_manager()
    state = qm.get_state()

    return StatusResponse(
        queue_length=len(state.pending) + (1 if state.current_task else 0),
        current_task=state.current_task,
        pending_count=len(state.pending),
        completed_count=len(state.completed),
    )


@router.get("/api/events")
async def sse_events(request: Request):
    """SSE endpoint for real-time updates."""
    queue = asyncio.Queue(maxsize=100)
    qm = get_queue_manager()

    def on_event(event: str, data: dict):
        try:
            queue.put_nowait({"event": event, "data": data})
        except asyncio.QueueFull:
            logging.warning(f"SSE queue full, dropped event: {event}")

    # Register listener BEFORE getting initial state to avoid race condition
    qm.add_listener(on_event)

    def serialize(obj):
        """Serialize object to JSON, handling datetime and Pydantic models."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(mode='json')
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        return obj

    async def event_generator() -> AsyncGenerator:
        try:
            shutdown = get_shutdown_event()
        except RuntimeError:
            shutdown = None

        try:
            # Send initial status AFTER registering listener (atomic snapshot)
            state = qm.get_state()
            yield {
                "event": "status",
                "data": json.dumps({
                    "pending_count": len(state.pending),
                    "current": state.current_task.model_dump(mode='json') if state.current_task else None,
                }),
            }

            while True:
                # Check if server is shutting down
                if shutdown and shutdown.is_set():
                    break

                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=5)  # Shorter timeout for faster shutdown
                    yield {
                        "event": msg["event"],
                        "data": json.dumps(serialize(msg["data"])),
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}

        finally:
            qm.remove_listener(on_event)

    return EventSourceResponse(event_generator())


@router.post("/api/queue/clear", response_model=TaskResponse)
async def clear_queue():
    """Clear all pending tasks."""
    qm = get_queue_manager()
    count = qm.clear_pending()
    return TaskResponse(task_id="", message=f"Cleared {count} pending tasks")


@router.post("/api/worker/kill", response_model=TaskResponse)
async def kill_worker():
    """Kill the currently running task."""
    worker = get_worker()
    killed = await worker.kill_current()

    if killed:
        return TaskResponse(task_id="", message="Killed current task")
    else:
        return TaskResponse(task_id="", message="No task running")


# ----------------------------------------------------------------------------
# Generation Endpoints
# ----------------------------------------------------------------------------

@router.post("/api/generate", response_model=TaskResponse)
async def start_generation(req: GenerateRequest):
    """Start a new generation pipeline."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_PIPELINE,
        req.model_dump(),
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Generation queued (task {task.id[:8]})",
    )


@router.post("/api/generate-from-grammar", response_model=TaskResponse)
async def start_generation_from_grammar(req: GenerateFromGrammarRequest):
    """Create a new gallery directly from pasted Tracery grammar."""
    try:
        json.loads(req.grammar)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON grammar")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_FROM_GRAMMAR,
        req.model_dump(),
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Grammar gallery queued (task {task.id[:8]})",
    )


# ----------------------------------------------------------------------------
# Gallery Endpoints
# ----------------------------------------------------------------------------

@router.get("/gallery/{run_id}", response_class=HTMLResponse)
async def get_gallery(run_id: str):
    """Serve a gallery page with interactive features."""
    from gallery import generate_gallery_for_directory

    prompts_dir = paths.prompts_dir
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Regenerate gallery with interactive mode
    try:
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        return HTMLResponse(content=gallery_path.read_text())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/saved/{filename}")
async def get_saved_file(filename: str):
    """Serve a flat archived image from saved/ directory."""
    saved_dir = paths.saved_dir
    file_path = saved_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure file is within the saved directory
    try:
        file_path.resolve().relative_to(saved_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Only serve PNG files from saved/
    if not filename.endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are served")

    return FileResponse(file_path, media_type="image/png")


@router.get("/archive/{run_id}", response_class=HTMLResponse)
async def get_archive_gallery(run_id: str):
    """Serve an archived gallery page (read-only)."""
    from gallery import generate_gallery_for_directory

    saved_dir = paths.saved_dir
    run_dir = saved_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Archive not found")

    # Regenerate gallery - archives are read-only (interactive=False)
    try:
        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)
        return HTMLResponse(content=gallery_path.read_text())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/archive/{run_id}/{filename:path}")
async def get_archive_file(run_id: str, filename: str):
    """Serve static files (images, etc.) from an archive directory."""
    saved_dir = paths.saved_dir
    run_dir = saved_dir / run_id
    file_path = run_dir / filename

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Archive not found")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure file is within the run directory
    try:
        file_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type
    media_type, _ = mimetypes.guess_type(str(file_path))
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type)


@router.get("/gallery/{run_id}/{filename:path}")
async def get_gallery_file(run_id: str, filename: str):
    """Serve static files (images, etc.) from a gallery directory."""
    prompts_dir = paths.prompts_dir
    run_dir = prompts_dir / run_id
    file_path = run_dir / filename

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure file is within the run directory
    try:
        file_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type
    media_type, _ = mimetypes.guess_type(str(file_path))
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type)


@router.get("/api/gallery/{run_id}")
async def get_gallery_info(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
) -> GalleryDetailResponse:
    """Get gallery details including grammar and prompts."""

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    info = _extract_run_info(run_dir)
    if not info:
        raise HTTPException(status_code=404, detail="Invalid gallery")

    prefix = info["prefix"]

    # Load grammar and prompts using service
    grammar = service.load_grammar(run_dir, prefix)
    prompts = service.load_prompts(run_dir, prefix)

    # List images using service
    images = []
    image_files = service.list_images(run_dir, prefix)
    for img_path in image_files:
        # Parse prompt_idx and image_idx from filename: prefix_promptIdx_imageIdx.png
        parts = img_path.stem.split('_')
        if len(parts) >= 3:
            try:
                prompt_idx = int(parts[-2])
                image_idx = int(parts[-1])
                prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else ""
                images.append({
                    "prompt_idx": prompt_idx,
                    "image_idx": image_idx,
                    "filename": img_path.name,
                    "prompt": prompt_text,
                })
            except (ValueError, IndexError):
                pass

    return GalleryDetailResponse(
        info=GalleryInfo(
            run_id=run_id,
            prefix=prefix,
            user_prompt=info["user_prompt"],
            prompt_count=info["prompt_count"],
            image_count=info["image_count"],
            created_at=info["display_time"],
            model=info["model"],
            gallery_path=info["gallery_path"],
            thumbnail=info["thumbnail"],
        ),
        grammar=grammar,
        prompts=prompts,
        images=images,
    )


@router.get("/api/gallery/{run_id}/grammar")
async def get_grammar(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
):
    """Get the grammar JSON for a gallery."""

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    prefix = service.get_prefix(run_dir)
    grammar = service.load_grammar(run_dir, prefix)
    if grammar is None:
        raise HTTPException(status_code=404, detail="Grammar not found")

    return {"grammar": grammar}


@router.get("/api/gallery/{run_id}/grammar/history")
async def get_grammar_history(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
):
    """Get persisted grammar revisions for a gallery."""
    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    prefix = service.get_prefix(run_dir)
    grammar = service.load_grammar(run_dir, prefix) or ""
    history = load_grammar_history(run_dir, prefix, current_grammar=grammar)
    return {"history": history}


@router.put("/api/gallery/{run_id}/grammar", response_model=TaskResponse)
async def update_grammar(
    run_id: str,
    req: GrammarUpdateRequest,
    service: GalleryService = Depends(get_gallery_service),
):
    """Update the grammar for a gallery."""

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Validate JSON
    try:
        json.loads(req.grammar)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON grammar")

    prefix = service.get_prefix(run_dir)
    grammar_file = service.get_grammar_file(run_dir, prefix)
    append_grammar_revision(run_dir, prefix, req.grammar, action="save")
    grammar_file.write_text(req.grammar)

    return TaskResponse(task_id="", message="Grammar updated")


@router.put("/api/gallery/{run_id}/layout", response_model=TaskResponse)
async def update_gallery_layout(
    run_id: str,
    req: GalleryLayoutUpdateRequest,
    service: GalleryService = Depends(get_gallery_service),
):
    """Persist gallery layout settings for a run."""
    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    MetadataManager.update(
        run_dir,
        gallery_layout={
            "images_per_prompt": req.images_per_prompt,
            "max_prompts": req.max_prompts,
        },
    )

    return TaskResponse(task_id="", message="Layout updated")


@router.post("/api/gallery/{run_id}/regenerate", response_model=TaskResponse)
async def regenerate_prompts(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
    req: RegeneratePromptsApiRequest | None = None,
):
    """Regenerate prompts from the current grammar."""

    try:
        run_dir = service.get_run_directory(run_id)
        metadata = service.load_metadata(run_dir)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")
    except MetadataNotFoundError:
        raise HTTPException(status_code=404, detail="No metadata found")

    prefix = service.get_prefix(run_dir)
    grammar = req.grammar if req and req.grammar is not None else service.load_grammar(run_dir, prefix)
    if grammar is None:
        raise HTTPException(status_code=404, detail="Grammar not found")

    try:
        json.loads(grammar)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON grammar")

    prefix = service.get_prefix(run_dir)
    append_grammar_revision(run_dir, prefix, grammar, action="save")
    service.get_grammar_file(run_dir, prefix).write_text(grammar)

    count = req.count if req and req.count else metadata.get("count", 50)

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.REGENERATE_PROMPTS,
        {
            "run_id": run_id,
            "grammar": grammar,
            "count": count,
            "images_per_prompt": req.images_per_prompt if req else None,
            "max_prompts": req.max_prompts if req else None,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Prompt regeneration queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/generate-all", response_model=TaskResponse)
async def generate_all_images(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
    req: GenerateAllImagesRequest | None = None,
):
    """Queue generation of all images for a gallery."""

    try:
        service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Build params dict with all settings
    params = {
        "run_id": run_id,
        "images_per_prompt": req.images_per_prompt if req else 1,
        "resume": req.resume if req else True,
    }

    # Add optional image settings (only if provided)
    if req:
        if req.width is not None:
            params["width"] = req.width
        if req.height is not None:
            params["height"] = req.height
        if req.seed is not None:
            params["seed"] = req.seed
        if req.max_prompts is not None:
            params["max_prompts"] = req.max_prompts
        params["enhance"] = req.enhance
        params["enhance_softness"] = req.enhance_softness

    qm = get_queue_manager()
    task = qm.add_task(TaskType.GENERATE_ALL_IMAGES, params)

    return TaskResponse(
        task_id=task.id,
        message=f"Image generation queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/enhance-all", response_model=TaskResponse)
async def enhance_all_images(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
    req: EnhanceAllImagesRequest | None = None,
):
    """Queue enhancement of all images for a gallery."""

    try:
        service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.ENHANCE_ALL_IMAGES,
        {
            "run_id": run_id,
            "softness": req.softness if req else 0.5,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Enhancement queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/image/{prompt_idx}/generate", response_model=TaskResponse)
async def generate_single_image(
    run_id: str,
    prompt_idx: int,
    service: GalleryService = Depends(get_gallery_service),
    req: GenerateImageRequest | None = None,
):
    """Queue generation of a single image."""

    try:
        service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_IMAGE,
        {
            "run_id": run_id,
            "prompt_idx": prompt_idx,
            "image_idx": req.image_idx if req else 0,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Image generation queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/image/{prompt_idx}/enhance", response_model=TaskResponse)
async def enhance_single_image(
    run_id: str,
    prompt_idx: int,
    service: GalleryService = Depends(get_gallery_service),
    req: EnhanceImageRequest | None = None,
):
    """Queue enhancement of a single image."""

    try:
        service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.ENHANCE_IMAGE,
        {
            "run_id": run_id,
            "prompt_idx": prompt_idx,
            "image_idx": req.image_idx if req else 0,
            "softness": req.softness if req else 0.5,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Enhancement queued (task {task.id[:8]})",
    )


@router.get("/api/gallery/{run_id}/logs")
async def get_gallery_logs(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
    tail: int = 100,
):
    """Get the worker log file for a gallery.

    Args:
        run_id: The gallery run ID
        tail: Number of lines from the end to return (default 100, 0 for all)

    Returns:
        Log file contents
    """

    if tail < 0:
        raise HTTPException(status_code=400, detail="tail must be non-negative")

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Find the log file
    log_files = list(run_dir.glob("*_worker.log"))
    if not log_files:
        return {"logs": "", "message": "No log file found"}

    log_file = log_files[0]
    content = log_file.read_text()

    if tail > 0:
        lines = content.split('\n')
        content = '\n'.join(lines[-tail:])

    return {"logs": content, "filename": log_file.name}


@router.post("/api/gallery/{run_id}/archive", response_model=TaskResponse)
async def archive_gallery(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
):
    """Archive a gallery to the saved folder as flat files with embedded metadata."""

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    if service.is_backup_run(run_dir):
        raise HTTPException(status_code=400, detail="Cannot archive a backup")

    try:
        saved_dir = paths.saved_dir
        saved_files = backup_run(run_dir, saved_dir, reason="manual_archive")
        generate_master_index(paths.generated_dir, interactive=True)
        return TaskResponse(
            task_id="",
            message=f"Archived {len(saved_files)} images to saved/"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/gallery/{run_id}", response_model=TaskResponse)
async def delete_gallery(
    run_id: str,
    service: GalleryService = Depends(get_gallery_service),
):
    """Delete a gallery and all its contents.

    Only active galleries in prompts/ can be deleted.
    Archives in saved/ are protected and cannot be deleted.
    """

    try:
        run_dir = service.get_run_directory(run_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Validate it's not an archive (extra safety check)
    if service.is_backup_run(run_dir):
        raise HTTPException(status_code=400, detail="Cannot delete archived galleries")

    # Queue the delete task
    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.DELETE_GALLERY,
        {"run_id": run_id},
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Gallery deletion queued (task {task.id[:8]})",
    )
