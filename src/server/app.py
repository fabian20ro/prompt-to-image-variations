"""FastAPI application for the web UI."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import paths

from .queue_manager import QueueManager
from .worker import Worker


logger = logging.getLogger("prompt-to-image-variations")


# Global instances
queue_manager: QueueManager | None = None
worker: Worker | None = None
shutdown_event: asyncio.Event | None = None


def get_queue_manager() -> QueueManager:
    """Get the queue manager instance."""
    global queue_manager
    if queue_manager is None:
        raise RuntimeError("Queue manager not initialized")
    return queue_manager


def get_worker() -> Worker:
    """Get the worker instance."""
    global worker
    if worker is None:
        raise RuntimeError("Worker not initialized")
    return worker


def get_shutdown_event() -> asyncio.Event:
    """Get the shutdown event."""
    global shutdown_event
    if shutdown_event is None:
        raise RuntimeError("Shutdown event not initialized")
    return shutdown_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown with structured logging."""
    global queue_manager, worker, shutdown_event

    logger.info("Starting Image Prompt Generator application")

    # Ensure generated directory exists
    paths.generated_dir.mkdir(parents=True, exist_ok=True)
    paths.prompts_dir.mkdir(exist_ok=True)
    paths.grammars_dir.mkdir(exist_ok=True)

    # Initialize shutdown event
    shutdown_event = asyncio.Event()

    # Initialize queue manager
    queue_manager = QueueManager(paths.queue_path)

    # Initialize and start worker
    worker = Worker(queue_manager, paths.generated_dir)
    worker_task = asyncio.create_task(worker.run())

    def _log_worker_failure(done: asyncio.Task) -> None:
        """Report an unexpected worker crash at error level."""
        if done.cancelled():
            return
        exc = done.exception()
        if exc is not None:
            logger.error("Worker task crashed: %s", exc, exc_info=exc)

    worker_task.add_done_callback(_log_worker_failure)

    logger.info("Application startup complete")

    yield

    logger.info("Shutting down Image Prompt Generator application")

    # Shutdown: signal SSE connections to close
    shutdown_event.set()
    await asyncio.sleep(0.5)  # Grace period for SSE connections to close

    # Stop worker
    worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Image Prompt Generator",
        description="Web UI for generating image prompts",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Import and include routes
    from .routes import router
    app.include_router(router)

    # Mount static files for generated content.
    # Keep this after API routes to avoid route shadowing.
    app.mount(
        "/generated",
        StaticFiles(directory=str(paths.generated_dir), check_dir=False),
        name="generated",
    )

    # Redirect root to index
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/index")

    return app


# Create the app instance
app = create_app()
