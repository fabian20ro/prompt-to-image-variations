"""Pydantic models for the web server API."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    """Types of tasks that can be queued."""
    GENERATE_PIPELINE = "generate_pipeline"
    GENERATE_FROM_GRAMMAR = "generate_from_grammar"
    REGENERATE_PROMPTS = "regenerate_prompts"
    GENERATE_IMAGE = "generate_image"
    ENHANCE_IMAGE = "enhance_image"
    GENERATE_ALL_IMAGES = "generate_all_images"
    ENHANCE_ALL_IMAGES = "enhance_all_images"
    DELETE_GALLERY = "delete_gallery"


class TaskStatus(str, Enum):
    """Status of a queued task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskProgress(BaseModel):
    """Progress information for a running task."""
    stage: str = ""
    current: int = 0
    total: int = 0
    message: str = ""


class Task(BaseModel):
    """A task in the queue."""
    id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    pid: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    progress: TaskProgress = Field(default_factory=TaskProgress)
    result: dict[str, Any] | None = None
    error: str | None = None


class QueueState(BaseModel):
    """State of the task queue persisted to disk."""
    version: int = 1
    current_task: Task | None = None
    pending: list[Task] = Field(default_factory=list)
    completed: list[Task] = Field(default_factory=list)


# API Request Models

class GenerateRequest(BaseModel):
    """Request to start a new generation pipeline."""
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=5000)
    count: int = Field(50, ge=1, le=10000)
    prefix: str = Field("image", min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    no_cache: bool = False
    generate_images: bool = False
    images_per_prompt: int = Field(1, ge=1, le=100)
    width: int = Field(864, ge=64, le=4096)
    height: int = Field(1152, ge=64, le=4096)
    seed: int | None = Field(None, ge=0)
    max_prompts: int | None = Field(None, ge=1)
    tiled_vae: bool = False
    enhance: bool = False
    enhance_softness: float = Field(0.5, ge=0.0, le=1.0)
    enhance_after: bool = False


class RegeneratePromptsRequest(BaseModel):
    """Request to regenerate prompts from edited grammar (internal queue task)."""
    run_id: str
    grammar: str
    count: int | None = None


class RegeneratePromptsApiRequest(BaseModel):
    """API request to regenerate prompts (run_id and grammar from URL/file)."""
    model_config = ConfigDict(extra="forbid")

    grammar: str | None = None
    count: int | None = Field(None, ge=1, le=10000)
    images_per_prompt: int | None = Field(None, ge=0, le=100)
    max_prompts: int | None = Field(None, ge=1)


class GrammarUpdateRequest(BaseModel):
    """Request to update a gallery's grammar."""
    model_config = ConfigDict(extra="forbid")

    grammar: str


class GenerateImageRequest(BaseModel):
    """Request to generate a specific image."""
    image_idx: int = Field(0, ge=0, le=1000)


class EnhanceImageRequest(BaseModel):
    """Request to enhance a specific image."""
    model_config = ConfigDict(extra="forbid")

    image_idx: int = Field(0, ge=0, le=1000)
    softness: float = Field(0.5, ge=0.0, le=1.0)


class GenerateAllImagesRequest(BaseModel):
    """Request to generate all images for a gallery."""
    model_config = ConfigDict(extra="forbid")

    images_per_prompt: int = Field(1, ge=0, le=100)
    resume: bool = True
    # Image generation settings (override metadata defaults)
    width: int | None = Field(None, ge=64, le=4096)
    height: int | None = Field(None, ge=64, le=4096)
    seed: int | None = Field(None, ge=0)
    max_prompts: int | None = Field(None, ge=1)
    # Enhancement settings
    enhance: bool = False
    enhance_softness: float = Field(0.5, ge=0.0, le=1.0)


class GalleryLayoutUpdateRequest(BaseModel):
    """Request to persist gallery layout state."""
    images_per_prompt: int = Field(1, ge=0, le=100)
    max_prompts: int | None = Field(None, ge=1)


class GenerateFromGrammarRequest(BaseModel):
    """Request to create a run directly from pasted Tracery grammar."""
    model_config = ConfigDict(extra="forbid")

    grammar: str = Field(..., min_length=2)
    title: str | None = Field(None, max_length=500)
    count: int = Field(50, ge=1, le=10000)
    prefix: str = Field("image", min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    images_per_prompt: int = Field(1, ge=0, le=100)
    max_prompts: int | None = Field(None, ge=1)
    width: int = Field(864, ge=64, le=4096)
    height: int = Field(1152, ge=64, le=4096)
    seed: int | None = Field(None, ge=0)


class EnhanceAllImagesRequest(BaseModel):
    """Request to enhance all images for a gallery."""
    model_config = ConfigDict(extra="forbid")

    softness: float = Field(0.5, ge=0.0, le=1.0)


# API Response Models

class StatusResponse(BaseModel):
    """Response for queue status endpoint."""
    queue_length: int
    current_task: Task | None
    pending_count: int
    completed_count: int


class TaskResponse(BaseModel):
    """Response after creating a task."""
    task_id: str
    message: str


class GalleryInfo(BaseModel):
    """Information about a gallery run."""
    run_id: str
    prefix: str
    user_prompt: str
    prompt_count: int
    image_count: int
    created_at: str
    model: str | None
    gallery_path: str
    thumbnail: str | None


class GalleryDetailResponse(BaseModel):
    """Detailed gallery information including grammar."""
    info: GalleryInfo
    grammar: str | None
    prompts: list[str]
    images: list[dict[str, Any]]
