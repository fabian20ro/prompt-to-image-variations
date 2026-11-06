"""Tests for Pydantic models and input validation."""

import pytest

from server.models import (
    TaskType,
    TaskStatus,
    Task,
    QueueState,
    TaskProgress,
    GenerateRequest,
    GenerateFromGrammarRequest,
    EnhanceImageRequest,
    GalleryLayoutUpdateRequest,
)


class TestModels:
    """Tests for Pydantic models."""

    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.GENERATE_PIPELINE.value == "generate_pipeline"
        assert TaskType.GENERATE_FROM_GRAMMAR.value == "generate_from_grammar"
        assert TaskType.REGENERATE_PROMPTS.value == "regenerate_prompts"
        assert TaskType.GENERATE_IMAGE.value == "generate_image"
        assert TaskType.ENHANCE_IMAGE.value == "enhance_image"
        assert TaskType.GENERATE_ALL_IMAGES.value == "generate_all_images"
        assert TaskType.ENHANCE_ALL_IMAGES.value == "enhance_all_images"
        assert TaskType.DELETE_GALLERY.value == "delete_gallery"

    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_task_creation(self):
        """Test Task model creation with defaults."""
        task = Task(
            id="test-id",
            type=TaskType.GENERATE_PIPELINE,
        )
        assert task.id == "test-id"
        assert task.type == TaskType.GENERATE_PIPELINE
        assert task.status == TaskStatus.PENDING
        assert task.pid is None
        assert task.params == {}

    def test_task_progress(self):
        """Test TaskProgress model."""
        progress = TaskProgress(
            stage="generating_images",
            current=5,
            total=10,
            message="Generating image_5.png",
        )
        assert progress.stage == "generating_images"
        assert progress.current == 5
        assert progress.total == 10
        assert progress.message == "Generating image_5.png"

    def test_queue_state_empty(self):
        """Test empty QueueState."""
        state = QueueState()
        assert state.version == 1
        assert state.current_task is None
        assert state.pending == []
        assert state.completed == []

    def test_generate_request_defaults(self):
        """Test GenerateRequest with default values."""
        req = GenerateRequest(prompt="a dragon")
        assert req.prompt == "a dragon"
        assert req.count == 50
        assert req.prefix == "image"
        assert req.temperature == 0.7
        assert req.no_cache is False
        assert req.generate_images is False
        assert req.width == 864
        assert req.height == 1152
        assert req.enhance_tiled_vae is False
        assert req.enhance is False

    def test_generate_request_custom(self):
        """Test GenerateRequest with custom values."""
        req = GenerateRequest(
            prompt="a cat",
            count=10,
            prefix="cat",
            generate_images=True,
            enhance=True,
            enhance_softness=0.3,
        )
        assert req.prompt == "a cat"
        assert req.count == 10
        assert req.prefix == "cat"
        assert req.generate_images is True
        assert req.enhance is True
        assert req.enhance_softness == 0.3

    @pytest.mark.parametrize("legacy_field", ["model", "steps", "quantize", "lora_paths"])
    def test_generate_request_rejects_removed_model_controls(self, legacy_field):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="a cat", **{legacy_field: 4})

    def test_generate_from_grammar_request_defaults(self):
        req = GenerateFromGrammarRequest(grammar='{"origin": ["test"]}')
        assert req.prefix == "image"
        assert req.count == 50
        assert req.images_per_prompt == 1

    def test_generate_from_grammar_request_allows_zero_images_per_prompt(self):
        req = GenerateFromGrammarRequest(grammar='{"origin": ["test"]}', images_per_prompt=0)
        assert req.images_per_prompt == 0


class TestInputValidation:
    """Tests for Pydantic model input validation."""

    def test_generate_request_prompt_required(self):
        """Test that prompt is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest()  # Missing required prompt

    def test_generate_request_prompt_not_empty(self):
        """Test that prompt cannot be empty."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_generate_request_count_bounds(self):
        """Test that count must be within bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", count=0)

        # Too high
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", count=10001)

        # Just right
        req = GenerateRequest(prompt="test", count=100)
        assert req.count == 100

    def test_generate_request_dimensions_bounds(self):
        """Test that width/height must be within bounds."""
        from pydantic import ValidationError

        # Too small
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=32)

        # Too large
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", height=5000)

        # Just right
        req = GenerateRequest(prompt="test", width=512, height=768)
        assert req.width == 512
        assert req.height == 768

    def test_generate_request_temperature_bounds(self):
        """Test that temperature must be within bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=-0.1)

        # Too high
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=2.5)

        # Just right
        req = GenerateRequest(prompt="test", temperature=1.5)
        assert req.temperature == 1.5

    def test_gallery_layout_update_request_allows_zero_images_per_prompt(self):
        req = GalleryLayoutUpdateRequest(images_per_prompt=0)
        assert req.images_per_prompt == 0

    def test_generate_request_prefix_pattern(self):
        """Test that prefix must match allowed pattern."""
        from pydantic import ValidationError

        # Valid prefixes
        GenerateRequest(prompt="test", prefix="image")
        GenerateRequest(prompt="test", prefix="my-prefix")
        GenerateRequest(prompt="test", prefix="prefix_123")

        # Invalid prefixes
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", prefix="prefix with spaces")

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", prefix="prefix.with.dots")

    def test_enhance_softness_bounds(self):
        """Test that softness must be within 0-1."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            EnhanceImageRequest(softness=-0.1)

        # Too high
        with pytest.raises(ValidationError):
            EnhanceImageRequest(softness=1.5)

        # Just right
        req = EnhanceImageRequest(softness=0.7)
        assert req.softness == 0.7
