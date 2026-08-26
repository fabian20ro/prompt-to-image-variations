"""Tests for the fixed ERNIE-Image-Turbo generator."""

import sys
import re
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from image_generator import (
    GUIDANCE,
    INFERENCE_STEPS,
    MODEL_NAME,
    QUANTIZATION,
    _get_model,
    _model_cache,
    clear_model_cache,
    generate_image,
)


def test_fixed_model_contract():
    assert MODEL_NAME == "ernie-image-turbo"
    assert QUANTIZATION == 4
    assert INFERENCE_STEPS == 8
    assert GUIDANCE == 1.0


def test_clear_model_cache():
    _model_cache[False] = object()
    clear_model_cache()
    assert not _model_cache


def test_get_model_cache_hit():
    cached = MagicMock()
    _model_cache[True] = cached
    assert _get_model(tiled_vae=True) is cached
    clear_model_cache()


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_loads_local_ernie_q4(mock_settings, temp_dir):
    model_path = temp_dir / "ernie-q4"
    model_path.mkdir()
    mock_settings.image_generation.model_path = model_path

    config_module = ModuleType("mflux.models.common.config")
    config_module.ModelConfig = MagicMock()
    model_config = config_module.ModelConfig.ernie_image_turbo.return_value
    ernie_module = ModuleType("mflux.models.ernie_image")
    instance = MagicMock()
    ernie_module.ErnieImage = MagicMock(return_value=instance)

    with patch.dict(sys.modules, {
        "mflux.models.common.config": config_module,
        "mflux.models.ernie_image": ernie_module,
    }):
        result = _get_model()

    ernie_module.ErnieImage.assert_called_once_with(
        model_config=model_config,
        model_path=str(model_path),
        quantize=None,
    )
    assert result is instance


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_requires_provisioned_checkpoint(mock_settings, temp_dir):
    mock_settings.image_generation.model_path = temp_dir / "missing"
    with pytest.raises(FileNotFoundError, match="ERNIE q4 model not found"):
        _get_model()


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_applies_tiling_when_tiled_vae(mock_settings, temp_dir):
    model_path = temp_dir / "ernie-q4"
    model_path.mkdir()
    mock_settings.image_generation.model_path = model_path

    config_module = ModuleType("mflux.models.common.config")
    config_module.ModelConfig = MagicMock()
    ernie_module = ModuleType("mflux.models.ernie_image")
    instance = MagicMock()
    ernie_module.ErnieImage = MagicMock(return_value=instance)

    tiling_module = ModuleType("mflux.models.common.vae.tiling_config")
    TilingConfig = MagicMock()
    tiling_module.TilingConfig = TilingConfig

    with patch.dict(sys.modules, {
        "mflux.models.common.config": config_module,
        "mflux.models.ernie_image": ernie_module,
        "mflux.models.common.vae.tiling_config": tiling_module,
    }):
        _get_model(tiled_vae=True)

    assert TilingConfig.called
    assert instance.tiling_config is TilingConfig.return_value


@patch("image_generator.settings")
def test_get_model_import_error_when_mflux_missing(mock_settings, monkeypatch):
    """When mflux is not installed, _get_model surfaces an actionable ImportError."""
    model_path = MagicMock()
    model_path.expanduser.return_value = mock_settings.image_generation.model_path
    model_path.exists.return_value = True
    mock_settings.image_generation.model_path = model_path

    for mod in ("mflux", "mflux.models", "mflux.models.common",
                "mflux.models.common.config", "mflux.models.ernie_image"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError, match=r"mflux 0\.18\.0\+ is required"):
        _get_model()


@patch("image_generator.settings")
@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_uses_settings_defaults(mock_unload, mock_get_model, mock_settings, temp_dir):
    """When no explicit width/height given, generate_image must use settings defaults."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    # Configure the mock settings with non-default values.
    mock_settings.image_generation.default_width = 256
    mock_settings.image_generation.default_height = 256

    result = generate_image("test", temp_dir / "image.png")

    call_kwargs = mock_get_model.return_value.generate_image.call_args.kwargs
    assert call_kwargs["width"] == 256
    assert call_kwargs["height"] == 256


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_explicit_width_overrides_settings(mock_unload, mock_get_model, temp_dir):
    """Explicit width/height must override settings defaults."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    result = generate_image("test", temp_dir / "image.png", width=512, height=768)

    call_kwargs = mock_get_model.return_value.generate_image.call_args.kwargs
    assert call_kwargs["width"] == 512
    assert call_kwargs["height"] == 768



@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_uses_fixed_parameters(mock_unload, mock_get_model, temp_dir):
    """Explicit width/height parameters must be forwarded to the model."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model
    output = temp_dir / "image.png"

    result = generate_image("test prompt", output, seed=42, width=1024, height=768)

    assert result == output
    mock_unload.assert_called_once_with()
    model.generate_image.assert_called_once_with(
        seed=42,
        prompt="test prompt",
        num_inference_steps=8,
        guidance=1.0,
        height=768,
        width=1024,
    )
    generated.save.assert_called_once_with(str(output))


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_propagates_explicit_seed(mock_unload, mock_get_model, temp_dir):
    """An explicit seed must be forwarded as the seed kwarg to model.generate_image."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    output = temp_dir / "image.png"
    generate_image("test", output, seed=9876)

    assert (
        model.generate_image.call_args.kwargs["seed"] == 9876
    ), "explicit seed must be forwarded to the image generator call"


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_random_seed_and_tiling(mock_unload, mock_get_model, temp_dir):
    mock_get_model.return_value.generate_image.return_value = MagicMock()
    with patch("image_generator.random.randint", return_value=123) as randint:
        generate_image("test", temp_dir / "image.png", tiled_vae=True)
    mock_get_model.assert_called_once_with(True)
    mock_unload.assert_called_once_with()
    randint.assert_called_once()
    assert mock_get_model.return_value.generate_image.call_args.kwargs["seed"] == 123


@pytest.mark.parametrize(
    "width,height,error_msg",
    [
        (-1, 512, r"Width must be positive, got -1"),
        (512, 0, r"Height must be positive, got 0"),
        (511, 512, r"Width must be a multiple of 8, got 511"),
        (512, 513, r"Height must be a multiple of 8, got 513"),
    ],
)
def test_generate_image_rejects_invalid_dimensions(
    temp_dir, width, height, error_msg
):
    """Invalid dimensions must raise ValueError with the specific message."""
    with pytest.raises(ValueError, match=error_msg):
        generate_image("test", temp_dir / "image.png", width=width, height=height)


@pytest.mark.parametrize("width,height", [(64, 64), (256, 256), (864, 1152)])
@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_accepts_valid_dimensions(mock_unload, mock_get_model, temp_dir, width, height):
    """Valid dimensions must reach model.generate_image without raising."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    output = temp_dir / "image.png"
    result = generate_image("test", output, seed=1, width=width, height=height)

    assert result == output
    mock_unload.assert_called_once_with()


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_creates_output_parent_directory(mock_unload, mock_get_model, temp_dir):
    parent = temp_dir / "nested" / "deep"
    output = parent / "image.png"
    # Ensure the parent does not exist yet.
    assert not parent.exists()

    model = MagicMock()
    generated = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    generate_image("test", output)

    assert parent.is_dir()


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_propagates_file_not_found_when_model_missing(
    mock_unload, mock_get_model, temp_dir
):
    """When the model is missing, generate_image must surface FileNotFoundError
    without leaving dangling state or generating images."""
    mock_get_model.side_effect = FileNotFoundError(
        "ERNIE q4 model not found: /fake/path"
    )

    with pytest.raises(FileNotFoundError, match="ERNIE q4 model not found"):
        generate_image("test", temp_dir / "image.png")

    mock_unload.assert_called_once_with()
    assert mock_get_model.return_value.generate_image.call_count == 0


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_skips_tiling_by_default(mock_settings, temp_dir):
    """When tiled_vae is False (default), tiling config must not be applied."""
    model_path = temp_dir / "ernie-q4"
    model_path.mkdir()
    mock_settings.image_generation.model_path = model_path

    config_module = ModuleType("mflux.models.common.config")
    config_module.ModelConfig = MagicMock()
    ernie_module = ModuleType("mflux.models.ernie_image")
    instance = MagicMock()
    ernie_module.ErnieImage = MagicMock(return_value=instance)

    tiling_module = ModuleType("mflux.models.common.vae.tiling_config")
    TilingConfig = MagicMock()
    tiling_module.TilingConfig = TilingConfig

    with patch.dict(sys.modules, {
        "mflux.models.common.config": config_module,
        "mflux.models.ernie_image": ernie_module,
        "mflux.models.common.vae.tiling_config": tiling_module,
    }):
        _get_model(tiled_vae=False)

    assert not TilingConfig.called, (
        "TilingConfig must not be instantiated when tiled_vae=False"
    )


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_forwards_empty_prompt(mock_unload, mock_get_model, temp_dir):
    """An empty prompt must be forwarded unchanged to model.generate_image."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    generate_image("", temp_dir / "image.png", seed=1)

    assert (
        model.generate_image.call_args.kwargs["prompt"] == ""
    ), "empty prompt must be forwarded unchanged"


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_tiled_vae_not_in_generation_kwargs(mock_unload, mock_get_model, temp_dir):
    """tiled_vae affects model loading only; it must not appear in model.generate_image kwargs."""
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model

    generate_image("test", temp_dir / "image.png", seed=1, tiled_vae=True)

    assert (
        "tiled_vae" not in model.generate_image.call_args.kwargs
    ), "tiled_vae must only influence _get_model, never the generation call"
