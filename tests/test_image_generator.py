"""Renderer input failures; full lifecycle exercised at the migration seam."""

from dataclasses import replace
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config import settings
from image_generator import generate_image


@pytest.fixture
def renderer(monkeypatch, tmp_path):
    """Fake only the external mflux, MLX and LM Studio boundaries."""
    from image_generator import clear_model_cache
    from PIL import Image
    checkpoint = tmp_path / "model"
    for name in ("transformer", "text_encoder", "vae"):
        directory = checkpoint / name
        directory.mkdir(parents=True)
        (directory / "weights.safetensors").write_bytes(b"fake weights")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"quantization_level": "4"},
            "weight_map": {"weight": "weights.safetensors"},
        }))
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (checkpoint / "text_encoder" / name).write_text("{}")
    monkeypatch.setattr(settings, "image_generation", replace(settings.image_generation,
                       model_path=checkpoint, default_width=256, default_height=256))
    calls = []
    class Flux:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def generate_image(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(save=lambda **save: Image.new("RGB", (kwargs["width"], kwargs["height"])).save(save["path"]))
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(clear_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "mflux.models.common.config", SimpleNamespace(
        ModelConfig=SimpleNamespace(flux3_image_klein=lambda: "distilled")))
    monkeypatch.setitem(sys.modules, "mflux.models.flux3.variants", SimpleNamespace(Flux3Klein=Flux))
    monkeypatch.setattr("shutil.which", lambda _: "/fake/lms")
    process = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("subprocess.run", process)
    clear_model_cache()
    yield calls, process
    clear_model_cache()


def test_generate_image_requires_provisioned_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "image_generation", replace(
        settings.image_generation, model_path=tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="FLUX 3 q4 model not found"):
        generate_image("Photograph of a cup", tmp_path / "image.png")


@pytest.mark.parametrize("width,height", [(-1, 512), (512, 0), (511, 512), (512, 513)])
def test_generate_image_rejects_invalid_dimensions(tmp_path, width, height):
    with pytest.raises(ValueError):
        generate_image("Photograph of a cup", tmp_path / "image.png", width=width, height=height)


def test_generate_image_rejects_negative_seed_before_inference(tmp_path):
    with pytest.raises(ValueError, match="Seed must be non-negative"):
        generate_image("Cup", tmp_path / "image.png", seed=-5)


@pytest.mark.parametrize("width,height", [(64, 64), (256, 256), (864, 1152)])
def test_generate_image_accepts_valid_dimensions(renderer, tmp_path, width, height):
    from PIL import Image
    output = generate_image("Cup", tmp_path / "image.png", seed=1, width=width, height=height)
    assert Image.open(output).size == (width, height)


def test_generate_image_uses_settings_defaults_and_creates_parent(renderer, tmp_path):
    from PIL import Image
    output = generate_image("Cup", tmp_path / "nested" / "deep" / "image.png")
    assert Image.open(output).size == (256, 256)


def test_generate_image_uses_fixed_sampling_and_explicit_seed(renderer, tmp_path):
    calls, process = renderer
    output = generate_image("Cup", tmp_path / "image.png", seed=9876, width=512, height=768)
    assert calls == [{"prompt": "Cup", "seed": 9876, "num_inference_steps": 16, "width": 512, "height": 768}]
    assert json.loads(output.with_suffix(".generation.json").read_text())["seed"] == 9876
    assert process.call_args.args[0][-2:] == ["unload", "--all"]


def test_generate_image_random_seed_uses_full_uint32_range(renderer, tmp_path, monkeypatch):
    random = Mock(return_value=123)
    monkeypatch.setattr("image_generator.random.randint", random)
    output = generate_image("", tmp_path / "image.png")
    random.assert_called_once_with(0, 2**32 - 1)
    assert json.loads(output.with_suffix(".generation.json").read_text())["seed"] == 123
    assert renderer[0][0]["prompt"] == ""
