"""Renderer input failures; full lifecycle exercised at the migration seam."""

from dataclasses import replace

import pytest

from config import settings
from image_generator import generate_image


def test_generate_image_requires_provisioned_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "image_generation", replace(
        settings.image_generation, model_path=tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="FLUX 3 q4 model not found"):
        generate_image("Photograph of a cup", tmp_path / "image.png")


@pytest.mark.parametrize("width,height", [(-1, 512), (512, 0), (511, 512), (512, 513)])
def test_generate_image_rejects_invalid_dimensions(tmp_path, width, height):
    with pytest.raises(ValueError):
        generate_image("Photograph of a cup", tmp_path / "image.png", width=width, height=height)
