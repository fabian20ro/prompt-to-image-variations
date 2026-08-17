"""Tests for image_enhancer.py - SeedVR2 enhancement wrapper."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_enhancer import (
    clear_enhancer_cache,
    _get_enhancer,
    _enhancer_cache,
    enhance_image,
    collect_images,
)


class TestEnhancerCache:
    """Tests for enhancer caching functionality."""

    def test_clear_enhancer_cache(self):
        """Test that cache clearing works."""
        _enhancer_cache["test_key"] = "test_value"
        assert len(_enhancer_cache) > 0

        clear_enhancer_cache()
        assert len(_enhancer_cache) == 0

    def test_clear_enhancer_cache_calls_gc(self):
        """Test that garbage collection is called on cache clear."""
        import gc
        with patch.object(gc, 'collect') as mock_gc:
            clear_enhancer_cache()
            mock_gc.assert_called_once()


class TestGetEnhancer:
    """Tests for enhancer instantiation."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_enhancer_cache()

    def test_get_enhancer_cache_hit(self):
        """Test that cached enhancers are reused."""
        mock_enhancer = MagicMock()
        cache_key = True
        _enhancer_cache[cache_key] = mock_enhancer

        result = _get_enhancer(tiled_vae=True)
        assert result is mock_enhancer

    def test_get_enhancer_cache_hit_default_false(self):
        """Test that cached enhancers are reused on the default tiled_vae=False path."""
        mock_enhancer = MagicMock()
        _enhancer_cache[False] = mock_enhancer

        result = _get_enhancer(tiled_vae=False)
        assert result is mock_enhancer

    def test_get_enhancer_mflux_not_installed(self):
        """Test error when mflux is not installed."""
        with patch.dict("sys.modules", {"mflux.models.seedvr2.variants.upscale.seedvr2": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'mflux'")):
                with pytest.raises(ImportError, match="mflux is required"):
                    _get_enhancer()

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_creation(self):
        """Test SeedVR2 enhancer creation with mocked mflux."""
        mock_instance = MagicMock()

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer(tiled_vae=True)

        module_seedvr2.SeedVR2.assert_called_once_with(quantize=8)
        assert result is mock_instance

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_disables_tiling(self):
        """Test that tiled_vae=False disables tiling."""
        mock_instance = MagicMock()
        mock_instance.tiling_config = MagicMock()

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer(tiled_vae=False)

        assert result.tiling_config is None

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_default_no_args(self):
        """Test the default no-args path: quantize=8 and tiling disabled."""
        mock_instance = MagicMock()
        mock_instance.tiling_config = MagicMock()

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer()

        module_seedvr2.SeedVR2.assert_called_once_with(quantize=8)
        assert result.tiling_config is None

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_preserves_tiling_when_enabled(self):
        """Test that tiled_vae=True preserves the default tiling config."""
        mock_instance = MagicMock()
        original_tiling = MagicMock()
        mock_instance.tiling_config = original_tiling

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer(tiled_vae=True)

        assert result.tiling_config is original_tiling


class TestEnhanceImage:
    """Tests for single image enhancement."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_enhancer_cache()

    def test_enhance_image_file_not_found(self, temp_dir):
        """Test error when image doesn't exist."""
        missing = temp_dir / "nonexistent.png"
        with pytest.raises(FileNotFoundError, match="Image not found"):
            enhance_image(image_path=missing, output_path=missing)

    @patch("image_enhancer._get_enhancer")
    @patch("image_enhancer.unload_all_models")
    def test_enhance_image_basic(self, mock_unload, mock_get_enhancer, temp_dir):
        """Test basic image enhancement."""
        # Create a real image file
        img = Image.new("RGB", (100, 100), color="red")
        image_path = temp_dir / "test.png"
        img.save(image_path)
        output_path = temp_dir / "enhanced.png"

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result
        mock_get_enhancer.return_value = mock_enhancer

        # Mock ScaleFactor import
        mock_scale_factor = MagicMock()
        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(ScaleFactor=mock_scale_factor),
        }):
            result = enhance_image(
                image_path=image_path,
                output_path=output_path,
                softness=0.7,
                seed=42,
            )

        assert result == output_path
        mock_unload.assert_called_once_with()
        mock_enhancer.generate_image.assert_called_once()
        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["softness"] == 0.7
        assert call_kwargs["image_path"] == str(image_path)
        mock_result.save.assert_called_once_with(path=str(output_path), overwrite=True)
        # Verify resolution is always ScaleFactor(2) for 2x upscaling (line 116 of image_enhancer.py)
        mock_scale_factor.assert_called_once_with(2)

    @patch("image_enhancer._get_enhancer")
    @patch("image_enhancer.unload_all_models")
    def test_enhance_image_random_seed(self, _mock_unload, mock_get_enhancer, temp_dir):
        """Test that a random seed is used when none specified."""
        img = Image.new("RGB", (100, 100))
        image_path = temp_dir / "test.png"
        img.save(image_path)

        mock_enhancer = MagicMock()
        mock_enhancer.generate_image.return_value = MagicMock()
        mock_get_enhancer.return_value = mock_enhancer

        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(),
        }):
            enhance_image(image_path=image_path, output_path=image_path)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert isinstance(call_kwargs["seed"], int)

    @patch("image_enhancer._get_enhancer")
    @patch("image_enhancer.unload_all_models")
    def test_enhance_image_creates_output_dir(self, _mock_unload, mock_get_enhancer, temp_dir):
        """Test that output directory is created."""
        img = Image.new("RGB", (100, 100))
        image_path = temp_dir / "test.png"
        img.save(image_path)
        output_path = temp_dir / "nested" / "dir" / "enhanced.png"

        mock_enhancer = MagicMock()
        mock_enhancer.generate_image.return_value = MagicMock()
        mock_get_enhancer.return_value = mock_enhancer

        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(),
        }):
            enhance_image(image_path=image_path, output_path=output_path)

        assert output_path.parent.exists()


class TestCollectImages:
    """Tests for image collection from various sources."""

    def test_collect_single_file(self, temp_dir):
        """Test collecting a single image file."""
        img = Image.new("RGB", (10, 10))
        image_path = temp_dir / "test.png"
        img.save(image_path)

        result = collect_images(str(image_path))
        assert result == [image_path]

    def test_collect_single_file_not_image(self, temp_dir):
        """Test error for non-image file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("not an image")

        with pytest.raises(ValueError, match="Not an image file"):
            collect_images(str(text_file))

    def test_collect_directory(self, temp_dir):
        """Test collecting all images from a directory."""
        for name in ["a.png", "b.jpg", "c.txt"]:
            p = temp_dir / name
            if name.endswith((".png", ".jpg")):
                Image.new("RGB", (10, 10)).save(p)
            else:
                p.write_text("text")

        result = collect_images(str(temp_dir))
        names = [r.name for r in result]
        assert "a.png" in names
        assert "b.jpg" in names
        assert "c.txt" not in names

    def test_collect_empty_directory(self, temp_dir):
        """Test error for directory with no images."""
        empty = temp_dir / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="No images found in directory"):
            collect_images(str(empty))

    def test_collect_directory_with_only_non_images(self, temp_dir):
        """Test error for directory with only non-image files."""
        (temp_dir / "test.txt").write_text("text")
        (temp_dir / "notes.log").write_text("logs")

        with pytest.raises(ValueError, match="No images found in directory"):
            collect_images(str(temp_dir))

    def test_collect_glob_pattern(self, temp_dir):
        """Test collecting images via glob pattern."""
        for name in ["a.png", "b.png", "c.jpg"]:
            Image.new("RGB", (10, 10)).save(temp_dir / name)

        result = collect_images(str(temp_dir / "*.png"))
        assert len(result) == 2

    def test_collect_glob_no_match(self, temp_dir):
        """Test error when glob matches no images."""
        with pytest.raises(ValueError, match="No images found matching pattern"):
            collect_images(str(temp_dir / "*.xyz"))

    def test_collect_uppercase_extensions(self, temp_dir):
        """Test that uppercase image extensions are detected in all code paths."""
        img = Image.new("RGB", (10, 10))
        png_path = temp_dir / "test.PNG"
        jpg_path = temp_dir / "photo.JPG"
        webp_path = temp_dir / "image.WEBP"
        img.save(png_path)
        img.save(jpg_path)
        img.save(webp_path)

        # Single file path
        assert collect_images(str(png_path)) == [png_path]

        # Directory scan - verifies case-insensitive glob in production code (line 159-160)
        result = collect_images(str(temp_dir))
        names = {r.name for r in result}
        assert "test.PNG" in names
        assert "photo.JPG" in names
        assert "image.WEBP" in names

        # Glob pattern with uppercase extension - exercises the .upper() normalization (line 169)
        result = collect_images(str(temp_dir / "*.PNG"))
        assert len(result) == 1
        assert result[0] == png_path


def test_enhance_image_scale_factor_missing_raises_import_error(temp_dir):
    """Test that missing ScaleFactor triggers install-hint ImportError."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch

    img = Image.new("RGB", (10, 10))
    img_path = temp_dir / "test.png"
    out_path = temp_dir / "out.png"
    img.save(img_path)

    mock_enhancer = MagicMock()
    mock_enhancer.generate_image.return_value = MagicMock()

    with patch("image_enhancer.unload_all_models"), \
         patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
         patch.dict(sys.modules, {"mflux.utils.scale_factor": None}, clear=False):
        # Remove ScaleFactor from sys.modules if present to trigger ImportError
        sys.modules.pop("mflux.utils.scale_factor", None)
        with pytest.raises(ImportError, match="mflux is required"):
            enhance_image(img_path, out_path)


@pytest.mark.parametrize("softness", [-0.1, 1.1])
def test_enhance_image_invalid_softness(softness):
    from image_enhancer import enhance_image
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)
        with pytest.raises(ValueError, match="softness must be between 0.0 and 1.0"):
            enhance_image(img_path, out_path, softness=softness)


@pytest.mark.parametrize("width", [0, -4, 7])
def test_enhance_image_invalid_width(width):
    from image_enhancer import enhance_image
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)
        with pytest.raises(ValueError):
            enhance_image(img_path, out_path, width=width, height=800)


@pytest.mark.parametrize("height", [0, -4, 7])
def test_enhance_image_invalid_height(height):
    from image_enhancer import enhance_image
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)
        with pytest.raises(ValueError):
            enhance_image(img_path, out_path, width=800, height=height)


@pytest.mark.parametrize("width,height", [(0, -4), (-5, 7), (9, 3)])
def test_enhance_image_both_invalid_dimensions(width, height):
    """Test that simultaneous invalid width and height raise ValueError."""
    from image_enhancer import enhance_image
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)
        with pytest.raises(ValueError):
            enhance_image(img_path, out_path, width=width, height=height)


def test_enhance_image_valid_dimensions_pass():
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, width=1024, height=1536)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1536


def test_enhance_image_dimensions_not_passed_when_none():
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert "width" not in call_kwargs
        assert "height" not in call_kwargs


def test_enhance_image_width_only_passes_width():
    """Test that only width is forwarded to the enhancer when height is None."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, width=2048)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["width"] == 2048
        assert "height" not in call_kwargs


def test_enhance_image_height_only_passes_height():
    """Test that only height is forwarded to the enhancer when width is None."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, height=3072)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert "width" not in call_kwargs
        assert call_kwargs["height"] == 3072


def test_enhance_image_width_only_invalid_passes_validation():
    """Test width validation (non-multiple of 8) when height is None."""
    from image_enhancer import enhance_image
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        with pytest.raises(ValueError):
            enhance_image(img_path, out_path, width=9)


def test_enhance_image_tiled_vae_true_passes_to_get_enhancer():
    """Test that tiled_vae=True is forwarded to _get_enhancer."""
    from image_enhancer import enhance_image, _get_enhancer
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer) as mock_get, \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, tiled_vae=True)

        mock_get.assert_called_once_with(True)


def test_enhance_image_tiled_vae_false_default_passes_to_get_enhancer():
    """Test that default tiled_vae=False is forwarded to _get_enhancer."""
    from image_enhancer import enhance_image, _get_enhancer
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer) as mock_get, \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path)

        mock_get.assert_called_once_with(False)


def test_enhance_image_default_softness_forwarded():
    """Test that default softness=0.5 is forwarded to generate_image when not specified."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["softness"] == 0.5


def test_enhance_image_user_seed_propagates_to_generate_image():
    """Test that a user-specified seed is forwarded to generate_image."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, seed=42)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["seed"] == 42


def test_enhance_image_zero_seed_not_randomized():
    """Test that seed=0 is passed through unchanged (not replaced by random)."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock()}):
            enhance_image(img_path, out_path, seed=0)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["seed"] == 0


def test_enhance_image_resolution_always_scale_factor_2():
    """Test that resolution is always ScaleFactor(2) regardless of other kwargs."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result
        mock_scale_factor_instance = MagicMock()
        mock_scale_factor_cls = MagicMock(return_value=mock_scale_factor_instance)

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock(ScaleFactor=mock_scale_factor_cls)}):
            enhance_image(img_path, out_path, width=800, height=1200)

        mock_scale_factor_cls.assert_called_once_with(2)
        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert isinstance(call_kwargs["resolution"], MagicMock)


class TestCollectImagesEdgeCases:
    """Tests for edge cases in image collection."""

    def test_collect_directory_with_uppercase_extensions(self, temp_dir):
        """Test that uppercase file extensions are collected."""
        Image.new("RGB", (10, 10)).save(temp_dir / "a.PNG")
        Image.new("RGB", (10, 10)).save(temp_dir / "b.JPG")

        result = collect_images(str(temp_dir))
        names = [r.name for r in result]
        assert "a.PNG" in names
        assert "b.JPG" in names

    def test_collect_directory_with_jpeg_extension(self, temp_dir):
        """Test that .jpeg extension is collected."""
        Image.new("RGB", (10, 10)).save(temp_dir / "photo.jpeg")

        result = collect_images(str(temp_dir))
        assert result[0].name == "photo.jpeg"

    def test_collect_directory_follows_symlinked_subdirectories(self, temp_dir):
        """Test that images inside symlinked subdirectories are found."""
        import os
        # Create a real subdir with an image
        sub = temp_dir / "subdir"
        sub.mkdir()
        Image.new("RGB", (10, 10)).save(sub / "inside.png")

        # Symlink the subdir into temp_dir's root
        link = temp_dir / "linked_subdir"
        os.symlink(str(sub), str(link))

        result = collect_images(str(temp_dir))
        names = {r.name for r in result}
        assert "inside.png" in names

    def test_collect_single_jpeg_file(self):
        """Test collecting a single .jpeg file."""
        import tempfile
        from pathlib import Path as PPath
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (10, 10))
            fpath = PPath(tmpdir) / "photo.jpeg"
            img.save(fpath)
            result = collect_images(str(fpath))
            assert len(result) == 1

    def test_collect_directory_with_webp_extension(self, temp_dir):
        """Test that .webp extension is collected."""
        Image.new("RGB", (10, 10)).save(temp_dir / "anim.webp")

        result = collect_images(str(temp_dir))
        names = [r.name for r in result]
        assert "anim.webp" in names


def test_enhance_image_scale_factor_passed_to_generate_image():
    """Test that ScaleFactor(2) is correctly forwarded to generate_image."""
    from image_enhancer import enhance_image
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        out_path = Path(tmpdir) / "out.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result
        mock_scale_factor_instance = MagicMock()
        mock_scale_factor_cls = MagicMock(return_value=mock_scale_factor_instance)

        with patch("image_enhancer.unload_all_models"), \
             patch("image_enhancer._get_enhancer", return_value=mock_enhancer), \
             patch.dict(sys.modules, {"mflux.utils.scale_factor": MagicMock(ScaleFactor=mock_scale_factor_cls)}):
            enhance_image(img_path, out_path)

        mock_scale_factor_cls.assert_called_once_with(2)
        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert isinstance(call_kwargs["resolution"], MagicMock)
