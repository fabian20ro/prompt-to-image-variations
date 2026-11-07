"""Migration behavior at public grammar, renderer, and pipeline boundaries."""

import json
from unittest.mock import Mock
import pytest

from grammar_generator import generate_grammar
from tracery_runner import run_tracery


def test_standalone_cli_enhances_with_real_progressbar(monkeypatch, tmp_path):
    import sys
    import shutil
    from types import SimpleNamespace
    from click.testing import CliRunner
    from PIL import Image
    from cli import main
    from image_enhancer import clear_enhancer_cache

    source = tmp_path / "teapot.png"
    Image.new("RGB", (16, 16)).save(source)
    class SeedVR2:
        def __init__(self, **kwargs):
            pass

        def generate_image(self, **kwargs):
            return SimpleNamespace(save=lambda path, overwrite: Image.new("RGB", (32, 32)).save(path))

    monkeypatch.setitem(sys.modules, "mflux.models.seedvr2.variants.upscale.seedvr2", SimpleNamespace(SeedVR2=SeedVR2))
    monkeypatch.setitem(sys.modules, "mflux.utils.scale_factor", SimpleNamespace(ScaleFactor=lambda value: value))
    original_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda command: "/fake/lms" if command == "lms" else original_which(command))
    monkeypatch.setattr("subprocess.run", Mock(return_value=SimpleNamespace(returncode=0)))
    clear_enhancer_cache()
    try:
        result = CliRunner().invoke(main, ["--enhance-images", str(source), "--seed", "42"])
        assert result.exit_code == 0, result.output
        assert Image.open(source).size == (32, 32)
    finally:
        clear_enhancer_cache()


def test_escaped_backslash_does_not_hide_a_tracery_action():
    from tracery_runner import TraceryError
    grammar = {"origin": [r"\\[color:red\\]#color#"], "color": ["blue"]}
    with pytest.raises(TraceryError, match="actions/modifiers"):
        run_tracery(json.dumps(grammar), 1)


def test_expanded_caption_cleans_rule_boundary_punctuation_but_keeps_visible_text():
    result = run_tracery(json.dumps({"origin": ['Photograph. #scene#. The sign reads "Wait..".'],
                                     "scene": ["A cup on linen."]}), 1)
    assert result == ['Photograph. A cup on linen. The sign reads "Wait..".']


@pytest.fixture(autouse=True)
def mflux_package_metadata(monkeypatch):
    """The external runtime is faked; tests also work without installing Metal wheels."""
    from importlib import metadata
    original = metadata.version
    monkeypatch.setattr(metadata, "version", lambda name: "0.24.0" if name == "mflux" else original(name))


def test_creation_cannot_overwrite_captions_in_existing_flux_run(tmp_path):
    from pipeline import PipelineExecutor
    metadata = tmp_path / "image.metaprompt.json"
    metadata.write_text(json.dumps({"model": "flux3-image-klein", "count": 1}))
    prompt = tmp_path / "image_0.txt"
    prompt.write_text("Original caption")
    result = PipelineExecutor().run_from_grammar_text('{"origin":["New caption"]}', output_dir=tmp_path, resume=True)
    assert not result.success and "new run" in result.error
    assert prompt.read_text() == "Original caption"


def test_invalid_canvas_fails_before_inference_or_run_files(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from cli import main
    from pipeline import PipelineExecutor
    monkeypatch.setattr("requests.get", Mock(side_effect=AssertionError("Must not call LM Studio")))
    output = tmp_path / "invalid"
    result = CliRunner().invoke(main, ["--prompt", "Cup", "--width", "520", "--no-cache", "--output", str(output)])
    assert result.exit_code != 0 and "multiples of 16" in result.output
    assert not output.exists()
    result = PipelineExecutor().run_from_grammar_text('{"origin":["Cup"]}', width=520, output_dir=output)
    assert not result.success and "multiples of 16" in result.error
    assert not output.exists()


def test_existing_flux_run_keeps_canvas_and_records_current_render_contract(monkeypatch, tmp_path):
    from pipeline import PipelineExecutor
    from types import SimpleNamespace
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(prompts_dir=tmp_path.parent))
    metadata = tmp_path / "image.metaprompt.json"
    metadata.write_text(json.dumps({"model": "flux3-image-klein", "count": 1,
                                   "image_generation": {"width": 1024, "height": 768, "seed": 321}}))
    (tmp_path / "image_0.txt").write_text("Studio photograph of a cup.")
    result = PipelineExecutor().generate_all_images(tmp_path.name, images_per_prompt=0)
    assert result.success
    image = json.loads(metadata.read_text())["image_generation"]
    assert (image["width"], image["height"]) == (1024, 768)
    assert image["seed"] == 321
    assert image["mflux_version"] == "0.24.0" and image["variant"] == "distilled-noCFG"
    from cli import main
    from click.testing import CliRunner
    result = CliRunner().invoke(main, ["--from-prompts", str(tmp_path), "--generate-images", "--images-per-prompt", "0"])
    assert result.exit_code == 0, result.output
    image = json.loads(metadata.read_text())["image_generation"]
    assert (image["width"], image["height"]) == (1024, 768)
    assert image["seed"] == 321


def test_visible_literal_punctuation_survives_grammar_cleaning_and_expansion(monkeypatch):
    grammar = {"origin": ['Photograph of a sign reading "L’été \\#1 \\[OPEN\\]" in a #color# frame.'],
               "color": ["red", "blue", "green", "white", "black"]}
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [1]}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    reply = Mock()
    reply.json.return_value = {"output": [{"type": "message", "content": "Here is the grammar:\n" + json.dumps(grammar, ensure_ascii=False)}]}
    monkeypatch.setattr("requests.post", Mock(return_value=reply))
    result, _, _ = generate_grammar('Sign reading "L’été #1 [OPEN]"', use_cache=False)
    assert all('"L’été #1 [OPEN]"' in item for item in run_tracery(result, 7))
    grammar["origin"] = ['Photograph of a sign reading "<think>idea</think> ```" in a #color# frame.']
    reply.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
    result, _, _ = generate_grammar('Sign with literal markup', use_cache=False)
    assert '"<think>idea</think> ```"' in run_tracery(result, 1)[0]
    grammar["origin"] = ['Photograph of a sign reading "Batch #1 [OPEN]" in a #color# frame.']
    reply.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
    result, _, _ = generate_grammar('Label reading exactly "Batch #1 [OPEN]"', use_cache=False)
    assert '"Batch #1 [OPEN]"' in run_tracery(result, 1)[0]


@pytest.mark.parametrize("literal", ["#1", r"\#1"])
def test_repeated_quoted_user_text_is_escaped_only_once(monkeypatch, literal):
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [1]}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    reply = Mock()
    reply.json.return_value = {"output": [{"type": "message", "content": json.dumps({
        "origin": [f'A #color# sign reading "{literal}".'],
        "color": ["red", "blue", "green", "white", "black"],
    })}]}
    monkeypatch.setattr("requests.post", Mock(return_value=reply))
    grammar, _, _ = generate_grammar('The sign reads "#1" and repeats "#1".', use_cache=False)
    assert all('"#1"' in sample for sample in run_tracery(grammar, 7))


def test_imported_caption_run_records_renderer_without_claiming_qwen(monkeypatch, tmp_path):
    from pipeline import PipelineExecutor
    from types import SimpleNamespace
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(generated_dir=tmp_path))
    output = tmp_path / "import"
    result = PipelineExecutor().run_from_grammar_text('{"origin":["Studio photograph of a cup."]}', count=1, images_per_prompt=0, output_dir=output)
    assert result.success
    meta = json.loads((output / "image.metaprompt.json").read_text())
    assert meta["image_generation"]["mflux_version"] == "0.24.0"
    assert meta["image_generation"]["variant"] == "distilled-noCFG"
    assert meta["prompt_model"] is None
    assert meta["prompt_schema"] == "flux3-caption-v1"
    assert "Generated: 0 / 0 images" in (output / "image_gallery.html").read_text()


def test_qwen_configuration_errors_are_not_retried(monkeypatch):
    import requests
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": []}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"error":"Invalid context length"}'
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(requests.HTTPError):
        generate_grammar("A cup", use_cache=False)
    assert post.call_count == 1


def test_only_enhancement_exposes_tiling_controls():
    from click.testing import CliRunner
    from cli import main
    from server.models import GenerateRequest
    from pydantic import ValidationError
    help_text = CliRunner().invoke(main, ["--help"]).output
    assert "--no-enhance-tiled-vae" in help_text
    assert "--no-tiled-vae" not in help_text
    assert GenerateRequest(prompt="Cup", enhance_tiled_vae=True).enhance_tiled_vae
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Cup", tiled_vae=True)


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_passes_canvas_and_server_to_qwen_and_records_provenance(monkeypatch, tmp_path, dry_run):
    from click.testing import CliRunner
    from cli import main
    from types import SimpleNamespace
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(generated_dir=tmp_path))
    grammar = {"origin": ["Photograph of a #color# cup."], "color": ["red", "blue", "green", "white", "black"]}
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [1]}]}
    get = Mock(return_value=inventory)
    monkeypatch.setattr("requests.get", get)
    response = Mock()
    response.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    args = ["--prompt", "Cup", "--count", "1", "--no-cache", "--width", "1024", "--height", "768",
            "--base-url", "http://localhost:9999/v1", "--output", str(tmp_path / "run")]
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "image_worker.log").write_text("Starting worker")
    if dry_run:
        args.append("--dry-run")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert post.call_args.args[0] == "http://localhost:9999/api/v1/chat"
    assert "TARGET CANVAS: 1024:768" in post.call_args.kwargs["json"]["input"]
    if not dry_run:
        meta = json.loads((tmp_path / "run" / "image.metaprompt.json").read_text())
        assert meta["prompt_model"] == "qwen3.8-27b-splash"
        assert meta["prompt_schema"] == "flux3-caption-v1"
        image = meta["image_generation"]
        assert (image["width"], image["height"]) == (1024, 768)
        assert image["model"] == "flux3-image-klein"
        assert image["steps"] == 16 and image["guidance"] is None
        assert image["quantize"] == 4 and image["mflux_version"] == "0.24.0"


def test_api_rejects_invalid_canvas_and_unsafe_grammar():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import server.app  # Initialize the application's routes in runtime order.
    from server.routes import router
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post("/api/generate", json={"prompt": "Teapot", "width": 520}).status_code == 422
        response = client.post("/api/generate-from-grammar", json={"grammar": '{"origin":["#origin#"]}'})
        assert response.status_code == 400


def test_invalid_regeneration_preserves_existing_images(monkeypatch, tmp_path):
    from pipeline import PipelineExecutor
    from types import SimpleNamespace
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(prompts_dir=tmp_path.parent, saved_dir=tmp_path.parent / "backups"))
    (tmp_path / "image.metaprompt.json").write_text(json.dumps({"prefix": "image", "count": 1}))
    image = tmp_path / "image_0_0.png"
    from PIL import Image
    Image.new("RGB", (16, 16)).save(image)
    original = image.read_bytes()
    result = PipelineExecutor().regenerate_prompts(tmp_path.name, '{"origin":["#missing#"]}')
    assert not result.success
    assert "missing" in result.error
    assert image.read_bytes() == original


@pytest.mark.parametrize("operation", ["existing", "single", "all", "new"])
def test_pipeline_rejects_legacy_run_without_mutation(monkeypatch, tmp_path, operation):
    from pipeline import PipelineExecutor
    from types import SimpleNamespace
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(prompts_dir=tmp_path.parent))
    metadata = tmp_path / "image.metaprompt.json"
    original = json.dumps({"prefix": "image", "model": "ernie-image-turbo", "count": 1})
    metadata.write_text(original)
    (tmp_path / "image_0.txt").write_text("Original caption")
    executor = PipelineExecutor()
    if operation == "existing":
        result = executor.run_from_prompts(tmp_path, resume=True)
    elif operation == "single":
        result = executor.generate_single_image(tmp_path.name, 0)
    elif operation == "all":
        result = executor.generate_all_images(tmp_path.name)
    else:
        result = executor.run_from_grammar_text('{"origin":["New caption"]}', output_dir=tmp_path)
    assert not result.success
    assert "new run" in result.error
    assert metadata.read_text() == original
    assert (tmp_path / "image_0.txt").read_text() == "Original caption"


def test_renderer_rejects_unquantized_checkpoint_before_loading(monkeypatch, tmp_path):
    from dataclasses import replace
    from config import settings
    from image_generator import generate_image, clear_model_cache
    monkeypatch.setattr(settings, "image_generation", replace(settings.image_generation, model_path=tmp_path))
    clear_model_cache()
    with pytest.raises(ValueError, match="q4 checkpoint"):
        generate_image("A photograph", tmp_path / "image.png", width=512, height=512)


def test_checkpoint_requires_tokenizer_data_before_loading(monkeypatch, tmp_path):
    from config import settings
    from dataclasses import replace
    from image_generator import generate_image
    for component in ("transformer", "text_encoder", "vae"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / "0.safetensors").write_bytes(b"external weights")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"quantization_level": "4"}, "weight_map": {"weight": "0.safetensors"}}))
    (tmp_path / "text_encoder" / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(settings, "image_generation", replace(settings.image_generation, model_path=tmp_path))
    monkeypatch.setattr("subprocess.run", Mock(side_effect=AssertionError("Do not unload yet")))
    with pytest.raises(ValueError, match="tokenizer"):
        generate_image("Cup", tmp_path / "image.png")


def test_grammar_repairs_invalid_reply_once_and_revalidates_cache(monkeypatch, tmp_path):
    from grammar_generator import hash_prompt
    monkeypatch.setattr("grammar_generator.CACHE_DIR", tmp_path)
    (tmp_path / f"{hash_prompt('Teapot')}.tracery.json").write_text('{"origin":["#missing#"]}')
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [1]}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    valid = {"origin": ["Photograph of a #color# teapot."], "color": ["red", "blue", "green", "white", "black"]}
    replies = []
    rejected_caption = "Long rejected caption. " * 1000
    for grammar in ({"origin": [rejected_caption]}, valid):
        response = Mock()
        response.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
        replies.append(response)
    post = Mock(side_effect=replies)
    monkeypatch.setattr("requests.post", post)
    result, cached, _ = generate_grammar("Teapot")
    assert not cached
    assert run_tracery(result, 1)[0].startswith("Photograph of a ")
    assert post.call_count == 2
    assert "Repair" in post.call_args.kwargs["json"]["input"]
    assert rejected_caption not in post.call_args.kwargs["json"]["input"]
    assert generate_grammar("Teapot")[1]


def test_prompt_example_keeps_requested_subject_and_text_across_variations():
    from grammar_generator import get_system_prompt
    prompt = get_system_prompt()
    example = json.loads(prompt.split("Example grammar:\n", 1)[1])
    expanded = run_tracery(json.dumps(example), 50)
    assert all('A single closed ceramic tea canister' in text and '"MOUNTAIN TEA"' in text for text in expanded)
    assert all(len(options) == 7 for name, options in example.items() if name != "origin")


def test_expansion_rejects_recursive_grammar_before_recursing():
    from tracery_runner import TraceryError
    with pytest.raises(TraceryError, match="cycle"):
        run_tracery(json.dumps({"origin": ["#loop#"], "loop": ["#origin#"]}), 1)


def test_expansion_rejects_actions_instead_of_silently_losing_layout():
    from tracery_runner import TraceryError
    with pytest.raises(TraceryError, match="escape"):
        run_tracery(json.dumps({"origin": ['A box. [{"bbox":[0,0,1000,1000]}]']}), 1)


def test_renderer_rejects_non_flux_dimensions_before_loading(tmp_path):
    from image_generator import generate_image
    with pytest.raises(ValueError, match="multiples of 16"):
        generate_image("A photograph", tmp_path / "image.png", width=520, height=512)


def test_cached_grammar_is_specific_to_canvas_and_preserves_visible_quotes(monkeypatch, tmp_path):
    grammar = {"origin": ['Studio photograph. A sign reads “L’été”. #detail#.'],
               "detail": ["Oak frame", "Pine frame", "Steel frame", "Copper frame", "Bronze frame"]}
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [{"id": "qwen3.8-27b-splash"}]}]}
    reply = Mock()
    reply.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    post = Mock(return_value=reply)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("grammar_generator.CACHE_DIR", tmp_path)
    first, cached, _ = generate_grammar("A sign", width=1024, height=768)
    assert not cached
    assert '“L’été”' in run_tracery(first, 1)[0]
    assert generate_grammar("A sign", width=1024, height=768)[1]
    assert not generate_grammar("A sign", width=768, height=1024)[1]
    assert "TARGET CANVAS: 768:1024" in post.call_args.kwargs["json"]["input"]


def test_local_flux_render_uses_distilled_sampling_and_saves_image(monkeypatch, tmp_path):
    import sys
    import weakref
    from types import SimpleNamespace
    from dataclasses import replace
    from PIL import Image
    from config import settings
    from image_generator import generate_image, clear_model_cache

    for component in ("transformer", "text_encoder", "vae"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / "0.safetensors").write_bytes(b"fake external weights")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"quantization_level": "4"}, "weight_map": {"weight": "0.safetensors"}}))
    (tmp_path / "text_encoder" / "tokenizer_config.json").write_text("{}")
    (tmp_path / "text_encoder" / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(settings, "image_generation", replace(settings.image_generation, model_path=tmp_path))
    events = []
    models = []
    allocator_buffers = []
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(clear_cache=allocator_buffers.clear))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/lms")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: (events.append("unload") or SimpleNamespace(returncode=0)))
    class Flux:
        def __init__(self, **kwargs):
            assert events[-1] == "unload"
            assert all(ref() is None for ref in models)
            models.append(weakref.ref(self))
            assert kwargs["model_path"] == str(tmp_path)
            assert kwargs["quantize"] is None
        def generate_image(self, **kwargs):
            allocator_buffers.append("allocated")
            assert kwargs["num_inference_steps"] == 16
            assert "guidance" not in kwargs
            return SimpleNamespace(save=lambda **k: Image.new("RGB", (512, 512)).save(k["path"]))
    monkeypatch.setitem(sys.modules, "mflux.models.common.config", SimpleNamespace(
        ModelConfig=SimpleNamespace(flux3_image_klein=lambda: "distilled")))
    monkeypatch.setitem(sys.modules, "mflux.models.flux3.variants", SimpleNamespace(Flux3Klein=Flux))
    clear_model_cache()
    output = generate_image("Studio photograph of a teapot.", tmp_path / "test.png", seed=42, width=512, height=512)
    assert Image.open(output).size == (512, 512)
    saved = json.loads(output.with_suffix(".generation.json").read_text())
    assert saved["seed"] == 42 and saved["prompt"] == "Studio photograph of a teapot."
    assert saved["model"] == "flux3-image-klein" and saved["quantize"] == 4
    from image_enhancer import enhance_image, clear_enhancer_cache
    class Enhancer:
        def __init__(self, **kwargs):
            assert all(ref() is None for ref in models), "Release FLUX before SeedVR2"
            assert not allocator_buffers, "Release MLX cached buffers before SeedVR2"
            models.append(weakref.ref(self))
        def generate_image(self, **kwargs):
            return SimpleNamespace(save=lambda **k: Image.new("RGB", (1024, 1024)).save(k["path"]))
    monkeypatch.setitem(sys.modules, "mflux.models.seedvr2.variants.upscale.seedvr2", SimpleNamespace(SeedVR2=Enhancer))
    monkeypatch.setitem(sys.modules, "mflux.utils.scale_factor", SimpleNamespace(ScaleFactor=lambda n: n))
    clear_enhancer_cache()
    enhance_image(output, output)
    assert Image.open(output).size == (1024, 1024)
    enhance_image(output, output, tiled_vae=True)
    generate_image("Another photograph", output, seed=43, width=512, height=512)
    assert events == ["unload", "unload", "unload", "unload"]
    from pipeline import PipelineExecutor
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr("pipeline.paths", SimpleNamespace(prompts_dir=tmp_path))
    (run / "image.metaprompt.json").write_text(json.dumps({"model": "flux3-image-klein", "image_generation": {
        "seed": 100, "images_per_prompt": 2, "width": 512, "height": 512}}))
    (run / "image_1.txt").write_text("A cup")
    assert PipelineExecutor().generate_single_image("run", 1, 1).success
    assert json.loads((run / "image_1_1.generation.json").read_text())["seed"] == 103
    def inventory(*args, **kwargs):
        assert all(ref() is None for ref in models), "Release image models before Qwen"
        response = Mock()
        response.json.return_value = {"models": [{"key": "qwen3.8-27b-splash", "loaded_instances": [1]}]}
        return response
    monkeypatch.setattr("requests.get", inventory)
    response = Mock()
    response.json.return_value = {"output": [{"type": "message", "content": json.dumps({
        "origin": ["Photograph of a #color# cup."], "color": ["red", "blue", "green", "white", "black"]})}]}
    monkeypatch.setattr("requests.post", Mock(return_value=response))
    generate_grammar("Cup", use_cache=False)
    clear_model_cache()
    clear_enhancer_cache()


def test_qwen_generates_expandable_captions_without_reasoning(monkeypatch):
    grammar = {"origin": ["Studio photograph of a #subject#."],
               "subject": ["cup", "teapot", "vase", "bowl", "plate", "jug", "saucer"]}
    inventory = Mock()
    inventory.json.return_value = {"models": [{"key": "qwen3.8-27b-splash",
                                               "loaded_instances": [{"id": "qwen3.8-27b-splash"}]}]}
    reply = Mock()
    reply.json.return_value = {"output": [{"type": "message", "content": json.dumps(grammar)}]}
    monkeypatch.setattr("requests.get", Mock(return_value=inventory))
    post = Mock(return_value=reply)
    monkeypatch.setattr("requests.post", post)

    result, cached, _ = generate_grammar("Ceramic studio still life", use_cache=False)

    assert not cached
    assert all(text.startswith("Studio photograph of a ") for text in run_tracery(result, 7))
    assert post.call_args.kwargs["json"]["model"] == "qwen3.8-27b-splash"
    assert post.call_args.kwargs["json"]["reasoning"] == "off"
