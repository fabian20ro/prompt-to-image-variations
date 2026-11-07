"""Opt-in real LM Studio/Metal acceptance checks; never run by pytest.

uv run python scripts/verify_flux3.py --mode corpus
uv run python scripts/verify_flux3.py --mode e2e
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grammar_generator import generate_grammar
from config import settings
from image_enhancer import clear_enhancer_cache
from image_generator import clear_model_cache, generate_image, validate_checkpoint
from lm_studio import unload_all_models
from pipeline import PipelineExecutor
from tracery_runner import run_tracery


CORPUS = [
    ("Outdoor portrait photograph of one elderly woman with silver braids beside an oak tree.", []),
    ("Studio product photograph of one red ceramic teapot. Keep its red color and white background.", []),
    ("Landscape photograph of a lake beneath snow-covered mountains at dawn, no people.", []),
    ("Watercolor illustration of one fox sleeping beneath a pine tree; keep watercolor fixed.", []),
    ('Vertical tea poster, exact headline "MOUNTAIN TEA". One canister below the text.', ["MOUNTAIN TEA"]),
    ('Minimal French cafe poster reading exactly "L’été à Paris" in elegant serif lettering.', ["L’été à Paris"]),
    ('Japanese travel poster reading exactly "京都" above a watercolor temple illustration.', ["京都"]),
    ("Photograph of exactly two cyclists, the cyclist in yellow at frame-left and in blue at frame-right.", []),
    ("Flat vector illustration, three orange circles above a single blue rectangle on cream paper.", []),
    ('Two-panel comic: left shows a closed door with "BEFORE"; right an open door with "AFTER".', ["BEFORE", "AFTER"]),
    ('Wide bakery banner reading "FRESH BREAD" at left and one loaf at right.', ["FRESH BREAD"]),
    ('Product label reading exactly "Batch #1 [OPEN]" on a green bottle, photographed on white.', ["Batch #1 [OPEN]"]),
]


def checkpoint(output: Path, revision: str) -> list[dict]:
    """Verify the saved bundle and record the explicitly observed download revision."""
    import mlx.core as mx
    from PIL import Image
    path = validate_checkpoint()
    snapshot = Path.home() / ".cache/huggingface/hub/models--black-forest-labs--flux-3-image-klein/snapshots" / revision
    if not revision or not snapshot.is_dir() or not all(file.exists() for file in snapshot.rglob("*")):
        raise ValueError("Supply the complete source snapshot revision used by mflux-save")
    started = time.monotonic()
    mx.reset_peak_memory()
    image = generate_image("Studio photograph of one red ceramic teapot on white linen. Soft light enters from frame-left, producing a short grey shadow to the right. The round body has a curved handle and a short spout. Straight-on medium view, plain cream background.", output / "checkpoint.png", seed=42, width=512, height=512)
    record = {"success": Image.open(image).size == (512, 512), "source_revision": revision,
              "source_repo": "black-forest-labs/flux-3-image-klein", "mflux_version": "0.24.0",
              "saved_bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
              "seconds": round(time.monotonic() - started, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 3)}
    if record["success"]:
        (path / "provenance.json").write_text(json.dumps(record, indent=2))
    clear_model_cache()
    (output / "checkpoint.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return [record]


def corpus(output: Path) -> list[dict]:
    records = []
    for index, (prompt, literals) in enumerate(CORPUS):
        for repetition in range(2):
            started = time.monotonic()
            record = {"case": index, "repetition": repetition, "prompt": prompt}
            try:
                width, height = (1152, 864) if index == 10 else (864, 1152)
                grammar, _, raw = generate_grammar(prompt, use_cache=False, width=width, height=height)
                samples = run_tracery(grammar, 7)
                record.update(grammar=json.loads(grammar), raw_response=raw, samples=samples,
                              exact_text=all(text in sample for text in literals for sample in samples),
                              words=[len(sample.split()) for sample in samples])
                record["success"] = record["exact_text"]
            except Exception as exc:
                record.update(success=False, error=str(exc))
            record["seconds"] = round(time.monotonic() - started, 2)
            records.append(record)
            (output / "corpus.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
            print(json.dumps({key: record[key] for key in ("case", "repetition", "success", "seconds")}), flush=True)
    return records


def e2e(output: Path) -> list[dict]:
    import mlx.core as mx
    import requests
    from PIL import Image

    records = []
    for index in range(2):
        clear_model_cache()
        clear_enhancer_cache()
        unload_all_models()
        mx.reset_peak_memory()
        started = time.monotonic()
        run = output / f"e2e-{index}"
        executor = PipelineExecutor(on_progress=lambda stage, current, total, message: print(
            f"{stage}: {current}/{total} {message}", flush=True))
        result = executor.run_full_pipeline(
            prompt='Studio photograph for a vertical tea poster reading "MOUNTAIN TEA", one ceramic canister.',
            count=1, no_cache=True, generate_images=True, width=864, height=1152,
            seed=42 + index, enhance=True, enhance_after=True, enhance_tiled_vae=True, output_dir=run,
        )
        record = {"run": index, "result": asdict(result), "seconds": round(time.monotonic() - started, 2),
                  "peak_gb": round(mx.get_peak_memory() / 1e9, 3)}
        if result.success:
            record["dimensions"] = list(Image.open(run / "image_0_0.png").size)
            record["metadata"] = json.loads((run / "image.metaprompt.json").read_text())
            record["image_provenance"] = json.loads((run / "image_0_0.generation.json").read_text())
        clear_model_cache()
        clear_enhancer_cache()
        record["active_after_release_gb"] = round(mx.get_active_memory() / 1e9, 4)
        record["cache_after_release_gb"] = round(mx.get_cache_memory() / 1e9, 4)
        inventory = requests.get("http://localhost:1234/api/v1/models", timeout=30)
        inventory.raise_for_status()
        record["loaded_lms_models"] = [m["key"] for m in inventory.json()["models"] if m.get("loaded_instances")]
        record["success"] = result.success and record.get("dimensions") == [1728, 2304] and not record["loaded_lms_models"]
        records.append(record)
        (output / "e2e.json").write_text(json.dumps(records, indent=2, default=str))
        print(json.dumps(record, default=str), flush=True)
        if not record["success"]:
            break
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint", "corpus", "e2e"), required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--output", type=Path, default=Path("generated/verification") / datetime.now().strftime("flux3-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if args.mode == "checkpoint":
        results = checkpoint(args.output, args.source_revision)
    else:
        results = corpus(args.output) if args.mode == "corpus" else e2e(args.output)
    raise SystemExit(0 if all(row["success"] for row in results) else 1)
