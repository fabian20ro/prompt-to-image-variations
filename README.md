# Prompt to Image Variations

Local procedural prompt generation and image rendering for ERNIE-Image-Turbo on Apple Silicon.

The application asks a local Gemma model in LM Studio to create an ERNIE-oriented Tracery grammar, expands that grammar into prompt variations, and optionally renders them with a persistent 4-bit ERNIE-Image-Turbo checkpoint through mflux. SeedVR2 2× enhancement is available as a second local stage.

No paid or hosted inference API is used.

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- LM Studio with `google/gemma-4-26b-a4b-qat`
- LM Studio CLI (`lms`) on `PATH`
- Enough free disk space to download the source checkpoint and save the approximately 6.2 GB q4 checkpoint

## Installation

```bash
uv sync --group dev
uv sync --extra images --group dev
```

Start the LM Studio local server at `http://localhost:1234/v1`. The configured model ID is:

```text
google/gemma-4-26b-a4b-qat
```

Override it when necessary:

```bash
export PROMPT_GEN_LM_STUDIO_MODEL=google/gemma-4-26b-a4b-qat
```

## Provision ERNIE-Image-Turbo q4

The application intentionally loads only a previously saved 4-bit checkpoint. Default location:

```text
~/Library/Caches/mflux/models/ernie-image-turbo-4bit
```

Create it once with mflux 0.18.0 or newer:

```bash
mkdir -p ~/Library/Caches/mflux/models
uv run mflux-save \
  --model ernie-image-turbo \
  --quantize 4 \
  --path ~/Library/Caches/mflux/models/ernie-image-turbo-4bit
```

Custom location:

```bash
export PROMPT_GEN_ERNIE_MODEL_PATH=/absolute/path/to/ernie-image-turbo-4bit
```

After verifying generation, the larger source checkpoint in the Hugging Face cache may be removed manually. Do not remove the saved q4 directory.

## Memory handoff

Grammar inference and image inference do not coexist in unified memory. Immediately before loading ERNIE or SeedVR2, the application runs:

```bash
lms unload --all
```

Image generation fails closed if the CLI is missing, times out, or cannot unload the models. Before a later grammar request, the application checks LM Studio's native model inventory and synchronously reloads Gemma by its exact model ID. Transient load cancellations are retried three times.

## CLI

Generate prompts:

```bash
uv run python src/cli.py \
  --prompt "a barn owl on a mossy branch" \
  --count 20
```

Generate prompts and ERNIE images:

```bash
uv run python src/cli.py \
  --prompt "a bilingual science museum poster" \
  --count 4 \
  --generate-images \
  --images-per-prompt 1 \
  --width 1024 \
  --height 1024
```

Generate and enhance:

```bash
uv run python src/cli.py \
  --prompt "a cinematic mountain observatory" \
  --count 2 \
  --generate-images \
  --enhance \
  --enhance-after
```

Enhance existing images:

```bash
uv run python src/cli.py --enhance-images path/to/image.png
uv run python src/cli.py --enhance-images path/to/folder/
uv run python src/cli.py --enhance-images "generated/prompts/*/*.png"
```

Resume from existing artifacts:

```bash
uv run python src/cli.py --from-grammar generated/grammars/example.tracery.json --count 50
uv run python src/cli.py --from-prompts generated/prompts/run-id --generate-images --resume
```

Important options:

| Option | Purpose |
|---|---|
| `--prompt TEXT` | Source image idea |
| `--count INT` | Number of Tracery expansions |
| `--generate-images` | Render expanded prompts |
| `--images-per-prompt INT` | Images per prompt; `0` keeps a prompt-only layout |
| `--width`, `--height` | Output dimensions; multiples of 8 |
| `--seed INT` | Reproducible starting seed |
| `--max-prompts INT` | Limit prompts rendered as images |
| `--no-tiled-vae` | Disable memory-saving VAE tiling |
| `--enhance` | Apply SeedVR2 2× enhancement |
| `--enhance-softness FLOAT` | SeedVR2 softness from 0 to 1 |
| `--enhance-after` | Release ERNIE before batch enhancement |
| `--dry-run` | Print grammar without rendering |
| `--serve` | Start the web UI |

Model architecture, quantization, inference steps, and guidance are deliberately not configurable. Generation always uses ERNIE-Image-Turbo q4, 8 inference steps, and guidance 1.0.

## Web UI

```bash
uv run python src/cli.py --serve
```

Open `http://localhost:8000`. The UI provides:

- queued grammar and prompt generation
- direct Tracery grammar import
- prompt-only gallery layouts
- per-prompt or batch image generation
- SeedVR2 enhancement
- live progress through server-sent events
- editable grammar history
- galleries and archived images

## Prompt format

Gemma returns strict Tracery JSON. Expanded ERNIE prompts follow:

1. exact subject
2. concrete description, visible details, and spatial relationships
3. one clear style and medium
4. technical/capture finish: composition, viewpoint, lighting, palette, focus, texture

Structured posters, infographics, interfaces, and comics lead with their overall layout. Each varying Tracery rule normally contains seven distinct alternatives; five or six are accepted when additional choices would reduce quality. Rules with two to four alternatives are rejected.

Visible text is instantiated rather than represented by placeholders. User-provided text remains exact and quoted. Grammar cache keys include the ERNIE prompt-schema version, preventing reuse of obsolete prompt formats.

## Output

```text
generated/
├── grammars/                 # Cached Tracery grammars and raw LM responses
├── prompts/<run-id>/         # Prompt text, metadata, gallery, images, worker log
├── saved/                    # Archived PNG images with embedded metadata
├── queue.json
└── index.html
```

Generation metadata always records:

```json
{
  "model": "ernie-image-turbo",
  "steps": 8,
  "guidance": 1.0,
  "quantize": 4
}
```

## Verification

```bash
uv run ruff check src tests
uv run pytest -q
```

Tests mock LM Studio and mflux. Hardware smoke tests require the provisioned checkpoint and working Metal access.

## Troubleshooting

**LM Studio unreachable** — Start its local server and ensure the base URL is `http://localhost:1234/v1`.

**Gemma not available** — Install/load `google/gemma-4-26b-a4b-qat`. Confirm with `lms ps`.

**Image generation refuses to start** — Confirm `lms` is on `PATH` and `lms unload --all` succeeds.

**ERNIE checkpoint missing** — Run the q4 provisioning command or set `PROMPT_GEN_ERNIE_MODEL_PATH`.

**Out of memory** — Use tiled VAE, reduce dimensions, and enable deferred enhancement.

**Invalid grammar JSON** — Retry with `--no-cache`; inspect the cached raw LM response.

## Credits

- [ERNIE-Image](https://github.com/baidu/ERNIE-Image) by Baidu
- [mflux](https://github.com/filipstrand/mflux) by Filip Strand
- [Tracery](https://github.com/galaxykate/tracery)
