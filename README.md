# Prompt to Image Variations

Local procedural prompt generation and image rendering for FLUX 3 Image Klein on Apple Silicon.

The application asks a local Qwen model in LM Studio to create a FLUX-oriented Tracery grammar, expands that grammar into prompt variations, and optionally renders them with a persistent 4-bit FLUX 3 Image Klein checkpoint through mflux. SeedVR2 2× enhancement is available as a second local stage.

No paid or hosted inference API is used.

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- LM Studio with `qwen3.8-27b-splash`
- LM Studio CLI (`lms`) on `PATH`
- Enough free disk space to download the source checkpoint and save the approximately 5 GB q4 checkpoint

## Installation

```bash
uv sync --group dev
uv sync --extra images --group dev
```

Start the LM Studio local server at `http://localhost:1234/v1`. The configured model ID is:

```text
qwen3.8-27b-splash
```

Override it when necessary:

```bash
export PROMPT_GEN_LM_STUDIO_MODEL=qwen3.8-27b-splash
```

## Provision FLUX 3 Image Klein q4

The application intentionally loads only a previously saved 4-bit checkpoint. Default location:

```text
~/Library/Caches/mflux/models/flux3-klein-q4
```

Accept access terms for [the distilled model](https://huggingface.co/black-forest-labs/flux-3-image-klein), then log in locally. Never share the token in chat. Create the checkpoint once with mflux 0.24.0:

```bash
mkdir -p ~/Library/Caches/mflux/models
uv run hf auth login
lms unload --all
uv run mflux-save \
  --model flux3-image-klein \
  --quantize 4 \
  --path ~/Library/Caches/mflux/models/flux3-klein-q4
```

Custom location:

```bash
export PROMPT_GEN_IMAGE_MODEL_PATH=/absolute/path/to/flux3-klein-q4
```

After verifying generation, the larger source checkpoint in the Hugging Face cache may be removed manually. Do not remove the saved q4 directory.

## Memory handoff

Grammar inference and image inference do not coexist in unified memory. Immediately before loading FLUX 3 or SeedVR2, the application runs:

```bash
lms unload --all
```

Image generation fails closed if the CLI is missing, times out, or cannot unload the models. FLUX is released before SeedVR2; image models are released before Qwen. Before an uncached grammar request, the application checks LM Studio's native model inventory and synchronously reloads Qwen by its exact model ID with an 8192-token context. Transient load failures have at most three attempts; invalid configuration is not retried.

Within a batch, FLUX weights stay loaded but only the current prompt's embeddings are retained. Failed inference/image saves clear those embeddings. Other LM Studio clients can still reload Qwen during image work: unloading is not an exclusive memory reservation.

Generation and enhancement stream `[image-memory]` JSON records to stderr before and after each image (or `failed` on an ordinary exception). Records contain the output filename and MLX `active_bytes`, `cache_bytes`, and `process_peak_bytes`, without prompt text. The peak is the process-wide high-water mark, not a reset per-image peak; these are MLX allocations, not total system memory. Native aborts cannot emit the final snapshot. Web workers forward these records to the live log stream.

## CLI

Generate prompts:

```bash
uv run python src/cli.py \
  --prompt "a barn owl on a mossy branch" \
  --count 20
```

Generate prompts and FLUX 3 images:

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
| `--width`, `--height` | Output dimensions; multiples of 16 |
| `--seed INT` | Reproducible starting seed |
| `--max-prompts INT` | Limit prompts rendered as images |
| `--no-enhance-tiled-vae` | Disable SeedVR2 tiling; FLUX uses its built-in strip decoder |
| `--enhance` | Apply SeedVR2 2× enhancement |
| `--enhance-softness FLOAT` | SeedVR2 softness from 0 to 1 |
| `--enhance-after` | Release FLUX 3 before batch enhancement |
| `--dry-run` | Print grammar without rendering |
| `--serve` | Start the web UI |

Model architecture, quantization, inference steps, and guidance are deliberately not configurable. Generation always uses distilled FLUX 3 Image Klein q4, 16 inference steps, and no CFG. Dimensions must be 64–4096, divisible by 16. Legacy/unknown-model runs cannot be rendered in place; import their grammar into a new run instead. Historical galleries remain readable.

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

Qwen returns strict Tracery JSON. Each expansion is a complete continuous caption, following [BFL's guide](https://docs.bfl.ai/guides/prompting_unified_building) and [upsampler instructions](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json):

1. Medium/style first; explicit user constraints stay fixed. Unspecified medium and style vary too—there is no automatic photography default.
2. Coherent scene, lighting and framing, then concrete subject details and spatial relationships.
3. Exact quoted visible wording; no quality-tag padding. Typically 100–350 words, shorter for simple subjects.

Structured posters, infographics, interfaces, and comics describe reading order and text placement in prose. Target canvas is provided to Qwen. Each generated varying rule targets seven distinct alternatives, with five as the minimum. Fixed rules may have one. Imported grammars may use fewer choices. Modifiers, actions, missing references, and cycles are rejected; literal `#`, `[` and `]` must be escaped with a backslash (doubled in JSON). Structured bounding-box layouts are not implemented.

Qwen treats the input as a creative brief, not an exhaustive list of allowed details. Relevant unspecified traits—appearance, clothing, pose, materials, setting, light, palette, composition and typography—are creative axes. Open briefs usually use 2–5 meaningful variation groups, within eight rules total; dependent choices stay bundled to avoid contradictory combinations. A specified photograph stays photographic, but an unspecified medium can explore photography, painting, illustration, print, collage and renders.

Cache keys include schema, template contents, prompt model, canvas, temperature and input, so edited instructions automatically select a new cache key. Cache hits are revalidated. Invalid fresh output receives one repair attempt before a clear error. The example in `templates/system_prompt.txt` keeps its requested ceramic canister and exact headline fixed, with seven alternatives each for medium/treatment, layout/viewpoint/typography, vessel appearance, and setting/light.

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
  "model": "flux3-image-klein",
  "steps": 16,
  "guidance": null,
  "quantize": 4
}
```

Run metadata also records the checkpoint path, mflux version, distilled variant, prompt schema/model, and source revision when a checkpoint provenance manifest is available (otherwise `null`, never guessed). Each rendered PNG has a `.generation.json` sidecar with its actual seed, exact caption and original dimensions. Enhancement leaves that generation record intact.

## Verification

```bash
uv run ruff check src tests
uv run pytest -q
```

Tests mock LM Studio and mflux. Hardware smoke tests require the provisioned checkpoint and working Metal access.

```bash
uv run python scripts/verify_flux3.py --mode corpus
uv run python scripts/verify_flux3.py --mode e2e
```

The corpus generates 12 requests twice; inspect its saved samples for semantic coherence. E2E runs two uncached Qwen → FLUX q4 → SeedVR2 cycles at 864×1152, recording dimensions, memory and LM Studio state. Outputs are in `generated/verification/`. These checks are opt-in and use local inference. Worker heartbeats keep long tasks observable; `PROMPT_GEN_WORKER_TIMEOUT` controls process-polling intervals, not a generation deadline. Web and CLI batches have no overall time limit by default. Cancel stuck work from the queue; heartbeats do not prove useful inference progress. Callers that explicitly supply a finite `Worker(task_timeout=...)` still get a deadline, reported as a timeout failure rather than user cancellation.

Worker failures retain the exit code/signal, any reported task error, and a bounded stderr tail (20 entries, up to 1024 characters each). Progress output, oversized lines and invalid UTF-8 cannot hide the final native error. Process termination and stderr-reader cleanup use bounded waits.

Measured results and remaining acceptance limits: [verification report](docs/flux3-verification.md).

## Troubleshooting

**LM Studio unreachable** — Start its local server and ensure the base URL is `http://localhost:1234/v1`.

**Qwen not available** — Install/load `qwen3.8-27b-splash`. Confirm with `lms ps`.

**Image generation refuses to start** — Confirm `lms` is on `PATH` and `lms unload --all` succeeds.

**FLUX 3 checkpoint missing** — Run the q4 provisioning command or set `PROMPT_GEN_IMAGE_MODEL_PATH`.

**Out of memory** — Reduce dimensions and enable deferred enhancement. SeedVR2 tiling affects enhancement only.

**Leaked semaphore warning** — Indicates shutdown cleanup, not proof of a GPU memory leak. Inspect the task's exit code/signal, recent stderr and `[image-memory]` records. Also check whether another LM Studio client reloaded Qwen during image work.

**Invalid grammar JSON** — Retry with `--no-cache`; inspect the cached raw LM response.

## Credits

- [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) by Black Forest Labs
- [mflux](https://github.com/filipstrand/mflux) by Filip Strand
- [Tracery](https://github.com/galaxykate/tracery)
