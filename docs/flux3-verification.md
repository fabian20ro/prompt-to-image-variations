# FLUX 3 / Qwen verification

Date: 2026-11-07. Machine: Apple Silicon, 36 GiB unified memory.

Status: **hardware acceptance passed; approved FLUX source-cache cleanup completed**. Prompt-generation caveat remains: 23/24 corpus requests accepted, one rejected after bounded repair.

## Implemented contract

- Qwen prompt writer: local LM Studio key `qwen3.8-27b-splash`, reasoning off, 8192-token context. No paid inference APIs.
- Caption-only Tracery grammar: seven alternatives preferred, five minimum; exact quoted text preserved; one bounded repair before a clear validation error.
- Renderer: mflux 0.24.0, FLUX 3 Image Klein distilled, saved q4, 16 steps, no CFG. Dimensions divisible by 16.
- Saved checkpoint: `~/Library/Caches/mflux/models/flux3-klein-q4`.
- SeedVR2 retained at q8, 2× enhancement; VAE tiling control applies only to enhancement.
- LM Studio unload before image operations; image-model references and MLX allocator caches released at model handoffs.
- Per-image `.generation.json` records actual seed, prompt, dimensions, checkpoint and source revision. Historical runs remain readable but cannot silently resume with another renderer.

## Automated and browser checks

Matt Pocock TDD workflow: red→green vertical slices at the user-approved grammar/expansion, image/enhancement, pipeline/artifact and CLI/API seams. External inference/process boundaries mocked; hardware checks separate.

- Latest merged suite: **1118 passed** on Python **3.11.16** and **3.13.15**, each in a clean `uv sync --locked --group dev` environment without optional image packages. One upstream Starlette/AnyIO deprecation warning. The pre-merge migration suite had 375 passes.
- `uv run ruff check src tests scripts/verify_flux3.py`: passed.
- `uv pip check`: 72 packages compatible. `git diff --check`: passed.
- Repository-wide `ruff check .` also visits pre-existing scratch scripts (`test_new_logic.py`, `test_scratch.py`) containing Python 3.12-only f-string syntax; those are outside the configured CI lint scope and remain unchanged.
- Browser: fixed grammar import, cyclic edit rejection without destructive save, successful caption regeneration, prompt-only gallery showing 0/0 images. Evidence run: `generated/prompts/20261107_002605_50da405c4032`.

Final two-axis code review found one Standards issue (inconsistent escape/reference parsing) and one Spec issue (repeated quoted literals double-escaped). Both reproduced at public seams, fixed red→green, and independently rechecked. The validator now shares one escape-aware reference graph; literal protection uses one longest-match pass, preserving already-escaped and overlapping wording. All 23 accepted corpus grammars and both hardware-run grammars were replayed successfully after these fixes. GPU checks above/below used the preceding grammar implementation; renderer and model lifecycle code were unchanged by review.

Merge with `master` (`74adfd0`): retained seeded Tracery expansion, CLI listing/JSON/config features, configuration/path validation, worker/gallery fixes and compatible new coverage. Kept fixed 2× SeedVR2: the pinned runtime does not accept master's width/height overrides. Retired only superseded ERNIE/private-cache, dead pipeline-config and unsupported enhancer-dimension tests; adapted remaining fixtures to FLUX validation. A real-progressbar CLI regression exposed `ProgressBar.write` (unsupported by Click), fixed test-first to `click.echo`. Lock consistency, Ruff and whitespace checks pass; saved corpus/hardware grammars replay after the merge. Real GPU inference was not repeated for this merge; hardware timings below remain the earlier observations.

## Real-model evidence

### Saved q4

Fresh-process 512×512 teapot render: **25.04 s**, **5.875 GB peak MLX allocation**. Saved bundle: **5,333,800,130 bytes**, approximately 5.0 GiB.

Source: `black-forest-labs/flux-3-image-klein`, revision `8454017f9ebaada30c998fccd9f67a80c7580f68`. Checkpoint component indexes and shard availability validated before inference; quantization metadata is q4.

Artifacts: `generated/verification/flux3-20261107/checkpoint.{png,json}`. Bundle `provenance.json` records the source revision.

### Qwen corpus

Twelve requests, twice each; seven expanded samples per successful grammar. Coverage includes photography, watercolor, vector art, explicit counts/positions, a two-panel comic, French/Japanese text and literal `Batch #1 [OPEN]` punctuation.

Final result: **23/24 accepted**. All varying rules in accepted grammars have seven alternatives. Exact requested text checks passed for accepted cases. One portrait response still supplied four `origin` alternatives after repair; correctly rejected, not cached. Do not treat this as a guarantee of semantic correctness: captions were manually sampled, not exhaustively image-scored.

Baseline was 22/24. Literal-text protection and the revised coherent scene example fixed both punctuation-label cases. An intermediate 17-case tuning run was intentionally interrupted; not a completed corpus.

Artifacts: `generated/verification/flux3-20261107-final/corpus.json` (final), `flux3-20261107/corpus.json` (baseline), `flux3-20261107-tuned/corpus.json` (partial).

### Consecutive pipeline checks

One complete cold-Qwen → FLUX 864×1152 → SeedVR2 1728×2304 run succeeded in **350.35 s**, **15.926 GB peak MLX allocation**. After release, active and cached MLX memory both measured **0.0 GB**. The enhanced poster was visually inspected; headline remained readable. Separate SeedVR2 512→1024 smoke test passed in **37.24 s**, peak **15.457 GB**.

The initial consecutive-cycle attempt stopped correctly because LM Studio was reloaded by run end. `lms ps` reported Qwen processing a prompt at **61,696-token context**, unlike this application's requested 8192. Server logs show a later `lms-cli` load. After the user paused other clients, one retry still encountered leftover reloads (214.77 s, successful image but failed final inventory check). Following another unload, the interference ceased. No unrelated clients stopped and no acceptance checks weakened.

**Final consecutive-cycle gate passed**, in one process with fresh Qwen grammar generation each cycle:

- Seed 42: **157.29 s**, 1728×2304 output, **15.926 GB** peak MLX allocation.
- Seed 43: **290.63 s**, 1728×2304 output, **15.926 GB** peak MLX allocation.
- Both: LM Studio inventory empty at completion; active and cached MLX allocation each rounded to **0.0000 GB** after release. Metadata records q4, 16 steps, no CFG, actual seed, prompt model and source revision.
- Both enhanced posters visually inspected: readable headline, coherent canister composition. This does not imply perfect adherence to every generated material/count detail.

Passing evidence: `generated/verification/flux3-20261107-final-gate/e2e.json` and `e2e-{0,1}/image_0_0.png`. Earlier failed lifecycle attempts remain in `flux3-20261107-accepted/` and `flux3-20261107-uninterrupted/`; their directory names do not imply acceptance.

### Offline isolation

Fresh-process 512×512 q4 render passed in **19.39 s**, **5.875 GB** peak MLX allocation, with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_CACHE=/tmp/flux3-offline-verification-20261107` pointing to an absent/empty cache. Saved q4 contains no symlinks. No source-download fallback needed. Output: `generated/verification/flux3-20261107-uninterrupted/offline.png`.

Cold SeedVR2 provisioning downloaded 6.8 GiB from `numz/SeedVR2_comfyUI`, revision `09ced71023636e9bc8cdf9cdecfb2625d1e691e8`. During the earlier attempt, Qwen was observed loaded again; manual `lms unload --all` restored free memory from 14% to 62%. That stalled process was terminated before the successful retry. Its original poster remains in `flux3-20261107-final/e2e-0/`.

## Cleanup and limits

ERNIE q4 was removed at the user's request. After hardware and offline verification, deleted only the explicitly approved `~/.cache/huggingface/hub/models--black-forest-labs--flux-3-image-klein`. Confirmed absent; removed **19,844,716 KiB (18.9 GiB)** of allocated cache data.

Saved q4 remains at `~/Library/Caches/mflux/models/flux3-klein-q4` (**5,208,824 KiB**). SeedVR2 cache remains at `~/.cache/huggingface/hub/models--numz--SeedVR2_comfyUI` (**7,113,620 KiB**). Both sizes unchanged by cleanup. No other model/cache deletions.

Post-deletion fresh-process offline render also passed: **19.48 s**, 512×512. Output: `generated/verification/flux3-20261107-final-gate/post-cleanup.png`.

Structured bounding-box layout, reference editing, video, LoRAs and Base-model support are outside this caption-only migration. No old-model compatibility adapter or historical metadata rewrite.

Reproduce hardware checks with `scripts/verify_flux3.py --mode checkpoint|corpus|e2e` via `uv run python`. These perform real local inference and are not part of pytest. Avoid loading another LM Studio model during image inference.
