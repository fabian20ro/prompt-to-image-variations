# FLUX 3 Image Klein and Qwen migration plan

Date: 2026-11-06. Implementation update: 2026-11-07 — caption-only migration implemented with seam-first TDD; merged master and verified 1118 tests on Python 3.11/3.13 (375 before merge), with browser grammar workflows previously verified. Saved FLUX q4 passes fresh-process offline rendering; two consecutive Qwen → FLUX → SeedVR2 cycles passed with memory released and LM Studio empty before the master merge. Approved source cache removed (18.9 GiB); saved q4 and SeedVR2 retained. Tuned live Qwen corpus accepts 23/24 requests (one safely rejected after repair). See [verification evidence](flux3-verification.md). ERNIE checkpoint removed at the user's explicit request. Optional structured layout remains deferred. Sections below retain the original planning baseline.

## Current status

- Branch: `codex/add-ernie-image-support`; clean worktree at audit start. Latest commit: `2e4a432`, ERNIE grammar rules.
- Fresh verification: 361 tests pass; configured Ruff checks pass. These tests mock inference; they do not prove today's hardware pipeline.
- Application defaults still select `google/gemma-4-26b-a4b-qat`, ERNIE Turbo q4, 8 steps, guidance 1.0, and prompt schema `ernie-v2`.
- LM Studio is reachable. Native inventory and CLI both identify the loaded model as `qwen3.8-27b-splash`, sourced from `incoai/Qwen3.8-27B-Splash`, 4-bit, approximately 17.4 GB on disk. Native API advertises reasoning `off` and `on`. Loaded context is 61,696 tokens. The configured Gemma ID is absent from inventory, so an uncached grammar request with current defaults would fail its model lookup.
- Persistent ERNIE q4 bundle remains present, approximately 6.2 GB. No FLUX 3 bundle in the app's model cache. Approximately 85 GiB free on the workspace volume.
- Locked mflux: 0.18.0. Current published release: 0.24.0, requiring Python >=3.11. Project declares >=3.10; current environment is Python 3.12.11.
- Keep: Tracery, galleries, seed/dimensions, prompt-only workflows, resume within matching runs, SeedVR2, local-only inference, mandatory `lms unload --all` before image operations.

## Verified upstream contract

Target: **FLUX 3 Image Klein distilled**, `black-forest-labs/flux-3-image-klein`; mflux alias `flux3-image-klein`. Distilled sampling uses 16 steps without CFG. The model uses its own Qwen3-VL-4B text encoder; that is separate from the LM Studio Qwen prompt writer. The model card recommends a prompt upsampler and explicitly permits a chosen LLM with appropriate instructions. Qwen can perform that job through detailed Tracery generation. [Model card](https://huggingface.co/black-forest-labs/flux-3-image-klein)

Current mflux documents q4 persistence, dimensions divisible by 16, `Flux3Klein`, and a generated-image wrapper. Main-branch decode behavior now uses row strips; its old tiling option does not affect text-to-image decoding. Confirm these details against the exact installed release before wiring the app. [mflux implementation guide](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux3/README.md), [0.24.0 release](https://pypi.org/project/mflux/0.24.0/)

Weights are gated. Establish existing account access and license acceptance before provisioning; if access is missing, stop that phase for the user's account action. No hosted inference required.

## Proposed decisions

1. One image backend: distilled FLUX 3 Image Klein, saved MLX q4. Replace ERNIE rather than reintroducing model selection or compatibility dispatch.
2. One configured prompt writer: exact LM Studio key `qwen3.8-27b-splash`. Keep model inventory/load/chat requests on localhost.
3. Keep generation parameters internal: 16 steps; omit guidance for distilled inference. Metadata must say CFG is unused, not inherit ERNIE's 1.0.
4. Use Qwen once to produce a reusable grammar of fully described scenes. Each expansion must already be a complete caption; avoid a second rewrite that could alter locked content.
5. Seven alternatives per variable rule as the target, five as the minimum. Fixed rules may have one. Do not pad choices with synonyms or alter requested content to manufacture variation.
6. Default to caption-only text-to-image. Add layout support as a separately verified extension described below. Reference editing, video, LoRAs, and Base-model support are outside this migration.
7. No historical metadata rewrites. Existing galleries remain artifacts; legacy ERNIE runs cannot silently resume with FLUX and become mixed-model runs.
8. No deletion during migration. After acceptance, prepare an exact ERNIE/source-cache cleanup list; deletion needs explicit scope approval. Earlier authorization covered retired Z-Image/FLUX.2 caches, not this new retirement.

## Phase 1 — Prove the runtime before migrating

Files: `pyproject.toml`, `uv.lock`, `src/image_generator.py` prototype boundary.

- Select and pin the tested mflux release; start with 0.24.0. Raise declared Python minimum to 3.11 and resolve the images/dev environment. Do not depend on a floating Git main branch.
- Inspect that release's constructors, quantized loader, saving, precision, dimension checks, VAE options, and SeedVR2 API. Resolve any difference from current online documentation explicitly.
- Check gated model access without exposing tokens. Determine source-download plus saved-bundle disk needs and actual machine memory before loading.
- Unload LM Studio before provisioning/loading image weights. Save q4 to `~/Library/Caches/mflux/models/flux3-klein-q4` using the released `mflux-save` command.
- Confirm saved quantization metadata and required components; reload from that bundle in a fresh process and render one fixed caption. Directory existence alone is insufficient proof of q4.
- Verify SeedVR2 still imports, renders, and saves under the new dependency set.

Gate: persistent local q4 image generation and SeedVR2 work with the pinned release. Record actual latency and peak memory; do not substitute upstream hardware benchmark numbers.

## Phase 2 — Replace the prompt-model contract

Files: `src/config.py`, `src/grammar_generator.py`, `tests/test_grammar_generator.py`.

- Change the default to the exact inventory key. Remove Gemma-specific comments and assumptions.
- Retain synchronous native model loading and bounded retries, but retry transient conditions only. Surface invalid requests, missing models, and server errors with useful response details.
- Smoke-test native chat with `reasoning: off`, `store: false`, and bounded output. Validate actual reasoning suppression and complete JSON with Splash; advertised capability alone is insufficient.
- Establish a modest context budget from measured template + request + output length. Do not inherit the currently loaded 61K context as an application requirement.
- Tune sampling with repeated grammar requests, not a presumed Gemma-compatible preset. Increase output allowance only if measured valid grammars truncate.
- Centralize server URL handling so CLI dry-run and normal generation use the same supplied URL; currently the explicit CLI URL is passed directly only to dry-run grammar generation.

Gate: repeated uncached grammars from the exact Qwen ID; a later explicit lifecycle test proves reload after unload. No fallback to arbitrary loaded models or external APIs.

## Phase 3 — Rebuild prompt instructions and examples

Files: `templates/system_prompt.txt`, `tests/test_system_prompt.py`, prompt-quality fixtures.

Use BFL's current [prompt guide](https://docs.bfl.ai/guides/prompting_unified_building) and the actual [text-to-image upsampler instructions](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json), rather than replacing ERNIE's name in its template.

- Expanded output: one continuous caption with medium/style early, followed by a coherent scene, light/framing, and concrete subject details. Preserve user priorities rather than enforcing a rigid word order.
- Use explicit spatial relationships, exact counts, and visible materials. Preserve quoted wording and original scripts; remove quality-tag padding. Length follows scene complexity, with BFL's upsampler range of roughly 100–350 words as a starting target, not forced padding.
- Pass target canvas information to Qwen so composition matches dimensions; do not print canvas instructions in the final image caption.
- Keep a chosen medium fixed across variations unless style exploration was requested. Group dependent choices—lighting/weather, framing/layout, material/finish—into coherent alternatives.
- Replace the current example, which independently varies medium and incompatible camera/lighting details. Provide a complete seven-choice photography example and a text/layout example with locked wording.
- Reuse the detailed grammar as local prompt enhancement. Store raw response, validated grammar, and exact expanded prompt for inspection.

Gate: inspect expansions, not just source JSON. Sample combinations across each variable; all must preserve subject, count, medium, visible text, and a coherent composition.

## Phase 4 — Validate grammars and isolate caches

Files: `src/grammar_generator.py`, `src/tracery_runner.py`, focused tests.

- New prompt-schema namespace, e.g. `flux3-caption-v1`. Cache identity includes prompt-model ID, template/schema version, canvas, and relevant sampling settings; current keys include only schema plus user prompt.
- Validate cache hits as well as new output. Invalid grammars never enter the cache.
- Require variation reachable from `origin`; currently an unused variable rule can satisfy the variation check.
- Detect unresolved references and cycles; explicitly support or reject modifiers/actions rather than silently missing them with the current simple regex.
- Preserve literal visible text. Audit curly-quote normalization, literal hash characters, and bracket handling; a cleanup pass must not mutate requested typography or content.
- Provide one bounded repair request with validation feedback for malformed/underpopulated output, then a clear error. No endless regeneration loops.
- Apply shared validation to generated, pasted, and edited grammars with clear rules for user-authored fixed prompts.

Gate: test missing references, cycles, unreachable variation, duplicate choices, 2–4 choices, literal punctuation, truncated output, repair exhaustion, and stale caches.

## Phase 5 — Replace renderer and make memory ownership explicit

Files: `src/image_generator.py`, `src/image_enhancer.py`, `src/lm_studio.py`.

- Replace `ErnieImage` with the pinned release's `Flux3Klein` and explicit distilled config. Load only the verified local q4 bundle.
- Use 16 steps, no guidance argument, dimensions divisible by 16, and the correct generated-image saving API. Existing 864×1152 defaults already satisfy divisibility.
- Use a neutral image-model-path setting; remove `PROMPT_GEN_ERNIE_MODEL_PATH`.
- Retain fail-closed `lms unload --all` before every generation/enhancement call, including cache hits.
- Release FLUX weights before SeedVR2; release SeedVR2 before loading FLUX or Qwen again. Current deferred enhancement clears generator cache, but immediate enhancement can retain both models.
- Prefer generate-all then enhance-all for batches. Ensure immediate/standalone paths have equally explicit release boundaries and cancellation cleanup.
- Map VAE controls to real capabilities of the pinned implementation. If FLUX decode has no meaningful tiling switch, remove that generator control; retain SeedVR2 tiling where effective. Do not assign an unused `tiling_config` attribute and imply it works.

Gate: tests cover call order and unload errors; hardware tests cover reloads and multiple images without accumulating model memory.

## Phase 6 — Integrate metadata, workers, and UI

Files grouped by concern: pipeline/metadata; server models/routes/worker; CLI/gallery/index.

- One source of truth for fixed model facts. Record image model/revision, quantization, steps, no-CFG status, mflux version, prompt model/schema, dimensions, seed, and enhancement settings.
- Reject invalid dimensions consistently in CLI/API; reflect increments of 16 in UI.
- Preserve prompt-only layouts, saved grammars, seed, resume, galleries, and enhancement controls. Keep model/steps/guidance/quantization selectors absent.
- Refuse renderer-mismatched resume before writing files. Offer a new run from old prompt text; never relabel existing ERNIE images as FLUX.
- Check worker timeouts against measured cold load + grammar + render + enhancement latency. Distinguish model loading from denoising in progress reporting.
- Update README, installed-command examples, configuration docs, and code maps. Archive obsolete lessons; retain historical journal entries.

Gate: CLI and browser exercise the same contracts, old galleries remain readable, and rejected legacy resumes leave original files unchanged.

## Phase 7 — Optional FLUX layout support

Files: prompt artifact schema, renderer/serializer, layout fixtures; integrate into the same validation path.

BFL supports caption element IDs plus a JSON table with normalized boxes. [Layout guide](https://docs.bfl.ai/guides/prompting_layout)

Confirmed locally: feeding a literal box array through today's Tracery runner removes the array. Therefore do not embed raw JSON arrays inside ordinary grammar strings.

- Represent layout as structured companion data. Expand caption/element-description strings, then serialize the final JSON array after Tracery expansion.
- Validate unique IDs, correspondence with caption tokens, integer bounds 0–1000, ordered coordinates `[top,left,bottom,right]`, text content, and aspect-ratio-aware geometry.
- Keep each layout variation coherent as a whole; independently randomizing coordinates and captions is invalid.
- Preserve a single final prompt artifact containing exactly what FLUX receives. Show readable caption/layout data in the gallery without requiring users to edit raw serialization.

Gate: raw final prompt round-trips, boxes survive serialization, and rendered typography/layout matches the selected plan. Caption-only migration can ship first; this phase must not be represented as complete merely by mentioning boxes in the system prompt.

## Acceptance and rollout

1. Automated checks: full pytest suite, configured Ruff, dependency consistency, no live API/GPU requirements in unit tests. Add focused behavioral tests where contracts change.
2. Prompt corpus: at least 12 requests across portraits, products, landscapes, illustration, exact-text posters, multilingual text, multiple subjects, locked styles, and constrained layouts. Generate twice per request with cache bypass; inspect sampled expansions and record repair/error rates.
3. Two consecutive end-to-end runs from unloaded LM Studio: load Qwen → grammar → expand → unload all LMS models → local FLUX q4 render → release FLUX → SeedVR2 → release enhancer → repeat. Verify metadata, image dimensions, actual process state, and visual outputs.
4. Exercise CLI plus browser/queue paths, prompt-only generation, regenerate-one, resume, standalone enhancement, cancellation, and missing-checkpoint/unload-failure cases.
5. Use representative 1MP images as well as a small smoke render. Confirm detailed text and compositions do not regress silently at q4. If q4 is inadequate, report the measured tradeoff before changing the requested quantization.
6. Roll out by coherent commits: runtime, Qwen lifecycle, prompts/validation, renderer, integrations/docs. Retain previous commit and ERNIE bundle until acceptance; rollback is code/dependency restoration, not runtime multi-model compatibility.

## Decisions still open

- Recommended initial scope: caption-only generation, followed by optional structured layouts. Reference editing is a separate feature.
- Assumed carried-forward preferences: local q4, SeedVR2, no paid inference, no backward-compatibility layer, and minimal controls.
- User action only if gated weights are inaccessible; checkpoint cleanup requires a separately reviewed exact-path list.
