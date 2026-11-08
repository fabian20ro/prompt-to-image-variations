# Lessons Learned

> This file is maintained by AI agents working on this project.
> It captures validated, reusable insights discovered during development.
> **Read this file at the start of every task. Update it at the end of every iteration.**

## How to Use This File

### Reading (Start of Every Task)
Before starting any work, read this file to avoid repeating known mistakes
and to leverage proven approaches.

### Writing (End of Every Iteration)
After completing a task or iteration, evaluate whether any new insight was
gained that would be valuable for future sessions. If yes, add it to the
appropriate category below.

### Promotion from Iteration Log
Patterns that appear 2+ times in `ITERATION_LOG.md` should be promoted
here as a validated lesson.

### Pruning
If a lesson becomes obsolete (e.g., a dependency was removed, an API changed),
move it to the Archive section at the bottom with a date and reason.

---

## Architecture & Design Decisions

**[2026-02-15]** FastAPI lifespan + static mounting - In apps already using lifespan hooks, mount static files directly with `check_dir=False` instead of `@app.on_event("startup")` to avoid deprecation warnings and keep startup logic unified.

## Code Patterns & Pitfalls

**[2026-11-07]** Separate requested constraints from invented defaults in prompt examples — Fixing an unrequested medium/viewpoint/palette in origin removes those axes from every expansion. Keep relevant unspecified traits variable, including medium/style per user preference; bundle dependencies but leave independent groups recombinable. Example combination checks prove structural variation, not live LLM adherence or image quality.

**[2026-11-07]** Resolve API-version conflicts against the pinned runtime — mflux 0.24.0 SeedVR2 accepts `resolution`, not separate `width`/`height` kwargs. Preserve the fixed 2× contract when merging older generic controls. Exercise real Click progress bars at the CLI seam: Click's `ProgressBar` has no `write` method; emit status with `click.echo`.

**[2026-11-07]** Match Tracery escape semantics once — A backslash escapes the next character, including another backslash. Regexes that merely ignore delimiters preceded by a backslash can admit active actions or miss cycles. Share one escape-aware reference graph; protect quoted literals in one longest-match pass so repeated/already-escaped text is not escaped again. Public grammar/expansion regressions cover both.

**[2026-11-07]** Validate before mutating run artifacts — Grammar regeneration must expand/validate before archiving or deleting PNGs. Creation cannot overwrite existing run captions while skipping existing images; resume through the saved-prompt path. Public-seam regression tests cover both.
**[2026-11-07]** Preserve valid JSON before cleaning LLM wrappers — Curly apostrophes, literal `<think>` text, and backticks can be image content. Parse valid JSON before wrapper removal/quote repair, and extract valid prefixed JSON before normalizing delimiters.
**[2026-11-07]** FLUX 3 mflux 0.24.0 decode uses built-in strips — Its old tiling setting does not control text-to-image decode. Keep the exposed tiling control specific to SeedVR2; do not cache two FLUX models by a meaningless boolean.

**[2026-11-06]** Serialize layout arrays after Tracery expansion — Literal JSON arrays inside grammar strings are parsed as Tracery actions and disappear. Keep structured layout data separate and append serialized JSON after expanding text fields; verified with a FLUX bounding-box prompt.

**[2026-02-07]** Validate model names before importing `mflux` - Importing `mflux` before rejecting unsupported models can trigger native MLX aborts in unsupported test paths.
**[2026-02-07]** Keep docs and signatures aligned - Drift between documented defaults and implementation defaults (`tiled_vae`) causes repeated operator mistake
**[2026-02-24]** Prefer MetadataManager for metadata operations — Direct JSON manipulation of run metadata bypasses validation and path logic; use MetadataManager consistently.s.
**[2026-05-11]** Preserve explicit zero-value gallery layouts — `images_per_prompt: 0` is a real prompt-only layout, so normalize `None` separately instead of coercing all falsy values to 1.
**[2026-05-15]** `uv run` may warn about a mismatched `VIRTUAL_ENV` under Hermes — the project command still uses the repo environment and `uv run pytest --collect-only -q` completed successfully with the warning present.

## Testing & Quality

**[2026-05-13]** Re-verify environment-specific failure notes before preserving them — stale caveats about native imports or toolchain aborts can outlive the real issue, so rerun the narrow test and update the note to the current observed state.
**[2026-02-07]** Prefer intent assertions for worker logs - Worker success/failure paths emit variable extra lines, so tests should assert sequencing/intent rather than exact call counts.
**[2026-02-07]** Use module fakes for MLX-heavy imports - Patching deep `mflux.*` paths can still initialize native MLX/Metal; prefer `patch.dict(sys.modules, ...)` with fake modules.
**[2026-02-24]** Mock external services in tests — Always mock LM Studio and mflux; requiring live GPU or inference server makes tests environment-dependent and fragile.
**[2026-05-14]** Click help output can wrap long option descriptions — when asserting `--help` text, prefer stable substrings over exact single-line matches for defaults embedded in long help strings.

## Performance & Infrastructure

**[2026-11-08]** Distinguish batch deadline from memory failure — The former 1800-second worker deadline terminated healthy batches and reproduced SIGTERM plus Python's leaked-semaphore shutdown warning in a CPU-only probe. Inner cancellation also hid timeout failures as cancelled/no-error. The worker now has no default overall deadline; its outer execution layer owns explicit deadline/shutdown status, separately from bounded child cleanup. The warning alone is not evidence of a GPU memory leak; the liveness setting is a polling interval, not a task deadline.

**[2026-11-08]** Bound embeddings separately from model weights — Installed mflux 0.24.0 `_Flux3KleinCore._encode_prompt` stores every unique prompt's tensors in an unbounded `prompt_cache`. The renderer now retains only the current prompt while reusing weights; failed inference/image saves clear embeddings. CPU weak-reference probes confirm reference release, not a native memory plateau or crash causation. Allocator-cache clearing alone cannot release live references. MLX diagnostic peaks stay process-wide so per-image logging does not reset wider hardware acceptance measurements.

**[2026-11-08]** Observe cleanup before test teardown — A harness calling `Worker.stop()` after a failure event can hide an orphaned image child by terminating it on the test's behalf. Capture child state at the real queue failure notification. Public lifecycle regressions now cover this ordering, timeout versus cancellation, native crash tails and bounded process/pipe cleanup.

**[2026-11-07]** LM Studio unload is not an exclusive memory lease — Other CLI clients can reload a model during multi-minute image work. Verify native inventory at lifecycle boundaries and pause competing clients for hardware acceptance; do not weaken unload checks or kill unrelated clients to make tests pass. Two uninterrupted FLUX/SeedVR2 cycles passed after the competing reloads ceased.

**[2026-02-07]** Open worker logs before long tasks - If log files are opened after pipeline execution starts, early progress information is permanently lost.

## Dependencies & External Services

**[2026-06-20]** mflux caches model families separately — Generated-model weights use `~/.cache/huggingface/hub`, while downloaded LoRAs use `~/Library/Caches/mflux/loras`; inspect both when retiring a model family.
**[2026-06-20]** Use LM Studio's native chat API for Gemma reasoning control — `/api/v1/chat` with `reasoning: "off"` suppresses reasoning output reliably; the OpenAI-compatible endpoint ignored `chat_template_kwargs.enable_thinking` for `google/gemma-4-26b-a4b-qat`.
**[2026-06-20]** Explicitly load LM Studio models after image-memory handoff — just-in-time chat can race a recent unload and return “model has not started loading/has been unloaded”; query `/api/v1/models`, synchronously call `/api/v1/models/load`, and retry transient load cancellation before chat.
**[2026-04-03]** Use `tool.uv.package = false` for this repo’s flat imports — The code and tests rely on `src/` being executed directly (`uv run python src/cli.py`, pytest path injection). During `uv` migration, keep the repo non-packaged instead of switching to module entrypoints unless imports are refactored first.
**[2026-05-08]** When uv wheel extraction fails with `Operation not permitted` in the default cache, point `UV_CACHE_DIR` at a writable local path like `/tmp/uv-cache` before retrying.

## Process & Workflow

**[2026-11-08]** Always name the remote destination when publishing a fresh branch — This repo uses `push.default=upstream`; a branch created from `origin/master` inherits master as its upstream. A source-only `git push -u origin <branch>` unexpectedly updated master. Create review branches with `--no-track`, publish with an explicit refspec such as `git push -u origin HEAD:refs/heads/<branch>`, then verify the remote ref and tracking destination before creating a PR. Repair an accidental master update with a user-approved ordinary revert, not a history rewrite; reapply the fixes after that revert so the PR has a real diff.

**[2026-02-07]** Emit queue state updates after clear - Emit `queue_updated` together with `queue_cleared` so existing SSE clients refresh immediately.
**[2026-02-07]** Prefer non-blocking UI confirmations - Replacing `alert/confirm` with toast/modal flows reduces disruptive UI pauses and improves task continuity.

---

## Archive

**[2026-05-14] Archived 2026-06-20** Surface shared quantize behavior in docs/help — Obsolete after the ERNIE-only migration removed public quantization controls and fixed ERNIE q4 plus SeedVR2 q8 internally.

<!-- Lessons that are no longer applicable. Keep for historical context. -->
<!-- Format: **[YYYY-MM-DD] Archived [YYYY-MM-DD]** Title — Reason for archival -->
