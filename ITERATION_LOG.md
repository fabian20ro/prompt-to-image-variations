# Iteration Log

> Append-only journal of AI agent work sessions on this project.
> **Add an entry at the end of every iteration.**
> When patterns emerge (same issue 2+ times), promote to `LESSONS_LEARNED.md`.

## Format

Each entry should follow this structure:

---

### [YYYY-MM-DD] Brief Description of Work Done

**Context:** What was the goal / what triggered this work
**What happened:** Key actions taken, decisions made
**Outcome:** Result - success, partial, or failure
**Insight:** (optional) What would you tell the next agent about this?
**Promoted to Lessons Learned:** Yes/No

---

### [2026-07-12] Corrected default-branch CI trigger

**Context:** Compound base-health evidence found that this repository's default branch is `master`, while CI push runs targeted `main`.
**What happened:** Changed the CI push trigger to `master`; pull-request coverage remains unchanged.
**Outcome:** Success — ruff passed, all 555 project tests passed, and the parent Hermes stack restarted healthy on 0.18.2.
**Insight:** Base-branch automation must be checked against the repository's actual default branch, not a conventional branch name.
**Promoted to Lessons Learned:** No

---

### [2026-05-15] Added the shared HTML components test file to the test codemap

**Context:** The test codemap and README coverage summary listed most major test surfaces, but they did not call out `tests/test_html_components.py`, which covers the shared CSS/JS building blocks used by the gallery and index pages.
**What happened:** Added `tests/test_html_components.py` to `docs/codemaps/testing.md` and the README's test coverage list so the shared UI component tests are easier to find.
**Outcome:** Success — docs now reflect the full test surface visible in the current checkout.
**Insight:** Small codemap omissions are easiest to catch when the test file list is compared directly against the current collected suite.
**Promoted to Lessons Learned:** No

---

### [2026-04-04] Added revision-safe grammar regeneration, layout persistence, and grammar-import galleries

**Context:** User reported that regenerating prompts after pasting a new grammar kept stale images attached to changed prompts, and that gallery canvas size could drift from selected `images per prompt` / `max prompts`. User also requested grammar undo/history and a way to create a gallery directly from pasted Tracery grammar.
**What happened:** Added persisted `gallery_layout` metadata with fallback for older runs; changed gallery rendering to respect persisted layout instead of hardcoded form defaults; added per-run persisted grammar history (`*_grammar_history.json`) plus gallery UI controls for undo/redo and revision restore; added local draft persistence so layout-triggered reloads do not drop unsaved grammar edits; changed regenerate flow to auto-save posted grammar, archive existing PNGs, delete active PNGs only after successful backup, and rebuild the gallery from the new prompts/layout; added `/api/generate-from-grammar` for grammar-first gallery creation; updated index UI with a grammar-import form; extended worker/task plumbing and route models; added regression tests for layout persistence, history, safe regeneration, and grammar-import queueing.
**Outcome:** Success — targeted suite passed (`158 passed`) and full suite passed (`324 passed`).
**Insight:** If a route both persists a file and records a revision snapshot, revision capture must happen before overwriting the file or the history bootstrap path will misclassify the first saved revision as the current baseline.
**Promoted to Lessons Learned:** No

---

### [2026-04-03] Migrated pip/venv workflow to uv

**Context:** User requested full migration from `pip` + `venv` to `uv`, with the legacy setup removed rather than kept for compatibility.
**What happened:** Added `pyproject.toml` and committed `uv.lock`; moved runtime/dev deps into `uv` metadata; made `mflux` an optional `images` extra; deleted `requirements.txt` and `requirements-dev.txt`; removed the local `venv/` directory; updated `.gitignore` to `.venv/`; rewrote README, AGENTS, codemaps, repo-local Claude skills/settings, runtime error messages, and CLI help text to use `uv sync` / `uv run`; refreshed the README test-count note after verification.
**Outcome:** Success — `uv lock`, `uv sync --group dev`, and `uv sync --group dev --extra images` all completed successfully; `uv run python src/cli.py --help` worked; full test suite passed under `uv` (`313 passed`).
**Insight:** For this repo’s current flat-import layout, `uv` works cleanly with `tool.uv.package = false`; switching to package/module entrypoints would be a separate refactor.
**Promoted to Lessons Learned:** Yes

---

### [2026-02-24] AI Agent Configuration Restructuring

**Context:** Applied standardized AI agent config guide (informed by "Evaluating AGENTS.md" and "SkillsBench" research) to restructure all agent configuration files.
**What happened:** Rewrote AGENTS.md to remove discoverable content (Quick Start, Project Maps, general coding rules), keeping only non-discoverable constraints (LM Studio, mflux, venv). Migrated two project-specific patterns ("mock external services", "prefer MetadataManager") from AGENTS.md Core Rules to LESSONS_LEARNED.md. Created four sub-agent files in `.claude/agents/` (architect, planner, agent-creator, ux-expert). Simplified CLAUDE.md pointer. Added SETUP_AI_AGENT_CONFIG.md for periodic maintenance.
**Outcome:** Success — AGENTS.md is now lean bootstrap context (~40 lines vs ~70), sub-agents are defined, learning system references are consolidated.
**Insight:** Keeping AGENTS.md small and non-discoverable makes it age better; discoverable instructions drift as code changes but non-discoverable constraints (external services, venv) remain stable.
**Promoted to Lessons Learned:** No

---

### [2026-02-15] Merged CLAUDE instructions into AGENTS and reversed link direction

**Context:** User requested AGENTS as the single instruction source, with CLAUDE reduced to a pointer.
**What happened:** Moved actionable guidance from `CLAUDE.md` into `AGENTS.md` (quick start, core rules, session checklist, dependencies, doc map), and replaced `CLAUDE.md` contents with a one-line instruction to read `AGENTS.md`.
**Outcome:** Success - AGENTS is now the canonical instruction file and CLAUDE is a redirect.
**Insight:** Keeping one canonical instruction file reduces drift between agent entrypoints.
**Promoted to Lessons Learned:** No

---

### [2026-02-15] Added project memory system files and wiring

**Context:** User requested a persistent AI-agent memory system with curated lessons, an append-only iteration log, and AGENTS workflow wiring.
**What happened:** Replaced `LESSONS_LEARNED.md` with a structured curated template, migrated existing validated lessons into categories, created `ITERATION_LOG.md` with required format, and updated `AGENTS.md` with mandatory read/write workflow rules.
**Outcome:** Success - memory system is now present and referenced from `AGENTS.md`.
**Insight:** Keep `LESSONS_LEARNED.md` high-signal and concise; put session detail in the log and only promote repeated/validated patterns.
**Promoted to Lessons Learned:** No

---

### [2026-05-08] Clarified dry-run CLI help and added regression coverage

**Context:** The CLI `--dry-run` option had stale help text and lacked a direct regression test.
**What happened:** Reworded the `--dry-run` help text to say it previews grammar without generating images, and added a CLI test that mocks grammar generation, verifies the preview output, and confirms the full pipeline is not constructed in dry-run mode.
**Outcome:** Success — focused CLI test file passed and the full suite stayed green.
**Insight:** Small CLI text tweaks are worth pinning with a behavior test when they describe a distinct execution path.
**Promoted to Lessons Learned:** No

---

### [2026-05-11] Corrected LM Studio base URL docs

**Context:** The project README and generate skill still pointed at the LM Studio host without the `/v1` API suffix, even though the code and other instructions use the full base URL.
**What happened:** Updated `README.md` and `.claude/skills/generate/SKILL.md` so the documented default matches `http://localhost:1234/v1`.
**Outcome:** Success — documentation now aligns with the configured default base URL.
**Insight:** When a tool-specific skill mirrors README setup instructions, keep both in sync with the actual configured API path.
**Promoted to Lessons Learned:** No

---

### [2026-05-12] Synced README test-suite count with collected tests

**Context:** The README’s development section listed an outdated test-suite count.
**What happened:** Ran `uv run pytest --collect-only -q` to verify the current collection count, then updated the README to say the suite currently collects 330 tests.
**Outcome:** Success — documentation now matches the observed test collection output.
**Insight:** Small maintenance docs can drift quietly; a quick collect-only check is enough to confirm the current count before editing.
**Promoted to Lessons Learned:** No

---
### [2026-05-13] Clarified zero-value prompt-only layout in CLI help

**Context:** The CLI already accepted `--images-per-prompt 0` as a prompt-only layout, but the `--help` text only mentioned the default and did not surface that contract.
**What happened:** Updated `src/cli.py` so the `--images-per-prompt` help string now says `0 = prompt-only layout`, and added a focused CLI test that asserts `--help` includes that wording.
**Outcome:** Success — `uv run pytest tests/test_cli.py -q` passed with 14 tests.
**Insight:** When the code treats `0` as a real sentinel, the CLI help should name that behavior explicitly so users do not assume it is invalid.
**Promoted to Lessons Learned:** No

---

### [2026-05-13] Synced README test-suite count with current collection

**Context:** The README’s development section had drifted from the live test collection count.
**What happened:** Ran `uv run pytest --collect-only -q` and confirmed the suite currently collects 331 tests, then updated the README to match.
**Outcome:** Success — documentation now matches the observed collection output.
**Insight:** Collect-only verification is the cheapest way to keep count-based docs honest.
**Promoted to Lessons Learned:** No

---

### [2026-05-13] Synced stale image-generator codemap note

**Context:** The testing codemap still claimed `tests/test_image_generator.py` aborted because of native MLX/mflux import behavior, but the current checkout had already passed the full suite.
**What happened:** Ran `uv run pytest -q` and `uv run pytest tests/test_image_generator.py -q` to verify the current state, then updated `docs/codemaps/testing.md` to say the image-generator test file runs cleanly and the suite passes.
**Outcome:** Success — the codemap now reflects the live verification result.
**Insight:** When a maintenance note mentions an environment-specific failure, re-run the narrow test before preserving the warning; stale caveats can survive long after the underlying issue is fixed.
**Promoted to Lessons Learned:** Yes

---

### [2026-05-14] Synced prompt-only gallery labels across UI surfaces

**Context:** The interactive gallery and index forms still used the shorter `Images/Prompt (0 = prompt-only)` label, while the CLI help and README already spelled out `0 = prompt-only layout`.
**What happened:** Updated the gallery and gallery index form labels to say `Images/Prompt (0 = prompt-only layout)`, and aligned the focused gallery/index tests with the new wording.
**Outcome:** Success — `uv run pytest tests/test_gallery.py tests/test_gallery_index.py -q` passed (11 tests).
**Insight:** Small copy syncs are easier to keep consistent when the exact runtime label is asserted in the tests that render the surface.
**Promoted to Lessons Learned:** No

---

### [2026-05-14] Clarified shared quantize behavior in CLI help and README

**Context:** The CLI `--quantize` flag already defaulted to 8 and was used by both prompt generation and standalone image enhancement, but the user-facing docs did not say that clearly.
**What happened:** Updated `src/cli.py` help text, the README options table, and the standalone enhancement section to note that `--quantize` applies to generation and enhancement and defaults to 8 when omitted; added a CLI help regression that asserts the new wording appears in `--help` output.
**Outcome:** Success — `uv run pytest tests/test_cli.py -q` passed (14 tests).
**Insight:** When one flag feeds multiple execution paths, the help text should name every path so users do not assume it is generation-only.
**Promoted to Lessons Learned:** Yes

---

### [2026-05-14] Exposed LM Studio base URL default in CLI help

**Context:** The CLI `--base-url` option already defaulted to `http://localhost:1234/v1`, but the help text did not surface that default for users scanning `--help` output.
**What happened:** Updated the `--base-url` help string to include the default URL and added a CLI help regression that asserts the base URL default is visible alongside the existing prompt-only layout/help coverage.
**Outcome:** Success — focused CLI tests passed (`14 passed`).
**Insight:** Click help output can wrap long option descriptions, so tests should assert stable substrings rather than a single exact line when a default is embedded in a long help string.
**Promoted to Lessons Learned:** Yes

### [2026-05-15] Noted benign uv environment warning during test-count verification

**Context:** While checking the live test collection count to keep docs honest, `uv run` emitted a `VIRTUAL_ENV` mismatch warning from the Hermes environment.
**What happened:** Ran `uv run pytest --collect-only -q`, confirmed the suite still collects 331 tests, and recorded that the warning did not block the repo-local command.
**Outcome:** Success — the docs count remains current and the environment quirk is now captured for future runs.
**Insight:** When `uv run` warns about `VIRTUAL_ENV` drift in this checkout, the repo command can still be trusted if the focused verification completes cleanly.
**Promoted to Lessons Learned:** Yes

### [2026-05-15] Documented grammar revision history in the gallery README

**Context:** The gallery now persists grammar revision history and exposes it in the UI, but the README only described the edit/regenerate path and did not mention the restore surface or the on-disk history file.
**What happened:** Updated the README gallery section to mention grammar-history review/restore and added `dragon_grammar_history.json` to the example output structure so the persisted revision file is discoverable.
**Outcome:** Success — user-facing docs now describe the grammar history capability that already exists in the app.
**Insight:** Small persistence features are easier to find later when the README names both the UI affordance and the file written to disk.
**Promoted to Lessons Learned:** No

### [2026-05-15] Documented benign Hermes uv warning in the README test notes

**Context:** `uv run pytest --collect-only -q` still completed successfully, but the shell emitted a recurring `VIRTUAL_ENV` mismatch warning under Hermes that can distract future maintainers during test runs.
**What happened:** Added a short note in the README's testing section explaining that the warning is benign when `uv run` finishes successfully and the repo environment is still used.
**Outcome:** Success — the warning is now documented alongside the test commands that can surface it.
**Insight:** Small environment quirks are easier to remember when they live next to the command that triggers them.
**Promoted to Lessons Learned:** No

### [2026-05-16] Documented prompt-only gallery layout in the README

**Context:** The gallery already supported the `Images/Prompt (0 = prompt-only layout)` control on per-gallery forms, but the README only said image settings were configured per-gallery and did not name the prompt-only affordance.
**What happened:** Added one README sentence under the Web UI section to call out the prompt-only layout control explicitly.
**Outcome:** Success — the user-facing docs now mention the shipped prompt-only gallery behavior in the place readers look for gallery setup details.
**Insight:** When a form exposes a useful zero-value sentinel, the README should name the exact control label so users can find it without hunting through the UI.
**Promoted to Lessons Learned:** No

---

### [2026-05-16] Added missing CLI and gallery-index test surfaces to the README coverage list

**Context:** The README’s development section listed the live 331-test collection count, but its coverage bullet list omitted `tests/test_cli.py` and `tests/test_gallery_index.py` even though both files are part of the current suite.
**What happened:** Ran `uv run pytest --collect-only -q` to verify the live test surface, then updated the README coverage list to include CLI help/validation and gallery index rendering.
**Outcome:** Success — the user-facing test summary now matches the collected suite more completely.
**Insight:** When a README summarizes test coverage, it should name the top-level test modules that users are most likely to search for, not just the broad subsystems.
**Promoted to Lessons Learned:** No

---

### [2026-05-17] Rejected invalid CLI counts before pipeline execution

**Context:** The CLI accepted integer options for prompt count and images per prompt, but the contract only makes sense for positive prompt counts and non-negative images-per-prompt values (`0` is the documented prompt-only layout).
**What happened:** Changed `--count` to `click.IntRange(min=1)` and `--images-per-prompt` to `click.IntRange(min=0)`, added CLI validation regressions for `--count 0` and `--images-per-prompt -1`, and refreshed the README test count after the suite grew by two tests.
**Outcome:** Success — focused CLI tests passed (`16 passed`) and the full suite passed (`333 passed`).
**Insight:** Click's `IntRange` gives early, consistent validation for numeric CLI contracts while preserving documented sentinel values like zero.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Planned ERNIE-Image-Turbo q4-only migration

**Context:** Requested a thorough plan to replace Z-Image and FLUX support with ERNIE-Image-Turbo at 4-bit quantization and add its Prompt Enhancer.
**What happened:** Audited model coupling across generation, pipeline, CLI, API, worker, HTML forms, metadata, templates, tests, dependencies, and README. Verified mflux 0.18.0 adds ERNIE-Image-Turbo with q4 support but explicitly omits Baidu's Prompt Enhancer; Baidu documents the PE as a separate causal-LM stage receiving prompt, width, and height.
**Outcome:** Planning in progress — ERNIE-only simplification is clear; Prompt Enhancer runtime and persistence behavior require product decisions before implementation.
**Insight:** mflux model support and official ERNIE Prompt Enhancer support are separate integrations; treating PE as part of the mflux generator would hide a required runtime boundary.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Reviewed ERNIE prompt-engineering guide

**Context:** The user supplied a third-party ERNIE Image prompt-engineering guide while refining the q4-only migration plan.
**What happened:** Compared its recommendations with Baidu's official Prompt Enhancer contract and mflux limitations.
**Outcome:** The guide informs the ERNIE-specific Tracery template and supports optional PE bypass, but does not provide a Prompt Enhancer runtime.
**Insight:** Separate prompt-authoring guidance from PE execution: the former shapes grammar output; the latter is a resolution-aware causal-LM rewrite stage returning structured JSON.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Removed retired model-family caches

**Context:** After selecting ERNIE-Image-Turbo q4 as the sole generator, the user requested removal of obsolete models and LoRAs from disk.
**What happened:** Removed the cached FLUX.2 Klein 9B model, Z-Image Turbo q4 model, their Hugging Face lock directories, and two Z-Image LoRAs with their locks. Left unrelated FLUX.1 cache stubs untouched. Verified the target paths are absent and the mflux LoRA cache is empty. Also verified LM Studio currently serves `google/gemma-4-26b-a4b-qat` locally as an MLX 4-bit model.
**Outcome:** Success — approximately 37.9 GB reclaimed; local grammar-model prerequisite confirmed.
**Insight:** mflux stores generator weights and LoRAs in different cache roots, so retiring a model family requires inspecting both locations.
**Promoted to Lessons Learned:** Yes

---

### [2026-06-20] Added LM Studio unload boundary to migration plan

**Context:** ERNIE q4 generation must reclaim memory held by any locally loaded LM Studio model.
**What happened:** Verified the installed LM Studio CLI supports non-interactive inspection with `lms ps --json` and unloading with `lms unload --all`; the configured Gemma instance currently occupies 15.64 GB.
**Outcome:** Plan updated conceptually: every mflux generation or SeedVR2 entry path must unload LM Studio models immediately before loading MLX image weights, and fail closed if unloading fails.
**Insight:** Grammar inference and image inference need an explicit lifecycle boundary; relying on users to unload LM Studio makes unified-memory failures nondeterministic.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Created ERNIE-only migration goal

**Context:** The user requested the agreed migration plan be converted into a `/goal` objective.
**What happened:** Created an active measurable goal covering ERNIE-Turbo q4, local Gemma Tracery generation, mandatory LM Studio unloading, SeedVR2 retention, legacy removal, full automated checks, and two consecutive end-to-end validations.
**Outcome:** Success — goal is active with explicit completion evidence and stop conditions.
**Insight:** No new reusable technical insight.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Implemented ERNIE-Turbo q4-only architecture

**Context:** Active goal replaced Z-Image/FLUX with persistent ERNIE-Image-Turbo q4, local Gemma prompt enhancement, strict LM Studio memory handoff, and retained SeedVR2.
**What happened:** Upgraded mflux to 0.18.0; added fixed ERNIE q4/8-step/guidance-1 loading; added fail-closed `lms unload --all` boundaries before ERNIE and SeedVR2 loads; removed legacy controls and adapters across CLI, API, UI, worker, pipeline, templates, and tests; rewrote docs and metadata; rejected removed API fields; removed the OpenAI client dependency.
**Outcome:** Success for code paths — 355 tests pass and Ruff is clean. Persistent q4 provisioning and hardware image E2E remain pending because the required unsandboxed command approval was rejected after the account reached its Codex usage limit.
**Insight:** Fixed model architecture is simpler and safer when enforced at request validation, pipeline signatures, loader construction, and recorded metadata rather than only hidden in UI controls.
**Promoted to Lessons Learned:** No

---

### [2026-06-20] Validated Gemma prompt enhancement through native LM Studio chat

**Context:** LM Studio's OpenAI-compatible chat endpoint ignored `chat_template_kwargs.enable_thinking`, causing Gemma to spend its output budget on reasoning and return no grammar.
**What happened:** Switched grammar generation to `POST /api/v1/chat` with the exact local model ID, a dedicated system prompt, `reasoning: "off"`, and `store: false`; ran an uncached live request from an empty LM Studio state.
**Outcome:** Success — LM Studio auto-loaded Gemma and returned valid ERNIE-oriented Tracery JSON; the app then unloaded Gemma cleanly.
**Insight:** Native LM Studio reasoning controls are required for predictable non-reasoning Gemma output in this setup.
**Promoted to Lessons Learned:** Yes

---

### [2026-06-20] Completed consecutive ERNIE q4 hardware verification

**Context:** Completion required two uninterrupted local runs spanning Gemma grammar generation, LM Studio memory handoff, ERNIE q4 rendering, fixed metadata, and SeedVR2 enhancement.
**What happened:** Provisioned checkpoint presence was verified at 6.2 GB. A live run exposed LM Studio's just-in-time reload race after `lms unload --all`; grammar generation was hardened with native inventory inspection, synchronous model loading at 8192 context, and three bounded retries for transient cancellation. Then two consecutive 512×512 runs completed with different prompts and seeds. Final audit moved the fail-closed unload boundary from cache misses to every ERNIE and SeedVR2 operation.
**Outcome:** Success — both runs produced valid 1024×1024 SeedVR2 outputs and metadata recording `ernie-image-turbo`, q4, 8 steps, and guidance 1.0. LM Studio ended empty. Full suite: 357 passed; Ruff clean.
**Insight:** A model unload command can complete before LM Studio's next just-in-time load is safe; an explicit load handshake makes alternating LLM/image workloads deterministic.
**Promoted to Lessons Learned:** Yes

---

### [2026-06-20] Aligned Gemma grammars with ERNIE prompt guidance

**Context:** Generated prompts needed the recommended ERNIE structure and at least five, preferably seven, alternatives per varying Tracery rule.
**What happened:** Rebuilt the Gemma system prompt around subject → details/spatial description → style/medium → technical/capture finish, added category-specific layout patterns, replaced the misleading single-option example with a complete seven-option example, advanced the cache schema to `ernie-v2`, and added structural validation for 5–7 distinct alternatives and valid rule references.
**Outcome:** Success — 361 tests pass and Ruff is clean. An uncached live Gemma poster grammar returned five varying rules with seven alternatives each, preserved exact headline text, and followed the ERNIE layout-first poster format.
**Insight:** Few-shot examples must obey the same cardinality and output structure demanded by the instructions; otherwise the model follows the example instead of the prose constraint.
**Promoted to Lessons Learned:** No

---

### [2026-08-04] Migrated repository identity

**Context:** Git transport for the original GitHub repository was disabled, requiring a recoverable migration to a fresh repository and clearer product name.
**What happened:** Renamed the product and package to Prompt to Image Variations, updated repository and Pages URLs, and retained historical run artifacts unchanged.
**Outcome:** Local identity patch prepared for `fabian20ro/prompt-to-image-variations`; full automated checks run as migration verification.
**Insight:** Historical reports should retain the repository identity recorded when they were generated; current code, package metadata, documentation, and operational links should use the replacement identity.
**Promoted to Lessons Learned:** No

---

<!-- New entries go above this line, most recent first -->
