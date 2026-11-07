# Pipeline Codemap

## Main Flow

1. `src/cli.py` parses flags and selects mode.
2. `src/pipeline.py` (`PipelineExecutor`) runs core stages.
3. `src/grammar_generator.py` generates/caches Tracery grammar.
4. `src/tracery_runner.py` expands grammar into prompt `.txt` files.
5. `src/gallery.py` creates/updates run gallery HTML.
6. `src/image_generator.py` optionally renders images.
7. `src/image_enhancer.py` optionally enhances images.
8. `src/gallery_index.py` refreshes `generated/index.html`.

## PipelineExecutor Surface

- `run_full_pipeline(...)`
- `run_from_grammar(...)`
- `run_from_grammar_text(...)`
- `run_from_prompts(...)`
- `regenerate_prompts(...)`
- `generate_single_image(...)`
- `enhance_single_image(...)`
- `generate_all_images(...)`
- `enhance_all_images(...)`

## Files Created Per Run

- Prompt files: `{prefix}_{n}.txt`
- Images: `{prefix}_{prompt_idx}_{image_idx}.png`
- Metadata: `{prefix}.metaprompt.json`
- Per-image provenance/actual seed: `{prefix}_{prompt_idx}_{image_idx}.generation.json`
- Grammar: `{prefix}_grammar.json`
- Optional raw LLM response: `{prefix}_raw_response.txt`
- Gallery: `{prefix}_gallery.html`

## Important Couplings

### ADR — present prompt artifacts (2026-10-11)

`utils.prompt_files_in_run` owns prompt-file identity for CLI inspection and
prompt/gallery readers. Known prefixes use `{prefix}_{n}.txt`, including prefixes
with underscores; raw-response sidecars are never prompts. Metadata-free legacy
lists retain opaque text filenames. Count present artifacts, not metadata's
intended count. Image-format handling remains unchanged; no files are rewritten.

- Metadata drives most downstream behavior (prefix, image settings, counts).
- Gallery generation depends on prompt/image naming conventions.
- Resume logic relies on existence of `{prefix}_{prompt_idx}_{image_idx}.png`.
- Enhancement modifies images in place; archive/backup utilities in `src/utils.py` are safety-critical.

## Extension Points

- Fixed renderer: FLUX 3 Image Klein distilled q4, 16 steps, no CFG; no model dispatch.
- `generation_settings()` owns fixed renderer provenance. Qwen produces complete captions via the versioned template.
- Grammar/cache/API validation shares `validate_grammar_structure` through `parse_grammar`.
- Creation requires a fresh output directory. Existing-run rendering rejects legacy/unknown models before writes.
- Image/enhancement/grammar public boundaries release the other models and MLX allocator cache.
- Add metadata fields through `src/metadata_manager.py` and readers that consume metadata.
