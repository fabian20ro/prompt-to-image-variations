# Testing Codemap

## Test Layout

- Config/settings: `tests/test_config.py`
- Grammar generation + cleanup: `tests/test_grammar_generator.py`
- Tracery runner: `tests/test_tracery_runner.py`
- Image generation wrapper: `tests/test_image_generator.py`
- Migration public seams: `tests/test_migration_contract.py` (LM Studio/mflux/process boundaries faked; real expansion/files/pipeline)
- Opt-in hardware/corpus acceptance: `scripts/verify_flux3.py`; outputs under `generated/verification/`
- Shared HTML/CSS/JS components: `tests/test_html_components.py`
- Gallery HTML generation: `tests/test_gallery.py`, `tests/test_gallery_index.py`
- Gallery service: `tests/test_gallery_service.py`
- Metadata manager: `tests/test_metadata_manager.py`
- Pipeline orchestration: `tests/test_pipeline.py`
- Server routes/models/queue/worker: `tests/test_routes.py`, `tests/test_models.py`, `tests/test_queue_manager.py`, `tests/test_worker.py`, `tests/test_worker_subprocess.py`

## Common Fixtures

- Shared fixtures live in `tests/conftest.py`.
- Most filesystem tests use temporary directories and synthetic run files.

## Notes

- `tests/test_image_generator.py` runs cleanly under `uv run pytest` in this checkout.
- Run `uv run pytest -q` for the current suite; automated tests do not require live LM Studio or Metal.
