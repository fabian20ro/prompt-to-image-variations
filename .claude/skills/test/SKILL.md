---
name: test
description: Run project tests with various options
allowed-tools:
  - Bash(uv run pytest:*)
  - Bash(uv sync:*)
---

# Test Runner

Run tests for the prompt-to-image-variations project.

## Usage

- `/test` - Run all tests
- `/test file.py` - Run specific test file
- `/test -k name` - Run tests matching pattern

## Steps

1. Sync deps if needed: `uv sync --group dev`
2. Run pytest with arguments: `uv run pytest $ARGUMENTS -v --tb=short`
3. Report results including:
   - Number of tests passed/failed
   - Any failure details with file:line references
   - Coverage summary if `--cov` was used

## Common Patterns

```bash
# Run all tests
uv run pytest -v --tb=short

# Run specific test file
uv run pytest tests/test_grammar_generator.py -v

# Run tests matching pattern
uv run pytest -k "grammar" -v

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run single test
uv run pytest tests/test_pipeline.py::TestPipelineResult::test_successful_result -v
```
