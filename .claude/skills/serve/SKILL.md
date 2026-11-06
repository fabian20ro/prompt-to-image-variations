---
name: serve
description: Start the web UI server
allowed-tools:
  - Bash(uv run python src/cli.py --serve:*)
  - Bash(uv sync:*)
---

# Web Server

Start the prompt-to-image-variations web server.

## Usage

- `/serve` - Start the web UI

## Steps

1. Sync deps if needed: `uv sync`
2. Start server: `uv run python src/cli.py --serve`
3. Report server URL: http://localhost:8000

## Notes

- Server opens browser automatically at http://localhost:8000
- Features: generation form, gallery browser, queue management
- Real-time progress updates via SSE
- Use Ctrl+C to stop the server
