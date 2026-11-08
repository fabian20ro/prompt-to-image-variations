"""Best-effort MLX memory snapshots; never import or initialize Metal here."""

import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def image_memory(operation: str, output_path: Path):
    """Stream per-image allocation snapshots, including ordinary failures.

    The peak is process-wide. Do not reset it: hardware acceptance and callers
    may measure the high-water mark across generation and enhancement together.
    """
    def emit(phase: str) -> None:
        mx = sys.modules.get("mlx.core")
        if mx is None or not all(callable(getattr(mx, name, None)) for name in (
            "get_active_memory", "get_cache_memory", "get_peak_memory",
        )):
            return
        try:
            record = {
                "operation": operation, "phase": phase, "image": output_path.name,
                "active_bytes": int(mx.get_active_memory()),
                "cache_bytes": int(mx.get_cache_memory()),
                "process_peak_bytes": int(mx.get_peak_memory()),
            }
            print(f"[image-memory] {json.dumps(record)}", file=sys.stderr, flush=True)
        except Exception:
            logging.getLogger(__name__).warning("MLX memory snapshot unavailable")

    emit("before")
    completed = False
    try:
        yield
        completed = True
    finally:
        emit("after" if completed else "failed")
