"""LM Studio process lifecycle helpers."""

import logging
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


class LMStudioUnloadError(RuntimeError):
    """Raised when loaded LM Studio models cannot be released."""


_MAX_RETRIES = 3
_BACKOFF_SECONDS = (1.0, 2.0)


def unload_all_models(timeout: float = 60.0) -> None:
    """Unload every LM Studio model before an mflux model is loaded.

    Retries on transient subprocess failure (e.g., model still in use).
    Raises after exhausting retries or on non-transient errors.
    """
    executable = shutil.which("lms")
    if executable is None:
        raise LMStudioUnloadError("LM Studio CLI `lms` was not found on PATH")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            result = subprocess.run(
                [executable, "unload", "--all"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_exc = LMStudioUnloadError("Timed out unloading LM Studio models")
            if attempt < _MAX_RETRIES - 1:
                logger.info(
                    "LM Studio unload attempt %d/%d timed out, retrying in %.1fs",
                    attempt + 2,
                    _MAX_RETRIES,
                    _BACKOFF_SECONDS[attempt],
                )
                time.sleep(_BACKOFF_SECONDS[attempt])
                continue
            raise last_exc

        if result.returncode == 0:
            return

        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        last_exc = LMStudioUnloadError(
            f"Failed to unload LM Studio models: {detail}"
        )
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_BACKOFF_SECONDS[attempt])
            continue
        raise last_exc

    raise last_exc  # type: ignore[misc]