"""Tests for src/lm_studio."""

from unittest.mock import patch, MagicMock
import subprocess
import pytest

from lm_studio import unload_all_models, LMStudioUnloadError


class TestUnloadAllModels:
    def test_raises_when_lms_not_found(self):
        with patch("lm_studio.shutil.which", return_value=None):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "not found on PATH" in str(exc_info.value)

    def test_succeeds_on_first_attempt(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run = MagicMock(return_value=mock_result)

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", mock_run) as run_mock:
            unload_all_models()
            assert run_mock.call_count == 1

    def test_retries_on_failure(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "still busy"
        success_result = MagicMock()
        success_result.returncode = 0

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return fail_result
            return success_result

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep") as sleep_mock:
            unload_all_models()
            assert call_count[0] == 3
            assert sleep_mock.call_count == 2

    def test_raises_after_exhausting_retries(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "busy"

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=fail_result), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "busy" in str(exc_info.value)

    def test_raises_on_timeout_after_exhausting_retries(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["lms", "unload", "--all"], timeout=60)

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise timeout_exc
            return MagicMock(returncode=0)

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep"):
            unload_all_models()
            assert call_count[0] == 3

    def test_timeout_on_final_attempt_raises(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["lms", "unload", "--all"], timeout=60)

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            raise timeout_exc

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep") as sleep_mock:
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert call_count[0] == 3
            assert sleep_mock.call_count == 2
            assert [a.args[0] for a in sleep_mock.call_args_list] == [1.0, 2.0]
            assert "Timed out" in str(exc_info.value)

    def test_uses_stdout_when_stderr_empty_on_failure(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = ""
        fail_result.stdout = "model locked by another process"

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=fail_result), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "model locked" in str(exc_info.value)

    def test_uses_unknown_error_when_both_streams_empty(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = ""
        fail_result.stdout = ""

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=fail_result), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "unknown error" in str(exc_info.value)

    def test_stderr_takes_precedence_over_stdout_on_failure(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "error from stderr"
        fail_result.stdout = "output from stdout"

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=fail_result), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "error from stderr" in str(exc_info.value)
            assert "output from stdout" not in str(exc_info.value)

    def test_backoff_sleep_values_on_retry(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        success_result = MagicMock()
        success_result.returncode = 0

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return fail_result
            return success_result

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep") as sleep_mock:
            unload_all_models()
            assert [a.args[0] for a in sleep_mock.call_args_list] == [1.0, 2.0]

    def test_custom_timeout_propagates_to_subprocess_run(self):
        success_result = MagicMock()
        success_result.returncode = 0
        success_result.stderr = ""
        success_result.stdout = ""

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=success_result) as run_mock:
            unload_all_models(timeout=45.0)
            call_args, call_kwargs = run_mock.call_args
            assert call_args == (["/usr/bin/lms", "unload", "--all"],)
            assert call_kwargs["timeout"] == 45.0

    def test_subprocess_run_receives_capture_output_and_text(self):
        success_result = MagicMock()
        success_result.returncode = 0

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=success_result) as run_mock:
            unload_all_models()
            call_args, call_kwargs = run_mock.call_args
            assert call_args == (["/usr/bin/lms", "unload", "--all"],)
            assert call_kwargs["capture_output"] is True
            assert call_kwargs["text"] is True

    def test_non_timeout_subprocess_exception_propagates_without_retry(self):
        raise_exc = OSError("device busy")

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch(
                 "lm_studio.subprocess.run",
                 side_effect=lambda *a, **kw: (_ for _ in ()).throw(raise_exc),
             ), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(OSError) as exc_info:
                unload_all_models()
            assert "device busy" in str(exc_info.value)

    def test_succeeds_when_stderr_present_on_returncode_zero(self):
        """Successful subprocess completion (returncode==0) must not raise
        even when stderr contains non-empty content — the function should
        accept success and ignore output on success."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "retrying"

        success_result = MagicMock(returncode=0, stderr="model unloaded with warnings")
        success_result.stdout = ""

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return fail_result
            return success_result

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep"):
            unload_all_models()
            assert call_count[0] == 2
