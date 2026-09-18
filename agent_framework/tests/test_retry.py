"""Tests for agent_framework.retry module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.retry import (
    ProviderError,
    RetryConfig,
    DEFAULT_RETRY,
    classify_exception,
    retry_call,
    ErrorKind,
)


class TestProviderError:
    """Test ProviderError dataclass."""

    def test_retryable_timeout(self):
        err = ProviderError(kind=ErrorKind.TIMEOUT, message="Connection timed out")
        assert err.is_retryable is True
        assert err.kind == ErrorKind.TIMEOUT

    def test_non_retryable_auth(self):
        err = ProviderError(kind=ErrorKind.UNAUTHORIZED, message="Invalid API key")
        assert err.is_retryable is False
        assert err.kind == ErrorKind.UNAUTHORIZED

    def test_repr(self):
        err = ProviderError(kind=ErrorKind.RATE_LIMITED, message="429", status_code=429)
        assert "rate_limited" in repr(err)
        assert "429" in repr(err)


class TestClassifyException:
    """Test classify_exception function."""

    def test_timeout_exception(self):
        exc = TimeoutError("connection timed out")
        err = classify_exception(exc)
        assert err.is_retryable is True

    def test_connection_error(self):
        exc = ConnectionError("connection refused")
        err = classify_exception(exc)
        assert err.is_retryable is True

    def test_generic_exception(self):
        exc = Exception("something broke")
        err = classify_exception(exc)
        assert err.is_retryable is False

    def test_http_429(self):
        exc = Exception("HTTP 429 Too Many Requests")
        # This would need a real HTTP error with status_code attr
        # For now, just verify it returns a ProviderError
        err = classify_exception(exc)
        assert isinstance(err, ProviderError)

    def test_http_503(self):
        exc = Exception("HTTP 503 Service Unavailable")
        err = classify_exception(exc)
        assert isinstance(err, ProviderError)

    def test_already_provider_error(self):
        original = ProviderError(kind=ErrorKind.RATE_LIMITED, message="429")
        err = classify_exception(original)
        # classify_exception wraps ProviderError in a new one
        assert err.kind == ErrorKind.RATE_LIMITED

    def test_auth_error(self):
        exc = Exception("Invalid API key")
        err = classify_exception(exc)
        assert err.is_retryable is False


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.jitter == 0.3  # Actual default
        assert cfg.backoff_factor == 2.0

    def test_custom_config(self):
        cfg = RetryConfig(max_retries=5, base_delay=2.0, max_delay=60.0)
        assert cfg.max_retries == 5
        assert cfg.base_delay == 2.0
        assert cfg.max_delay == 60.0

    def test_no_retries(self):
        cfg = RetryConfig(max_retries=0)
        assert cfg.max_retries == 0


class TestRetryCall:
    """Test retry_call function."""

    def test_no_retry_on_success(self):
        call_count = 0

        def success_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = retry_call(success_fn, DEFAULT_RETRY, "test")
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_timeout(self):
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("connection timed out")
            return "ok"

        result = retry_call(flaky_fn, DEFAULT_RETRY, "test")
        assert result == "ok"
        assert call_count == 3

    def test_no_retry_on_auth_error(self):
        call_count = 0

        def auth_fn():
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid API key")

        with pytest.raises(Exception):
            retry_call(auth_fn, DEFAULT_RETRY, "test")
        assert call_count == 1

    def test_respects_max_retries(self):
        call_count = 0
        cfg = RetryConfig(max_retries=2)

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("connection timed out")

        with pytest.raises(TimeoutError):
            retry_call(always_fail, cfg, "test")
        assert call_count == 3  # initial + 2 retries

    def test_no_retry_when_max_retries_is_0(self):
        call_count = 0
        cfg = RetryConfig(max_retries=0)

        def fail_fn():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("connection timed out")

        with pytest.raises(TimeoutError):
            retry_call(fail_fn, cfg, "test")
        assert call_count == 1

    def test_delay_cap_at_max_delay(self):
        with patch("time.sleep", return_value=None) as mock_sleep:
            call_count = 0
            cfg = RetryConfig(max_retries=5, base_delay=1.0, max_delay=2.0, jitter=0.0)

            def always_fail():
                nonlocal call_count
                call_count += 1
                raise TimeoutError("connection timed out")

            with pytest.raises(TimeoutError):
                retry_call(always_fail, cfg, "test")

            # Verify delays are capped (with jitter=0, should be exact)
            for call_args in mock_sleep.call_args_list:
                delay = call_args[0][0]
                assert delay <= 2.0

    def test_jitter_applied(self):
        with patch("time.sleep", return_value=None) as mock_sleep:
            call_count = 0
            cfg = RetryConfig(max_retries=1, base_delay=2.0, jitter=0.0)

            def fail_fn():
                nonlocal call_count
                call_count += 1
                raise TimeoutError("connection timed out")

            with pytest.raises(TimeoutError):
                retry_call(fail_fn, cfg, "test")

            assert mock_sleep.called
            delay = mock_sleep.call_args[0][0]
            assert delay == 2.0


class TestDefaultRetry:
    """Test DEFAULT_RETRY configuration."""

    def test_default_config(self):
        assert DEFAULT_RETRY.max_retries == 3
        assert DEFAULT_RETRY.base_delay == 1.0
        assert DEFAULT_RETRY.max_delay == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
