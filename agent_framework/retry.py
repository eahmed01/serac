"""Provider error handling and retry with exponential backoff.

Provides structured error classification (retryable vs non-retryable)
and a configurable retry wrapper for provider calls.
"""

from __future__ import annotations

import enum
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class ErrorKind(enum.Enum):
    """Classification of provider errors."""

    # Retryable — transient failures worth retrying
    RATE_LIMITED = "rate_limited"       # 429, too many requests
    SERVER_ERROR = "server_error"       # 500, 502, 503
    TIMEOUT = "timeout"                 # 504, read timeout
    CONNECTION_ERROR = "connection"     # connection refused, DNS

    # Non-retryable — client errors, retrying won't help
    BAD_REQUEST = "bad_request"         # 400
    UNAUTHORIZED = "unauthorized"       # 401
    FORBIDDEN = "forbidden"             # 403
    NOT_FOUND = "not_found"             # 404
    INVALID_INPUT = "invalid_input"     # model/schema error
    UNKNOWN = "unknown"                 # unclassified


_RETRYABLE = {ErrorKind.RATE_LIMITED, ErrorKind.SERVER_ERROR, ErrorKind.TIMEOUT, ErrorKind.CONNECTION_ERROR}

_STATUS_TO_KIND = {
    400: ErrorKind.BAD_REQUEST,
    401: ErrorKind.UNAUTHORIZED,
    403: ErrorKind.FORBIDDEN,
    404: ErrorKind.NOT_FOUND,
    429: ErrorKind.RATE_LIMITED,
    500: ErrorKind.SERVER_ERROR,
    502: ErrorKind.SERVER_ERROR,
    503: ErrorKind.SERVER_ERROR,
    504: ErrorKind.TIMEOUT,
}


@dataclass
class ProviderError:
    """Structured provider error with retry semantics."""

    kind: ErrorKind
    message: str
    status_code: Optional[int] = None
    raw: Optional[Exception] = None

    @property
    def is_retryable(self) -> bool:
        return self.kind in _RETRYABLE

    def __repr__(self) -> str:
        r = "retryable" if self.is_retryable else "non-retryable"
        return f"ProviderError({self.kind.value} [{r}]): {self.message}"


def classify_exception(exc: Exception) -> ProviderError:
    """Classify an exception into a ProviderError.

    Heuristics: HTTP status codes, exception types, message patterns.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "response", None)
    if isinstance(status, dict):
        status = status.get("status_code")

    if status is not None:
        kind = _STATUS_TO_KIND.get(int(status), ErrorKind.UNKNOWN)
        return ProviderError(kind=kind, message=str(exc), status_code=int(status), raw=exc)

    msg = str(exc).lower()
    typ = type(exc).__name__.lower()

    if "rate limit" in msg or "429" in msg or "ratelimit" in typ:
        kind = ErrorKind.RATE_LIMITED
    elif "timeout" in msg or "timed out" in msg or "readtimeout" in typ:
        kind = ErrorKind.TIMEOUT
    elif "connection" in msg or "refused" in msg or "connectionerror" in typ:
        kind = ErrorKind.CONNECTION_ERROR
    elif "500" in msg or "502" in msg or "503" in msg or "server" in msg:
        kind = ErrorKind.SERVER_ERROR
    elif "400" in msg or "bad request" in msg:
        kind = ErrorKind.BAD_REQUEST
    elif "401" in msg or "unauthorized" in msg:
        kind = ErrorKind.UNAUTHORIZED
    elif "invalid" in msg or "schema" in msg or "validation" in msg:
        kind = ErrorKind.INVALID_INPUT
    else:
        kind = ErrorKind.UNKNOWN

    return ProviderError(kind=kind, message=str(exc), raw=exc)


# ---------------------------------------------------------------------------
# Retry config + wrapper
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum retry attempts. 0 = no retries.
        base_delay: Base delay in seconds between retries.
        max_delay: Cap on exponential backoff delay.
        jitter: Random jitter factor (0.0–1.0). Applied as ±jitter * delay.
        backoff_factor: Exponential multiplier per attempt.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.3
    backoff_factor: float = 2.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute delay for the given retry attempt (1-indexed)."""
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter > 0:
            spread = self.jitter * delay
            delay += random.uniform(-spread, spread)

        return max(delay, 0.1)


def retry_call(func, config: RetryConfig, context: str = "provider") -> object:
    """Execute *func* with retry and exponential backoff.

    If the last call raised, the ProviderError from classify_exception is
    attached to the reraised exception as ``__provider_error__``.

    Returns the result of func on success.
    """
    last_err: Optional[ProviderError] = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_err = classify_exception(exc)

            if attempt < config.max_retries and last_err.is_retryable:
                delay = config.delay_for_attempt(attempt + 1)
                logger.warning(
                    "[%s] retry %d/%d after %.1fs — %s",
                    context, attempt + 1, config.max_retries, delay, last_err.kind.value,
                )
                time.sleep(delay)
            else:
                if last_err.is_retryable:
                    logger.error("[%s] retries exhausted: %s", context, last_err.kind.value)
                else:
                    logger.error("[%s] non-retryable: %s", context, last_err.kind.value)
                # Attach structured error for caller inspection
                exc.__provider_error__ = last_err  # type: ignore[attr-defined]
                raise

    # Should not reach here, but safety net
    raise RuntimeError(f"Retry loop exited unexpectedly: {last_err}")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

DEFAULT_RETRY = RetryConfig()
