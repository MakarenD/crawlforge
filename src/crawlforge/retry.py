"""Configurable asynchronous retry orchestration and error statistics."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict, TypeVar

from crawlforge.errors import (
    CrawlError,
    NetworkError,
    ParseError,
    PermanentError,
    TransientError,
)

logger = logging.getLogger(__name__)

type ErrorType = type[BaseException]
type AsyncCallable[T] = Callable[..., Awaitable[T]]
type SleepCallable = Callable[[float], Awaitable[None]]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Serializable details about one failed attempt."""

    timestamp: str
    error_type: str
    message: str
    url: str | None
    attempt: int
    retry_scheduled: bool
    retry_delay: float
    status: int | None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return a JSON-compatible representation of the record."""
        return asdict(self)


class RetryStats(TypedDict):
    """Snapshot of retry and classified-error statistics."""

    errors_by_type: dict[str, int]
    total_retries: int
    successful_retries: int
    average_retry_delay: float
    permanent_error_urls: list[str]


class RetryStrategy:
    """Execute asynchronous operations with bounded exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_on: list[ErrorType] | None = None,
        *,
        retry_limits: Mapping[ErrorType, int] | None = None,
        backoff_factors: Mapping[ErrorType, float] | None = None,
        max_backoff: float = 30.0,
        sleep: SleepCallable | None = None,
    ) -> None:
        """Configure retryable errors, per-type limits, and delay factors."""
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        self._validate_delay(backoff_factor, "backoff_factor")
        self._validate_delay(max_backoff, "max_backoff")

        configured_errors = tuple(
            retry_on if retry_on is not None else [TransientError, NetworkError]
        )
        if any(
            not isinstance(error_type, type)
            or not issubclass(error_type, BaseException)
            for error_type in configured_errors
        ):
            raise TypeError("retry_on must contain exception types")

        limits: dict[ErrorType, int] = dict(retry_limits or {})
        for error_type, limit in limits.items():
            if not isinstance(error_type, type) or not issubclass(
                error_type,
                BaseException,
            ):
                raise TypeError("retry_limits keys must be exception types")
            if limit < 0:
                raise ValueError("retry limits must be zero or greater")

        factors: dict[ErrorType, float] = dict(backoff_factors or {})
        for error_type, factor in factors.items():
            if not isinstance(error_type, type) or not issubclass(
                error_type,
                BaseException,
            ):
                raise TypeError("backoff_factors keys must be exception types")
            self._validate_delay(factor, "backoff factors")

        self.max_retries: int = max_retries
        self.backoff_factor: float = backoff_factor
        self.retry_on: tuple[ErrorType, ...] = configured_errors
        self.retry_limits: dict[ErrorType, int] = limits
        self.backoff_factors: dict[ErrorType, float] = factors
        self.max_backoff: float = max_backoff
        self._sleep: SleepCallable = sleep or asyncio.sleep
        self._attempt: ContextVar[int] = ContextVar(
            f"crawlforge_retry_attempt_{id(self)}",
            default=1,
        )
        self._records: list[ErrorRecord] = []
        self._errors_by_type: dict[str, int] = defaultdict(int)
        self._total_retries = 0
        self._successful_retries = 0
        self._total_retry_delay = 0.0
        self._permanent_error_urls: set[str] = set()

    @property
    def current_attempt(self) -> int:
        """Return the one-based attempt number for the current async context."""
        return self._attempt.get()

    @property
    def error_history(self) -> tuple[ErrorRecord, ...]:
        """Return an immutable snapshot of all recorded failures."""
        return tuple(self._records)

    async def execute_with_retry(
        self,
        coro: AsyncCallable[T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async callable and retry configured failures."""
        retries_by_type: dict[ErrorType, int] = defaultdict(int)
        retries_performed = 0
        inferred_url = self._infer_url(args, kwargs)

        for attempt_index in range(self.max_retries + 1):
            attempt = attempt_index + 1
            token = self._attempt.set(attempt)
            logger.info(
                "Attempt %d/%d for %s",
                attempt,
                self.max_retries + 1,
                inferred_url or "<operation>",
            )
            try:
                result = await coro(*args, **kwargs)
            except Exception as error:
                retry_type = self._matching_retry_type(error)
                url = self._error_url(error) or inferred_url
                retry = self._can_retry(
                    error,
                    retry_type,
                    retries_by_type,
                    retries_performed,
                )
                delay = (
                    self._retry_delay(
                        error,
                        retry_type,
                        retries_by_type,
                    )
                    if retry and retry_type is not None
                    else 0.0
                )
                self.record_error(
                    error,
                    url=url,
                    attempt=attempt,
                    retry_scheduled=retry,
                    retry_delay=delay,
                )

                if not retry:
                    logger.warning(
                        "Final failure for %s: type=%s attempt=%d",
                        url or "<operation>",
                        type(error).__name__,
                        attempt,
                    )
                    raise

                assert retry_type is not None
                for counter_type in self._counter_types(error, retry_type):
                    retries_by_type[counter_type] += 1
                retries_performed += 1
                self._total_retries += 1
                self._total_retry_delay += delay
                logger.info(
                    "Retry scheduled: type=%s url=%s attempt=%d next_delay=%.2fs",
                    type(error).__name__,
                    url or "<operation>",
                    attempt,
                    delay,
                )
                if delay:
                    await self._sleep(delay)
            else:
                if retries_performed:
                    self._successful_retries += 1
                logger.info(
                    "Operation succeeded for %s after %d attempt(s)",
                    inferred_url or "<operation>",
                    attempt,
                )
                return result
            finally:
                self._attempt.reset(token)

        raise AssertionError("retry loop exhausted without an operation outcome")

    def record_error(
        self,
        error: BaseException,
        *,
        url: str | None = None,
        attempt: int = 1,
        retry_scheduled: bool = False,
        retry_delay: float = 0.0,
    ) -> None:
        """Record a classified failure that occurred outside retry execution."""
        error_url = self._error_url(error) or url
        error_name = type(error).__name__
        status = error.status if isinstance(error, CrawlError) else None
        self._errors_by_type[error_name] += 1
        if isinstance(error, PermanentError) and error_url is not None:
            self._permanent_error_urls.add(error_url)
        self._records.append(
            ErrorRecord(
                timestamp=datetime.now(UTC).isoformat(),
                error_type=error_name,
                message=str(error),
                url=error_url,
                attempt=attempt,
                retry_scheduled=retry_scheduled,
                retry_delay=retry_delay,
                status=status,
            )
        )

    def get_stats(self) -> RetryStats:
        """Return aggregate errors, retry outcomes, and permanent URLs."""
        return {
            "errors_by_type": dict(sorted(self._errors_by_type.items())),
            "total_retries": self._total_retries,
            "successful_retries": self._successful_retries,
            "average_retry_delay": (
                self._total_retry_delay / self._total_retries
                if self._total_retries
                else 0.0
            ),
            "permanent_error_urls": sorted(self._permanent_error_urls),
        }

    def reset_stats(self) -> None:
        """Clear accumulated retry statistics and error history."""
        self._records.clear()
        self._errors_by_type.clear()
        self._total_retries = 0
        self._successful_retries = 0
        self._total_retry_delay = 0.0
        self._permanent_error_urls.clear()

    def _matching_retry_type(self, error: BaseException) -> ErrorType | None:
        if isinstance(error, (PermanentError, ParseError)):
            return None
        matches = [
            error_type for error_type in self.retry_on if isinstance(error, error_type)
        ]
        return min(
            matches,
            key=lambda error_type: self._type_distance(type(error), error_type),
            default=None,
        )

    def _can_retry(
        self,
        error: BaseException,
        error_type: ErrorType | None,
        retries_by_type: Mapping[ErrorType, int],
        retries_performed: int,
    ) -> bool:
        if error_type is None or retries_performed >= self.max_retries:
            return False
        limits = [
            (configured_type, limit)
            for configured_type, limit in self.retry_limits.items()
            if isinstance(error, configured_type)
        ]
        if any(
            retries_by_type.get(configured_type, 0) >= limit
            for configured_type, limit in limits
        ):
            return False
        if isinstance(error, CrawlError) and error.retry_limit is not None:
            return retries_by_type.get(error_type, 0) < error.retry_limit
        return True

    def _retry_delay(
        self,
        error: BaseException,
        error_type: ErrorType,
        retries_by_type: Mapping[ErrorType, int],
    ) -> float:
        factors = [
            (
                self._type_distance(type(error), configured_type),
                configured_type,
                factor,
            )
            for configured_type, factor in self.backoff_factors.items()
            if isinstance(error, configured_type)
        ]
        if factors:
            _distance, factor_type, factor = min(
                factors,
                key=lambda item: item[0],
            )
        else:
            factor_type = error_type
            factor = self.backoff_factor
        multiplier: float = (
            error.backoff_multiplier if isinstance(error, CrawlError) else 1.0
        )
        retry_after: float = error.retry_after if isinstance(error, CrawlError) else 0.0
        backoff: float = (
            factor * (2 ** retries_by_type.get(factor_type, 0)) * multiplier
        )
        return float(max(min(backoff, self.max_backoff), retry_after))

    def _counter_types(
        self,
        error: BaseException,
        retry_type: ErrorType,
    ) -> set[ErrorType]:
        return {
            retry_type,
            *(
                configured_type
                for configured_type in self.retry_limits
                if isinstance(error, configured_type)
            ),
            *(
                configured_type
                for configured_type in self.backoff_factors
                if isinstance(error, configured_type)
            ),
        }

    def _error_url(self, error: BaseException) -> str | None:
        return error.url if isinstance(error, CrawlError) else None

    def _infer_url(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str | None:
        keyword_url = kwargs.get("url")
        if isinstance(keyword_url, str):
            return keyword_url
        return args[0] if args and isinstance(args[0], str) else None

    def _validate_delay(self, value: float, name: str) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative value")

    def _type_distance(
        self,
        concrete_type: ErrorType,
        configured_type: ErrorType,
    ) -> int:
        try:
            return concrete_type.mro().index(configured_type)
        except ValueError:
            return len(concrete_type.mro())
