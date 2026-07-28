"""Tests for configurable asynchronous retry orchestration."""

from __future__ import annotations

import asyncio
import logging

import pytest

from crawlforge import (
    NetworkError,
    PermanentError,
    RetryStrategy,
    TransientError,
)


@pytest.mark.asyncio
async def test_transient_error_retries_with_exponential_backoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Temporary failures retry until success with an exponential schedule."""
    attempts = 0
    sleeps: list[float] = []

    async def operation(url: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError("temporarily unavailable", url=url)
        return "ok"

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    strategy = RetryStrategy(
        max_retries=3,
        backoff_factor=0.25,
        sleep=record_sleep,
    )
    with caplog.at_level(logging.INFO, logger="crawlforge.retry"):
        result = await strategy.execute_with_retry(operation, "https://example.test")

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [0.25, 0.5]
    assert "type=TransientError" in caplog.text
    assert "attempt=2" in caplog.text
    assert "next_delay=0.50s" in caplog.text


@pytest.mark.asyncio
async def test_permanent_error_is_recorded_without_retry() -> None:
    """Permanent failures escape immediately and retain their URL."""
    attempts = 0

    async def operation(url: str) -> None:
        nonlocal attempts
        attempts += 1
        raise PermanentError("HTTP 404: Not Found", url=url, status=404)

    strategy = RetryStrategy(max_retries=3, backoff_factor=0)

    with pytest.raises(PermanentError):
        await strategy.execute_with_retry(operation, "https://example.test/missing")

    assert attempts == 1
    assert strategy.get_stats() == {
        "errors_by_type": {"PermanentError": 1},
        "total_retries": 0,
        "successful_retries": 0,
        "average_retry_delay": 0.0,
        "permanent_error_urls": ["https://example.test/missing"],
    }


@pytest.mark.asyncio
async def test_permanent_error_cannot_be_enabled_for_retry() -> None:
    """Permanent failures remain non-retryable even if configured explicitly."""
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise PermanentError("forbidden")

    strategy = RetryStrategy(
        max_retries=3,
        backoff_factor=0,
        retry_on=[PermanentError],
    )

    with pytest.raises(PermanentError):
        await strategy.execute_with_retry(operation)

    assert attempts == 1
    assert strategy.get_stats()["total_retries"] == 0


@pytest.mark.asyncio
async def test_per_type_limits_and_backoff_factors_are_independent() -> None:
    """Each retryable error type uses its own limit and delay factor."""
    errors = [
        NetworkError("connection refused"),
        TransientError("service unavailable"),
        TransientError("still unavailable"),
    ]
    sleeps: list[float] = []

    async def operation() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    strategy = RetryStrategy(
        max_retries=4,
        backoff_factor=1.0,
        retry_limits={NetworkError: 1, TransientError: 2},
        backoff_factors={NetworkError: 0.5, TransientError: 2.0},
        sleep=record_sleep,
    )

    assert await strategy.execute_with_retry(operation) == "ok"
    assert sleeps == [0.5, 2.0, 4.0]


@pytest.mark.asyncio
async def test_subclasses_share_the_configured_type_retry_limit() -> None:
    """Subclass variation cannot bypass the configured base-type budget."""

    class FirstTransient(TransientError):
        pass

    class SecondTransient(TransientError):
        pass

    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FirstTransient("first")
        raise SecondTransient("second")

    strategy = RetryStrategy(
        max_retries=3,
        backoff_factor=0,
        retry_limits={TransientError: 1},
    )

    with pytest.raises(SecondTransient):
        await strategy.execute_with_retry(operation)

    assert attempts == 2
    assert strategy.get_stats()["total_retries"] == 1


@pytest.mark.asyncio
async def test_specific_subclass_policies_keep_independent_budgets() -> None:
    """Specific limits and factors remain independent under a shared base type."""

    class FirstTransient(TransientError):
        pass

    class SecondTransient(TransientError):
        pass

    errors = [
        FirstTransient("first"),
        SecondTransient("second"),
        SecondTransient("second again"),
    ]
    sleeps: list[float] = []

    async def operation() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    strategy = RetryStrategy(
        max_retries=3,
        retry_limits={
            TransientError: 3,
            FirstTransient: 1,
            SecondTransient: 2,
        },
        backoff_factors={FirstTransient: 10.0, SecondTransient: 20.0},
        max_backoff=100.0,
        sleep=record_sleep,
    )

    assert await strategy.execute_with_retry(operation) == "ok"
    assert sleeps == [10.0, 20.0, 40.0]


@pytest.mark.asyncio
async def test_retry_cancellation_propagates_during_backoff() -> None:
    """Cancellation is not converted into another retry or final error."""
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def operation() -> None:
        raise NetworkError("DNS unavailable")

    async def blocked_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    strategy = RetryStrategy(backoff_factor=1.0, sleep=blocked_sleep)
    task = asyncio.create_task(strategy.execute_with_retry(operation))
    await asyncio.wait_for(sleep_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert strategy.get_stats()["total_retries"] == 1
    assert len(strategy.error_history) == 1


@pytest.mark.asyncio
async def test_retry_statistics_and_history_can_be_reset() -> None:
    """Successful recovery updates aggregates and reset clears all state."""
    attempts = 0

    async def operation(url: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise NetworkError("connection refused", url=url)
        return "recovered"

    strategy = RetryStrategy(backoff_factor=0)

    assert (
        await strategy.execute_with_retry(operation, "https://example.test")
        == "recovered"
    )
    assert strategy.get_stats() == {
        "errors_by_type": {"NetworkError": 1},
        "total_retries": 1,
        "successful_retries": 1,
        "average_retry_delay": 0.0,
        "permanent_error_urls": [],
    }
    assert strategy.error_history[0].to_dict()["retry_scheduled"] is True

    strategy.reset_stats()

    assert strategy.error_history == ()
    assert strategy.get_stats()["errors_by_type"] == {}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"backoff_factor": -1.0}, "backoff_factor"),
        ({"max_backoff": float("inf")}, "max_backoff"),
        ({"retry_limits": {TransientError: -1}}, "retry limits"),
    ],
)
def test_invalid_retry_configuration_is_rejected(
    arguments: dict[str, object],
    message: str,
) -> None:
    """Invalid retry bounds cannot create an unsafe strategy."""
    with pytest.raises(ValueError, match=message):
        RetryStrategy(**arguments)  # type: ignore[arg-type]
