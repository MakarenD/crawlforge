"""Small cancellation-safe asynchronous execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_lifecycle_owned_thread[**P, T](
    function: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run blocking work in a thread that cannot outlive its awaiting task."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        try:
            await worker
        except BaseException as worker_error:
            raise cancelled from worker_error
        raise
