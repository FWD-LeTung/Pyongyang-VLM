"""Timing helpers for lightweight function instrumentation."""

from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar

from src.utils.logger import setup_logger


P = ParamSpec("P")
R = TypeVar("R")


def time_it(func: Callable[P, R]) -> Callable[P, R]:
    """Log the execution time of the wrapped function."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            logger = setup_logger(func.__module__)
            logger.info(
                "Function %s executed in %.4f seconds",
                func.__name__,
                elapsed,
            )

    return wrapper
