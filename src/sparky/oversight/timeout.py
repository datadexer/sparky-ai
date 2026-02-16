"""Timeout decorator for long-running experiment steps.

Usage:
    from sparky.oversight.timeout import with_timeout, ExperimentTimeout

    @with_timeout(seconds=900)
    def train_model(X, y):
        # ... training code ...
        return model
"""

import functools
import logging
import signal
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ExperimentTimeout(Exception):
    """Raised when an experiment step exceeds its time budget."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise ExperimentTimeout("Experiment step exceeded time budget")


def with_timeout(seconds: int = 900) -> Callable:
    """Decorator that kills a function if it runs longer than `seconds`.

    Uses SIGALRM — only works on Unix. Nested timeouts are not supported;
    the innermost timeout wins.

    Args:
        seconds: Maximum wall-clock seconds allowed. Default 900 (15 min).

    Returns:
        Decorated function that raises ExperimentTimeout on timeout.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except ExperimentTimeout:
                logger.error(
                    f"[TIMEOUT] {func.__name__} exceeded {seconds}s limit"
                )
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator
