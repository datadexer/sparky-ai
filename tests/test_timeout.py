"""Tests for sparky.oversight.timeout — experiment timeout decorator."""

import time

import pytest

from sparky.oversight.timeout import with_timeout, ExperimentTimeout


class TestWithTimeout:
    def test_completes_within_timeout(self):
        @with_timeout(seconds=5)
        def fast_func():
            return 42

        assert fast_func() == 42

    def test_raises_on_timeout(self):
        @with_timeout(seconds=1)
        def slow_func():
            time.sleep(5)
            return "should not reach"

        with pytest.raises(ExperimentTimeout):
            slow_func()

    def test_preserves_function_name(self):
        @with_timeout(seconds=10)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_passes_args_and_kwargs(self):
        @with_timeout(seconds=5)
        def add(a, b, extra=0):
            return a + b + extra

        assert add(1, 2, extra=3) == 6

    def test_restores_signal_handler(self):
        import signal
        original = signal.getsignal(signal.SIGALRM)

        @with_timeout(seconds=5)
        def noop():
            pass

        noop()
        restored = signal.getsignal(signal.SIGALRM)
        assert restored == original
