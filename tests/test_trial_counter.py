"""Tests for trial counter."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from sparky.tracking.trial_counter import TrialCounter


@pytest.fixture
def counter(tmp_path):
    return TrialCounter(tmp_path / "trials.jsonl", "p002")


class TestRecordTrial:
    def test_appends_to_file(self, counter):
        counter.record_trial("garch", {"window": 252}, 1.5, True)
        assert counter.log_path.exists()
        lines = counter.log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["strategy_family"] == "garch"
        assert entry["sharpe"] == 1.5
        assert entry["passed"] is True
        assert entry["project_id"] == "p002"

    def test_append_only(self, counter):
        counter.record_trial("garch", {"window": 126}, 0.8, False)
        counter.record_trial("garch", {"window": 252}, 1.5, True)
        counter.record_trial("carry", {"threshold": 0.05}, 0.3, False)
        lines = counter.log_path.read_text().strip().split("\n")
        assert len(lines) == 3


class TestCount:
    def test_empty(self, counter):
        assert counter.count() == 0

    def test_total_count(self, counter):
        counter.record_trial("garch", {}, 1.0, True)
        counter.record_trial("carry", {}, 0.5, False)
        assert counter.count() == 2

    def test_family_filter(self, counter):
        counter.record_trial("garch", {}, 1.0, True)
        counter.record_trial("garch", {}, 1.2, True)
        counter.record_trial("carry", {}, 0.5, False)
        assert counter.count("garch") == 2
        assert counter.count("carry") == 1
        assert counter.count("unknown") == 0

    def test_project_filter(self, tmp_path):
        path = tmp_path / "shared.jsonl"
        c1 = TrialCounter(path, "p001")
        c2 = TrialCounter(path, "p002")
        c1.record_trial("garch", {}, 1.0, True)
        c2.record_trial("garch", {}, 1.5, True)
        c2.record_trial("carry", {}, 0.8, True)
        assert c1.count() == 1
        assert c2.count() == 2


class TestBestSharpe:
    def test_empty(self, counter):
        assert counter.best_sharpe() == float("-inf")

    def test_finds_best(self, counter):
        counter.record_trial("garch", {}, 0.8, False)
        counter.record_trial("garch", {}, 1.5, True)
        counter.record_trial("carry", {}, 1.2, True)
        assert counter.best_sharpe() == 1.5

    def test_family_filter(self, counter):
        counter.record_trial("garch", {}, 1.5, True)
        counter.record_trial("carry", {}, 2.0, True)
        assert counter.best_sharpe("garch") == 1.5
        assert counter.best_sharpe("carry") == 2.0


class TestDsrN:
    def test_returns_total_count(self, counter):
        counter.record_trial("garch", {}, 1.0, True)
        counter.record_trial("carry", {}, 0.5, False)
        counter.record_trial("garch", {}, 1.2, True)
        assert counter.get_dsr_n() == 3


class TestConcurrentSafety:
    def test_concurrent_writes(self, tmp_path):
        path = tmp_path / "concurrent.jsonl"

        def write_trial(i):
            c = TrialCounter(path, "p002")
            c.record_trial("garch", {"i": i}, float(i), True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_trial, range(50)))

        c = TrialCounter(path, "p002")
        assert c.count() == 50
