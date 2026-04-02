"""Tests for the synchronous rate limiter."""

from __future__ import annotations

import time

import pytest

from scraper.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_call_no_wait(self):
        rl = RateLimiter(min_delay=1.0, max_daily=100)
        start = time.time()
        rl.wait()
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_enforces_delay(self):
        rl = RateLimiter(min_delay=0.2, max_daily=100)
        rl.wait()
        start = time.time()
        rl.wait()
        elapsed = time.time() - start
        assert elapsed >= 0.15

    def test_daily_cap(self):
        rl = RateLimiter(min_delay=0.0, max_daily=2)
        rl.wait()
        rl.wait()
        with pytest.raises(RuntimeError, match="Daily request cap"):
            rl.wait()

    def test_remaining_today(self):
        rl = RateLimiter(min_delay=0.0, max_daily=10)
        assert rl.remaining_today == 10

    def test_remaining_decrements(self):
        rl = RateLimiter(min_delay=0.0, max_daily=10)
        rl.wait()
        rl.wait()
        assert rl.remaining_today == 8

    def test_daily_count(self):
        rl = RateLimiter(min_delay=0.0, max_daily=100)
        assert rl.daily_count == 0
        rl.wait()
        assert rl.daily_count == 1
