"""Tests for the synchronous rate limiter."""

from __future__ import annotations

import random
import time
from unittest.mock import patch

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


class TestShaping:
    def test_shaping_disabled_when_min_delay_zero(self):
        rl = RateLimiter(min_delay=0.0, max_daily=100)
        # Default jitter_mean falls back to min_delay (0) → no shaping.
        assert rl._shaping_enabled() is False
        # _next_break_at is effectively infinite.
        assert rl._next_break_at >= 1 << 30

    def test_shaping_enabled_when_min_delay_positive(self):
        rl = RateLimiter(min_delay=2.5, max_daily=100)
        assert rl._shaping_enabled() is True
        # Break is scheduled within the configured range.
        lo, hi = rl._break_every
        assert lo <= rl._next_break_at <= hi

    def test_sample_delay_respects_floor_and_cap(self):
        rl = RateLimiter(
            min_delay=2.5,
            max_daily=100,
            jitter_mean=2.5,
            jitter_cap=10.0,
        )
        for _ in range(200):
            d = rl._sample_delay()
            assert d >= 2.5
            assert d <= 10.0

    def test_sample_delay_distribution_is_heavy_tailed(self):
        rl = RateLimiter(
            min_delay=0.5,
            max_daily=100,
            jitter_mean=2.0,
            jitter_cap=12.0,
        )
        random.seed(42)
        samples = [rl._sample_delay() for _ in range(2000)]
        # Heavy tail: at least some samples should land in the upper half.
        assert any(s > 6.0 for s in samples), "no long-tail samples drawn"
        # And the bulk should still be near the floor / mean.
        below_4 = sum(1 for s in samples if s <= 4.0)
        assert below_4 / len(samples) > 0.6

    def test_break_triggers_on_count_threshold(self):
        rl = RateLimiter(
            min_delay=0.01,
            max_daily=100,
            jitter_mean=0.01,
            jitter_cap=0.05,
            break_every=(2, 2),
            break_seconds=(0.05, 0.05),
        )
        with patch.object(time, "sleep") as mock_sleep:
            # Two normal waits — no break yet (next break at count==2 fires
            # *before* the third wait).
            rl.wait()
            rl.wait()
            sleeps_before = [c.args[0] for c in mock_sleep.call_args_list]
            assert all(s < 0.05 for s in sleeps_before), (
                f"unexpected long sleep before break: {sleeps_before}"
            )

            mock_sleep.reset_mock()
            rl.wait()
            sleeps_at_break = [c.args[0] for c in mock_sleep.call_args_list]
            # Break sleep is 0.05s; per-request sleep is bounded by jitter_cap=0.05.
            assert any(abs(s - 0.05) < 1e-6 for s in sleeps_at_break), (
                f"break sleep not observed: {sleeps_at_break}"
            )
            # Next break has been rescheduled past the current count.
            assert rl._next_break_at > rl._daily_count
