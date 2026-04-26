"""Rate limiter and daily request cap for respectful scraping.

Adds heavy-tailed jitter and periodic "reading breaks" so the request cadence
looks like a human reading court documents rather than a bot pulling a uniform
stream of requests. When ``min_delay`` is 0 the shaping is disabled so tests
stay fast.
"""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Enforces minimum delay, daily cap, and human-like cadence shaping.

    The delay between requests is sampled from an exponential distribution
    (heavy-tailed — most short, occasional long), floored at ``min_delay`` and
    capped at ``jitter_cap``. Every ``break_every`` requests the limiter sleeps
    a longer ``break_seconds`` "reading break" to model the human pattern of
    clicking a few items, then pausing.
    """

    def __init__(
        self,
        min_delay: float = 2.5,
        max_daily: int = 200,
        jitter_mean: float | None = None,
        jitter_cap: float = 12.0,
        break_every: tuple[int, int] = (15, 30),
        break_seconds: tuple[float, float] = (60.0, 180.0),
    ):
        self._min_delay = min_delay
        self._max_daily = max_daily
        self._jitter_mean = jitter_mean if jitter_mean is not None else min_delay
        self._jitter_cap = jitter_cap
        self._break_every = break_every
        self._break_seconds = break_seconds

        self._last_request_time: float = 0.0
        self._daily_count: int = 0
        self._day_start: float = time.time()
        self._next_break_at: int = self._schedule_next_break(start=0)

    @property
    def daily_count(self) -> int:
        return self._daily_count

    @property
    def remaining_today(self) -> int:
        self._maybe_reset_day()
        return max(0, self._max_daily - self._daily_count)

    def _maybe_reset_day(self) -> None:
        elapsed = time.time() - self._day_start
        if elapsed >= 86400:
            self._daily_count = 0
            self._day_start = time.time()
            self._next_break_at = self._schedule_next_break(start=0)

    def _shaping_enabled(self) -> bool:
        return self._jitter_mean > 0.0

    def _sample_delay(self) -> float:
        if not self._shaping_enabled():
            return self._min_delay
        sample = random.expovariate(1.0 / self._jitter_mean)
        return max(self._min_delay, min(self._jitter_cap, sample))

    def _schedule_next_break(self, start: int) -> int:
        if not self._shaping_enabled():
            return 1 << 30
        lo, hi = self._break_every
        return start + random.randint(lo, hi)

    def wait(self) -> None:
        """Block until it's safe to make the next request.

        Raises RuntimeError if the daily cap has been hit.
        """
        self._maybe_reset_day()
        if self._daily_count >= self._max_daily:
            raise RuntimeError(
                f"Daily request cap reached ({self._max_daily}). "
                f"Resume tomorrow or increase SCRAPER_MAX_DAILY_REQUESTS."
            )

        if self._shaping_enabled() and self._daily_count >= self._next_break_at:
            break_delay = random.uniform(self._break_seconds[0], self._break_seconds[1])
            logger.debug("Rate limiter: reading break of %.1fs", break_delay)
            time.sleep(break_delay)
            self._next_break_at = self._schedule_next_break(start=self._daily_count)

        now = time.time()
        elapsed = now - self._last_request_time
        needed = self._sample_delay()
        if elapsed < needed:
            wait_s = needed - elapsed
            logger.debug("Rate limiter: waiting %.1fs", wait_s)
            time.sleep(wait_s)

        self._last_request_time = time.time()
        self._daily_count += 1
