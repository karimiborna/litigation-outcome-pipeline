"""Rate limiter and daily request cap for respectful scraping."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Enforces a minimum delay between requests and a daily request cap."""

    def __init__(self, min_delay: float = 2.5, max_daily: int = 200):
        self._min_delay = min_delay
        self._max_daily = max_daily
        self._last_request_time: float = 0.0
        self._daily_count: int = 0
        self._day_start: float = time.time()

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

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_delay:
            wait = self._min_delay - elapsed
            logger.debug("Rate limiter: waiting %.1fs", wait)
            time.sleep(wait)

        self._last_request_time = time.time()
        self._daily_count += 1
