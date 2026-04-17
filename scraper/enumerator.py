"""Case number enumerator for discovering historical cases.

The court calendar only keeps recent dates. To find older cases, we
brute-force case number ranges — probing each one via the API to see
if it exists. Valid case numbers are saved to a JSON file for later
bulk download.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from scraper.config import ScraperConfig
from scraper.court_api import probe_case_exists
from scraper.session import SessionExpiredError, prompt_refresh

logger = logging.getLogger(__name__)

DEFAULT_VALID_CASES_PATH = Path("scraper/state/valid_cases.json")
PROBE_DELAY_SECONDS = 1.0
PROGRESS_LOG_INTERVAL = 100


def parse_case_range(start: str, end: str) -> list[str]:
    """Generate all case numbers between start and end (inclusive).

    Expects format like CSM25870000. Extracts the numeric suffix,
    iterates through the range, and re-applies the prefix.
    """
    prefix = ""
    for i, ch in enumerate(start):
        if ch.isdigit():
            prefix = start[:i]
            break
    else:
        raise ValueError(f"No numeric part found in case number: {start}")

    start_num = int(start[len(prefix) :])
    end_num = int(end[len(prefix) :])

    if end_num < start_num:
        raise ValueError(f"End ({end}) must be >= start ({start})")

    return [f"{prefix}{n}" for n in range(start_num, end_num + 1)]


class ValidCasesStore:
    """Persists the set of valid (existing) case numbers to disk."""

    def __init__(self, path: Path = DEFAULT_VALID_CASES_PATH):
        self._path = path
        self._valid: dict[str, int] = {}
        self._probed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._valid = raw.get("valid", {})
                self._probed = set(raw.get("probed", []))
                logger.info(
                    "Loaded %d valid cases, %d already probed",
                    len(self._valid),
                    len(self._probed),
                )
            except Exception:
                logger.exception("Failed to load valid cases, starting fresh")

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        not_found = sorted(self._probed - set(self._valid.keys()))
        data = {
            "valid": self._valid,
            "not_found": not_found,
            "probed": sorted(self._probed),
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_probed(self, case_num: str) -> bool:
        return case_num in self._probed

    def mark_probed(self, case_num: str, doc_count: int) -> None:
        self._probed.add(case_num)
        if doc_count > 0:
            self._valid[case_num] = doc_count

    @property
    def valid_cases(self) -> dict[str, int]:
        return dict(self._valid)

    @property
    def probed_count(self) -> int:
        return len(self._probed)

    @property
    def valid_count(self) -> int:
        return len(self._valid)


class CaseEnumerator:
    """Probes a range of case numbers to find which ones exist."""

    def __init__(
        self,
        config: ScraperConfig,
        session_id: str,
        store: ValidCasesStore,
        probe_delay: float = PROBE_DELAY_SECONDS,
    ):
        self._config = config
        self._session_id = session_id
        self._store = store
        self._probe_delay = probe_delay

    def enumerate(self, case_numbers: list[str]) -> dict:
        """Probe each case number and record results.

        Returns a summary dict with counts.
        """
        total = len(case_numbers)
        to_probe = [cn for cn in case_numbers if not self._store.is_probed(cn)]

        if not to_probe:
            logger.info("All %d case numbers already probed.", total)
            return {"total": total, "probed": 0, "found": 0, "skipped": total}

        logger.info(
            "%d case numbers to probe (%d already done, %d remaining)",
            total,
            total - len(to_probe),
            len(to_probe),
        )

        stats = {"total": total, "probed": 0, "found": 0, "skipped": total - len(to_probe)}

        for i, case_num in enumerate(to_probe):
            try:
                time.sleep(self._probe_delay)
                doc_count = probe_case_exists(case_num, self._session_id, self._config)
                self._store.mark_probed(case_num, doc_count)
                stats["probed"] += 1

                if doc_count > 0:
                    stats["found"] += 1
                    logger.info("  FOUND: %s (%d entries)", case_num, doc_count)
                else:
                    logger.info("  probing %s ... not found", case_num)

            except SessionExpiredError:
                self._store.save()
                self._session_id = prompt_refresh()
                from scraper.session_manager import start_keepalive
                start_keepalive(self._session_id)
                time.sleep(self._probe_delay)
                doc_count = probe_case_exists(case_num, self._session_id, self._config)
                self._store.mark_probed(case_num, doc_count)
                stats["probed"] += 1
                if doc_count > 0:
                    stats["found"] += 1
                    logger.info("  FOUND: %s (%d entries)", case_num, doc_count)

            except Exception:
                logger.debug("Probe error for %s, skipping", case_num)

            if (i + 1) % PROGRESS_LOG_INTERVAL == 0:
                self._store.save()
                done = stats["skipped"] + stats["probed"]
                logger.info(
                    "Progress: %d/%d probed, %d valid found so far",
                    done,
                    total,
                    self._store.valid_count,
                )

        self._store.save()
        return stats
