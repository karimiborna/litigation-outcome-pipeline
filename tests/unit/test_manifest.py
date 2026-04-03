"""Tests for scrape manifest (resume support)."""

from datetime import date
from pathlib import Path

from scraper.manifest import ScrapeManifest, load_manifest, save_manifest


class TestScrapeManifest:
    def test_fresh_manifest(self):
        m = ScrapeManifest()
        assert m.summary() == {
            "dates_searched": 0,
            "dates_completed": 0,
            "cases_scraped": 0,
            "cases_extracted": 0,
        }

    def test_date_tracking(self):
        m = ScrapeManifest()
        d = date(2026, 1, 15)

        assert not m.is_date_completed(d)
        m.mark_date_searched(d, total_cases=5)
        assert not m.is_date_completed(d)
        m.mark_date_completed(d)
        assert m.is_date_completed(d)

    def test_case_tracking(self):
        m = ScrapeManifest()
        cn = "SC26001"

        assert not m.is_case_scraped(cn)
        m.mark_case_scraped(cn, "A vs B", date(2026, 1, 15), pdfs_downloaded=3)
        assert m.is_case_scraped(cn)

        assert not m.is_case_extracted(cn)
        m.mark_case_extracted(cn, pdfs_extracted=2)
        assert m.is_case_extracted(cn)

    def test_summary(self):
        m = ScrapeManifest()
        m.mark_date_searched(date(2026, 1, 1), 2)
        m.mark_date_completed(date(2026, 1, 1))
        m.mark_case_scraped("SC001", "A vs B", date(2026, 1, 1), 1)
        m.mark_case_scraped("SC002", "C vs D", date(2026, 1, 1), 2)
        m.mark_case_extracted("SC001", 1)

        s = m.summary()
        assert s["dates_searched"] == 1
        assert s["dates_completed"] == 1
        assert s["cases_scraped"] == 2
        assert s["cases_extracted"] == 1


class TestManifestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        m = ScrapeManifest()
        m.mark_case_scraped("SC001", "A vs B", date(2026, 1, 1), 2)
        save_manifest(m, path)

        loaded = load_manifest(path)
        assert loaded.is_case_scraped("SC001")

    def test_load_nonexistent(self, tmp_path: Path):
        path = tmp_path / "does_not_exist.json"
        m = load_manifest(path)
        assert m.summary()["cases_scraped"] == 0
