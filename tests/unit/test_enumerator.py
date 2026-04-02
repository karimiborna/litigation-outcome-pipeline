"""Tests for case number enumerator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scraper.enumerator import CaseEnumerator, ValidCasesStore, parse_case_range


class TestParseCaseRange:
    def test_basic_range(self):
        cases = parse_case_range("CSM25870000", "CSM25870003")
        assert cases == [
            "CSM25870000",
            "CSM25870001",
            "CSM25870002",
            "CSM25870003",
        ]

    def test_single_case(self):
        cases = parse_case_range("CSM26871146", "CSM26871146")
        assert cases == ["CSM26871146"]

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="must be >= start"):
            parse_case_range("CSM25870005", "CSM25870002")

    def test_no_numeric_part_raises(self):
        with pytest.raises(ValueError, match="No numeric part"):
            parse_case_range("ABCDEF", "ABCDEF")

    def test_preserves_prefix(self):
        cases = parse_case_range("SC100", "SC102")
        assert cases == ["SC100", "SC101", "SC102"]


class TestValidCasesStore:
    def test_fresh_store(self, tmp_path: Path):
        store = ValidCasesStore(tmp_path / "valid.json")
        assert store.valid_count == 0
        assert store.probed_count == 0

    def test_mark_and_query(self, tmp_path: Path):
        store = ValidCasesStore(tmp_path / "valid.json")
        assert not store.is_probed("CSM001")

        store.mark_probed("CSM001", doc_count=3)
        assert store.is_probed("CSM001")
        assert store.valid_count == 1
        assert store.valid_cases == {"CSM001": 3}

    def test_zero_docs_not_valid(self, tmp_path: Path):
        store = ValidCasesStore(tmp_path / "valid.json")
        store.mark_probed("CSM002", doc_count=0)
        assert store.is_probed("CSM002")
        assert store.valid_count == 0

    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "valid.json"
        store = ValidCasesStore(path)
        store.mark_probed("CSM001", 3)
        store.mark_probed("CSM002", 0)
        store.mark_probed("CSM003", 5)
        store.save()

        store2 = ValidCasesStore(path)
        assert store2.valid_count == 2
        assert store2.probed_count == 3
        assert store2.is_probed("CSM002")
        assert store2.valid_cases == {"CSM001": 3, "CSM003": 5}


class TestCaseEnumerator:
    @patch("scraper.enumerator.probe_case_exists")
    def test_enumerates_and_records(self, mock_probe, tmp_path: Path):
        mock_probe.side_effect = [0, 3, 0, 5]

        config = MagicMock()
        store = ValidCasesStore(tmp_path / "valid.json")
        enumerator = CaseEnumerator(config, "SID", store, probe_delay=0)

        stats = enumerator.enumerate(["CSM001", "CSM002", "CSM003", "CSM004"])

        assert stats["total"] == 4
        assert stats["probed"] == 4
        assert stats["found"] == 2
        assert store.valid_count == 2
        assert store.valid_cases == {"CSM002": 3, "CSM004": 5}

    @patch("scraper.enumerator.probe_case_exists")
    def test_skips_already_probed(self, mock_probe, tmp_path: Path):
        mock_probe.return_value = 0

        config = MagicMock()
        store = ValidCasesStore(tmp_path / "valid.json")
        store.mark_probed("CSM001", 0)
        store.mark_probed("CSM002", 2)

        enumerator = CaseEnumerator(config, "SID", store, probe_delay=0)
        stats = enumerator.enumerate(["CSM001", "CSM002", "CSM003"])

        assert stats["skipped"] == 2
        assert stats["probed"] == 1
        assert mock_probe.call_count == 1

    @patch("scraper.enumerator.probe_case_exists")
    def test_all_already_probed(self, mock_probe, tmp_path: Path):
        config = MagicMock()
        store = ValidCasesStore(tmp_path / "valid.json")
        store.mark_probed("CSM001", 0)

        enumerator = CaseEnumerator(config, "SID", store, probe_delay=0)
        stats = enumerator.enumerate(["CSM001"])

        assert stats["probed"] == 0
        assert stats["skipped"] == 1
        mock_probe.assert_not_called()
