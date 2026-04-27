"""Tests for enumerator.parse_case_range — generates sequential case numbers."""

import pytest

from scraper.enumerator import parse_case_range


class TestParseCaseRange:
    def test_generates_inclusive_range(self):
        result = parse_case_range("CSM100", "CSM103")
        assert result == ["CSM100", "CSM101", "CSM102", "CSM103"]

    def test_single_case(self):
        result = parse_case_range("CSM100", "CSM100")
        assert result == ["CSM100"]

    def test_preserves_prefix(self):
        result = parse_case_range("ABC001", "ABC003")
        assert all(c.startswith("ABC") for c in result)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="must be >= start"):
            parse_case_range("CSM200", "CSM100")

    def test_no_numeric_part_raises(self):
        with pytest.raises(ValueError, match="No numeric part"):
            parse_case_range("ABCDEF", "ABCDEF")
