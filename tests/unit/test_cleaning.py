"""Tests for text cleaning utilities."""

from data.cleaning import (
    clean_extracted_text,
    collapse_whitespace,
    merge_page_texts,
    normalize_unicode,
    remove_header_footer_noise,
    remove_ocr_artifacts,
)


class TestNormalizeUnicode:
    def test_fullwidth_chars(self):
        assert normalize_unicode("\uff21\uff22\uff23") == "ABC"

    def test_already_normalized(self):
        text = "Hello world"
        assert normalize_unicode(text) == text


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        assert collapse_whitespace("hello   world") == "hello world"

    def test_multiple_newlines(self):
        result = collapse_whitespace("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_tabs_and_spaces(self):
        assert collapse_whitespace("hello\t\t  world") == "hello world"

    def test_leading_trailing(self):
        assert collapse_whitespace("  hello  ") == "hello"


class TestRemoveOcrArtifacts:
    def test_repeated_dots(self):
        result = remove_ocr_artifacts("Section........end")
        assert "........" not in result

    def test_page_numbers(self):
        result = remove_ocr_artifacts("Some text Page 1 of 3 more text")
        assert "Page 1 of 3" not in result

    def test_normal_text_untouched(self):
        text = "The defendant owes $500 to the plaintiff."
        assert remove_ocr_artifacts(text) == text


class TestRemoveHeaderFooterNoise:
    def test_electronically_filed(self):
        result = remove_header_footer_noise("Text before ELECTRONICALLY FILED on 01/01/2026")
        assert "ELECTRONICALLY" not in result

    def test_normal_text_untouched(self):
        text = "The court finds in favor of the plaintiff."
        assert remove_header_footer_noise(text) == text


class TestCleanExtractedText:
    def test_basic_cleaning(self):
        raw = "  Hello\uff21   world..........end  "
        result = clean_extracted_text(raw)
        assert "HelloA" in result
        assert ".........." not in result

    def test_aggressive_mode(self):
        raw = "Electronically filed 01/01/2026\nActual content here"
        result = clean_extracted_text(raw, aggressive=True)
        assert "Electronically" not in result
        assert "Actual content here" in result


class TestMergePageTexts:
    def test_merge_two_pages(self):
        pages = ["Page one content", "Page two content"]
        result = merge_page_texts(pages)
        assert "Page one content" in result
        assert "Page two content" in result
        assert "\n\n" in result

    def test_empty_pages_filtered(self):
        pages = ["Content", "", "  ", "More content"]
        result = merge_page_texts(pages)
        assert result == "Content\n\nMore content"
