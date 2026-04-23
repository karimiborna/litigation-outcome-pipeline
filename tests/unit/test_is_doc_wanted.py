from scraper.config import is_doc_type_wanted


class TestIsDocWanted:
    def test_exact_whitelist_match(self):
        result = is_doc_type_wanted("CLAIM_OF_PLAINTIFF")
        assert result is True

    def test_finds_substring(self):
        result = is_doc_type_wanted("CLAIM_OF_PLAINTIFF and other stuff")
        assert result is True

    def test_rejects_procedural_doc(self):
        result = is_doc_type_wanted("ernie")
        assert result is False
