"""Integration test for ValidCasesStore — writes/reads JSON to disk."""

import json

from scraper.enumerator import ValidCasesStore


class TestValidCasesStoreRoundTrip:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "cases.json"

        # Create store and add data
        store = ValidCasesStore(path=path)
        store.mark_probed("CSM001", doc_count=3)
        store.mark_probed("CSM002", doc_count=0)
        store.mark_probed("CSM003", doc_count=5)
        store.save()

        # Reload from disk into a new store instance
        store2 = ValidCasesStore(path=path)
        assert store2.valid_count == 2  # CSM001 and CSM003
        assert store2.probed_count == 3
        assert "CSM001" in store2.valid_cases
        assert "CSM002" not in store2.valid_cases

    def test_not_found_cases_tracked(self, tmp_path):
        path = tmp_path / "cases.json"

        store = ValidCasesStore(path=path)
        store.mark_probed("CSM100", doc_count=0)
        store.mark_probed("CSM200", doc_count=5)
        store.save()

        raw = json.loads(path.read_text())
        assert "CSM100" in raw["not_found"]
        assert "CSM200" not in raw["not_found"]

    def test_empty_store_creates_valid_json(self, tmp_path):
        path = tmp_path / "cases.json"
        store = ValidCasesStore(path=path)
        store.save()

        raw = json.loads(path.read_text())
        assert raw["valid"] == {}
        assert raw["probed"] == []
