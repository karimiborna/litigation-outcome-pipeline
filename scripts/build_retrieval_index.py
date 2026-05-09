from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import sys
from retrieval.embeddings import EmbeddingModel
from retrieval.config import RetrievalConfig
from retrieval.index import HybridCaseIndex
from retrieval.index import CaseDocument



# =========================================================
# LABEL INFERENCE
# =========================================================

def infer_outcome_from_filenames(filenames: Iterable[str]) -> str | None:
    normalized = [f.upper() for f in filenames]

    if any("JUDGMENT_ON_PLAINTIFF_S_CLAIM" in name for name in normalized):
        return "plaintiff_win"
    if any("JUDGMENT_ON_DEFENDANT_S_CLAIM" in name for name in normalized):
        return "defendant_win"
    if any("DISMISSAL" in name for name in normalized):
        return "dismissal"
    if any("STIPULATION" in name for name in normalized):
        return "settlement"

    return None


# =========================================================
# RAW FILE → CASE DOCUMENTS
# =========================================================

def build_case_documents(source_dir: Path) -> list[CaseDocument]:
    case_files: dict[str, list[Path]] = defaultdict(list)

    for file_path in sorted(source_dir.glob("*.txt")):
        case_id = file_path.name.split("_", 1)[0]
        case_files[case_id].append(file_path)

    documents: list[CaseDocument] = []

    for case_id, files in sorted(case_files.items()):

        chunks: list[str] = []

        for file_path in files:
            try:
                chunks.append(
                    file_path.read_text(encoding="utf-8").strip()
                )
            except UnicodeDecodeError:
                chunks.append(
                    file_path.read_text(encoding="latin-1").strip()
                )

        case_text = "\n\n".join([c for c in chunks if c])

        if not case_text:
            continue

        documents.append(
            CaseDocument(
                id=case_id,
                text=case_text,
                metadata={
                    "case_number": case_id,
                    "case_title": f"{case_id} ({len(files)} docs)",
                    "outcome": infer_outcome_from_filenames(
                        [p.name for p in files]
                    ),
                    "num_files": len(files),
                },
            )
        )

    return documents


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build Hybrid FAISS + BM25 case index"
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("scraper/data/processed"),
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/retrieval_index"),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(
            f"Source directory not found: {args.source_dir}"
        )

    # =====================================================
    # CONFIG
    # =====================================================

    config = RetrievalConfig()

    embedding_model_name = (
        args.model
        or config.embedding_model
    )


    embedding_model = EmbeddingModel(config)
    # =====================================================
    # BUILD DATASET
    # =====================================================

    print(f"Loading case files from {args.source_dir}")

    documents = build_case_documents(args.source_dir)

    print(f"Built {len(documents)} CaseDocument objects")

    # =====================================================
    # BUILD INDEX
    # =====================================================

    index = HybridCaseIndex(embedding_model)

    index.build(documents)

    # =====================================================
    # SAVE SNAPSHOT
    # =====================================================

    index.save(args.index_dir)

    print(f"Index saved to {args.index_dir}")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()