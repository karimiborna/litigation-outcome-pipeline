#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.index import CaseIndex


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


def build_case_documents(source_dir: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    case_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(source_dir.glob("*.txt")):
        case_id = file_path.name.split("_", 1)[0]
        case_files[case_id].append(file_path)

    texts: list[str] = []
    case_numbers: list[str] = []
    case_titles: list[str] = []
    outcomes: list[str | None] = []

    for case_id, files in sorted(case_files.items()):
        docs = []
        for file_path in files:
            try:
                docs.append(file_path.read_text(encoding="utf-8").strip())
            except UnicodeDecodeError:
                docs.append(file_path.read_text(encoding="latin-1").strip())
        case_text = "\n\n".join([doc for doc in docs if doc])
        if not case_text:
            continue
        texts.append(case_text)
        case_numbers.append(case_id)
        case_titles.append(f"{case_id} ({len(files)} docs)")
        outcomes.append(infer_outcome_from_filenames([p.name for p in files]))

    return texts, case_numbers, case_titles, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the retrieval FAISS index from processed case text files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("scraper/processed"),
        help="Directory containing processed case text files",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/retrieval_index"),
        help="Directory to save the FAISS index and metadata",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional embedding model override",
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    config = RetrievalConfig()
    if args.model:
        config = RetrievalConfig(EMBEDDING_MODEL=args.model)

    print(f"Loading processed texts from {args.source_dir}")
    texts, case_numbers, case_titles, outcomes = build_case_documents(args.source_dir)
    print(f"Found {len(texts)} case documents to index")

    index = CaseIndex(config)
    index.build(texts=texts, case_numbers=case_numbers, case_titles=case_titles, outcomes=outcomes)
    index.save(args.index_dir)
    print(f"Index built and saved to {args.index_dir}")


if __name__ == "__main__":
    main()
