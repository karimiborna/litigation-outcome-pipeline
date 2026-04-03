"""CLI entry point for the court scraper."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from scraper.config import ScraperConfig
from scraper.court_scraper import CourtScraper, build_date_range
from scraper.manifest import load_manifest, save_manifest

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
def main() -> None:
    """SF Superior Court small claims scraper.

    Requires SFTC_SESSION_ID in your .env file.
    Get it by visiting the court site in your browser and copying from the URL.
    """


@main.command()
@click.option(
    "--date",
    "target_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Single court date to scrape (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start of date range (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End of date range (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--days-back",
    type=int,
    default=120,
    help="Number of days back from today (used if no dates specified).",
)
@click.option("--no-extract", is_flag=True, help="Skip PDF text extraction.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def scrape(
    target_date: date | None,
    start_date: date | None,
    end_date: date | None,
    days_back: int,
    no_extract: bool,
    verbose: bool,
) -> None:
    """Scrape small claims cases and download PDFs.

    \b
    Examples:
      scrape scrape --date 2026-04-01
      scrape scrape --start-date 2026-03-01 --end-date 2026-04-01
      scrape scrape --days-back 30
    """
    _setup_logging(verbose)
    config = ScraperConfig()

    if target_date is not None:
        start_date = target_date
        end_date = target_date
    elif start_date is None or end_date is None:
        s, e = build_date_range(days_back)
        start_date = start_date or s
        end_date = end_date or e

    manifest_path = Path("scraper/state/manifest.json")
    manifest = load_manifest(manifest_path)

    console.print(f"\n[bold]Scraping {start_date} → {end_date}[/bold]")
    console.print(f"  Rate limit: {config.rate_limit_seconds}s between requests")
    console.print(f"  Extract text: {not no_extract}")
    if config.nvidia_api_key:
        console.print("  NVIDIA API key: configured")
    else:
        console.print(
            "  NVIDIA API key: [yellow]not set (scanned PDFs won't be extracted)[/yellow]"
        )
    console.print()

    scraper = CourtScraper(config, manifest, manifest_path)
    stats = scraper.scrape_date_range(start_date, end_date, extract=not no_extract)

    save_manifest(manifest, manifest_path)
    _print_stats(stats)


@main.command()
def status() -> None:
    """Show scraping progress from the manifest."""
    manifest_path = Path("scraper/state/manifest.json")
    manifest = load_manifest(manifest_path)
    summary = manifest.summary()

    table = Table(title="Scrape Progress")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Dates searched", str(summary["dates_searched"]))
    table.add_row("Dates completed", str(summary["dates_completed"]))
    table.add_row("Cases scraped", str(summary["cases_scraped"]))
    table.add_row("Cases extracted", str(summary["cases_extracted"]))

    console.print(table)


@main.command()
@click.option(
    "--case-number",
    type=str,
    required=True,
    help="Case number to extract PDFs for (e.g. CSM26871146).",
)
@click.option("-v", "--verbose", is_flag=True)
def extract(case_number: str, verbose: bool) -> None:
    """Run text extraction on already-downloaded PDFs for a case."""
    _setup_logging(verbose)
    config = ScraperConfig()

    pdf_dir = config.raw_dir / "pdfs"
    txt_dir = config.processed_dir / "extracted"

    pdfs = sorted(pdf_dir.glob(f"{case_number}_*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found for case {case_number}[/yellow]")
        raise SystemExit(1)

    console.print(f"Extracting text from {len(pdfs)} PDF(s) for {case_number}")
    txt_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    for pdf_path in pdfs:
        txt_path = txt_dir / f"{pdf_path.stem}.txt"
        if txt_path.exists():
            console.print(f"  Already extracted: {txt_path.name}")
            extracted_count += 1
            continue

        from scraper.extractor import extract_text

        text = extract_text(pdf_path, config.nvidia_api_key)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            extracted_count += 1
            console.print(f"  [green]Saved: {txt_path.name}[/green]")
        else:
            console.print(f"  [red]No text extracted: {pdf_path.name}[/red]")

    manifest_path = Path("scraper/state/manifest.json")
    manifest = load_manifest(manifest_path)
    manifest.mark_case_extracted(case_number, extracted_count)
    save_manifest(manifest, manifest_path)

    console.print(f"\n[green]Extracted {extracted_count}/{len(pdfs)} PDFs[/green]")


@main.command()
@click.option(
    "--start",
    required=True,
    help="First case number in range (e.g. CSM25870000).",
)
@click.option(
    "--end",
    required=True,
    help="Last case number in range (e.g. CSM26879999).",
)
@click.option(
    "--delay",
    type=float,
    default=1.0,
    help="Seconds between probes (default 1.0).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def enumerate(start: str, end: str, delay: float, verbose: bool) -> None:
    """Probe a range of case numbers to find which ones exist.

    \b
    Saves valid case numbers to scraper/state/valid_cases.json.
    Supports resume — re-running skips already-probed numbers.

    \b
    Examples:
      scrape enumerate --start CSM25870000 --end CSM25879999
      scrape enumerate --start CSM26870000 --end CSM26879999 --delay 0.5
    """
    _setup_logging(verbose)

    from scraper.config import ScraperConfig
    from scraper.enumerator import CaseEnumerator, ValidCasesStore, parse_case_range
    from scraper.session import get_session_id

    config = ScraperConfig()
    session_id = get_session_id(config)

    case_numbers = parse_case_range(start, end)
    console.print(f"\n[bold]Enumerating {len(case_numbers)} case numbers[/bold]")
    console.print(f"  Range: {start} → {end}")
    console.print(f"  Probe delay: {delay}s")
    console.print()

    store = ValidCasesStore()
    enumerator = CaseEnumerator(config, session_id, store, probe_delay=delay)
    stats = enumerator.enumerate(case_numbers)

    table = Table(title="Enumeration Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total in range", str(stats["total"]))
    table.add_row("Already probed", str(stats["skipped"]))
    table.add_row("Newly probed", str(stats["probed"]))
    table.add_row("Valid cases found", str(stats["found"]))
    table.add_row("Total valid (all time)", str(store.valid_count))
    console.print(table)


@main.command(name="download-cases")
@click.option("--no-extract", is_flag=True, help="Skip PDF text extraction.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def download_cases(no_extract: bool, verbose: bool) -> None:
    """Download PDFs for all valid cases found by enumerate.

    \b
    Reads case numbers from scraper/state/valid_cases.json
    and downloads their documents using the existing pipeline.

    \b
    Examples:
      scrape download-cases
      scrape download-cases --no-extract
    """
    _setup_logging(verbose)

    from scraper.config import ScraperConfig, is_doc_type_wanted
    from scraper.court_api import download_pdf, get_documents, sanitize_description
    from scraper.enumerator import ValidCasesStore
    from scraper.extractor import extract_text
    from scraper.rate_limiter import RateLimiter
    from scraper.session import SessionExpiredError, get_session_id, prompt_refresh

    config = ScraperConfig()
    session_id = get_session_id(config)
    store = ValidCasesStore()
    valid = store.valid_cases

    if not valid:
        console.print("[yellow]No valid cases found. Run 'scrape enumerate' first.[/yellow]")
        raise SystemExit(1)

    console.print(f"\n[bold]Downloading PDFs for {len(valid)} cases[/bold]")
    console.print(f"  Rate limit: {config.rate_limit_seconds}s between requests")
    console.print(f"  Extract text: {not no_extract}")
    console.print("  Document filter: [green]enabled[/green] (skipping procedural docs)")
    console.print()

    import requests as req

    http = req.Session()
    http.headers.update({"User-Agent": config.user_agent})
    rate_limiter = RateLimiter(
        min_delay=config.rate_limit_seconds,
        max_daily=config.max_daily_requests,
    )

    pdf_dir = config.raw_dir / "pdfs"
    txt_dir = config.processed_dir / "extracted"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("scraper/state/manifest.json")
    manifest = load_manifest(manifest_path)

    total_pdfs = 0
    cases_done = 0

    for case_num in sorted(valid.keys()):
        if manifest.is_case_scraped(case_num):
            cases_done += 1
            continue

        try:
            rate_limiter.wait()
            docs = get_documents(case_num, session_id, config)
        except SessionExpiredError:
            session_id = prompt_refresh()
            rate_limiter.wait()
            docs = get_documents(case_num, session_id, config)

        if not docs:
            console.print(f"  {case_num}: no documents")
            manifest.mark_case_scraped(case_num, "", date.today(), 0)
            cases_done += 1
            continue

        pdf_count = 0
        for doc in docs:
            desc = doc.get("DESCRIPTION", "doc")
            doc_url = doc.get("URL", "")
            if not doc_url:
                continue

            if not is_doc_type_wanted(desc):
                continue

            safe_desc = sanitize_description(desc)
            pdf_path = pdf_dir / f"{case_num}_{safe_desc}.pdf"

            if pdf_path.exists():
                pdf_count += 1
                continue

            rate_limiter.wait()
            if download_pdf(doc_url, pdf_path, http, config.pdf_download_timeout):
                pdf_count += 1
                if not no_extract:
                    txt_path = txt_dir / f"{pdf_path.stem}.txt"
                    if not txt_path.exists():
                        text = extract_text(pdf_path, config.nvidia_api_key)
                        if text:
                            txt_path.write_text(text, encoding="utf-8")

        manifest.mark_case_scraped(case_num, "", date.today(), pdf_count)
        if not no_extract and pdf_count > 0:
            manifest.mark_case_extracted(case_num, pdf_count)
        total_pdfs += pdf_count
        cases_done += 1

        if cases_done % 10 == 0:
            save_manifest(manifest, manifest_path)
            console.print(
                f"  Progress: {cases_done}/{len(valid)} cases, {total_pdfs} PDFs"
            )

    save_manifest(manifest, manifest_path)

    table = Table(title="Download Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Cases processed", str(cases_done))
    table.add_row("PDFs downloaded", str(total_pdfs))
    console.print(table)


def _print_stats(stats: dict) -> None:
    table = Table(title="Scrape Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Dates processed", str(stats["dates_processed"]))
    table.add_row("Cases scraped", str(stats["cases_scraped"]))
    table.add_row("PDFs downloaded", str(stats["pdfs_downloaded"]))
    table.add_row("Errors", str(stats["errors"]))

    console.print(table)


if __name__ == "__main__":
    main()
