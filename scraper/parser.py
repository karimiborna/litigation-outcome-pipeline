"""HTML parsers for SF Superior Court pages.

Extracts structured data from search results and case detail pages.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from data.schemas.case import (
    Attorney,
    CaseMetadata,
    Document,
    Party,
    PartyType,
    Proceeding,
)


def _text(el: Tag | None) -> str:
    """Extract cleaned text from a BeautifulSoup element."""
    if el is None:
        return ""
    return el.get_text(strip=True)


def _parse_date(text: str) -> date | None:
    """Try common court date formats."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Search results (New Filings page)
# ---------------------------------------------------------------------------


class SearchResult:
    """One row from the New Filings search results table."""

    def __init__(self, case_number: str, case_title: str, detail_url: str):
        self.case_number = case_number
        self.case_title = case_title
        self.detail_url = detail_url

    def __repr__(self) -> str:
        return f"SearchResult({self.case_number!r}, {self.case_title!r})"


def parse_search_results(html: str, base_url: str = "") -> list[SearchResult]:
    """Parse the filing search results table into SearchResult objects."""
    soup = BeautifulSoup(html, "lxml")
    results: list[SearchResult] = []

    table = soup.find("table", {"class": re.compile(r"display|dataTable", re.I)})
    if table is None:
        for t in soup.find_all("table"):
            if t.find("a"):
                table = t
                break

    if table is None:
        return results

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        link = cells[0].find("a")
        if link is None:
            continue

        case_number = _text(link)
        case_title = _text(cells[1])
        href = link.get("href", "")
        detail_url = urljoin(base_url, href) if href else ""

        if case_number:
            results.append(SearchResult(case_number, case_title, detail_url))

    return results


def parse_pagination(html: str) -> list[int]:
    """Extract available page numbers from the pagination controls."""
    soup = BeautifulSoup(html, "lxml")
    pages: set[int] = set()

    for link in soup.find_all("a"):
        text = _text(link)
        if text.isdigit():
            pages.add(int(text))

    return sorted(pages)


# ---------------------------------------------------------------------------
# Case detail page
# ---------------------------------------------------------------------------


def parse_case_header(soup: BeautifulSoup) -> dict:
    """Extract case number, title, and cause of action from the detail header."""
    header: dict[str, str | None] = {
        "case_number": None,
        "case_title": None,
        "cause_of_action": None,
    }

    for label_pattern, key in [
        (re.compile(r"case\s*number", re.I), "case_number"),
        (re.compile(r"case\s*title", re.I), "case_title"),
        (re.compile(r"cause\s*of\s*action", re.I), "cause_of_action"),
    ]:
        el = soup.find(string=label_pattern)
        if el and el.parent:
            sibling = el.parent.find_next_sibling()
            if sibling:
                header[key] = _text(sibling)
            else:
                next_text = el.parent.next_sibling
                if next_text:
                    val = str(next_text).strip().strip(":").strip()
                    if val:
                        header[key] = val

    if header["case_number"] is None:
        num_el = soup.find(string=re.compile(r"[A-Z]{2,4}-?\d{2}-?\d{4,}", re.I))
        if num_el:
            header["case_number"] = str(num_el).strip()

    return header


def parse_parties(soup: BeautifulSoup) -> list[Party]:
    """Parse the Parties tab into Party objects."""
    parties: list[Party] = []

    table = _find_tab_table(soup, "parties")
    if table is None:
        return parties

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        name = _text(cells[0])
        type_str = _text(cells[1]).upper()

        try:
            party_type = PartyType(type_str)
        except ValueError:
            party_type = PartyType.OTHER

        attorney_name = _text(cells[2]) if len(cells) > 2 else None
        is_pro_per = bool(attorney_name and re.search(r"pro\s*(per|se)", attorney_name, re.I))

        if name:
            parties.append(
                Party(
                    name=name,
                    party_type=party_type,
                    is_pro_per=is_pro_per or not attorney_name,
                    attorney_name=attorney_name if not is_pro_per else None,
                )
            )

    return parties


def parse_attorneys(soup: BeautifulSoup) -> list[Attorney]:
    """Parse the Attorneys tab."""
    attorneys: list[Attorney] = []

    table = _find_tab_table(soup, "attorneys")
    if table is None:
        return attorneys

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        name = _text(cells[0])
        bar_number = _text(cells[1]) if len(cells) > 1 else None
        address = _text(cells[2]) if len(cells) > 2 else None
        parties_repr = _text(cells[3]) if len(cells) > 3 else ""

        if name:
            attorneys.append(
                Attorney(
                    name=name,
                    bar_number=bar_number or None,
                    address=address or None,
                    parties_represented=[p.strip() for p in parties_repr.split(",") if p.strip()],
                )
            )

    return attorneys


def parse_proceedings(soup: BeautifulSoup) -> list[Proceeding]:
    """Parse the Register of Actions tab."""
    proceedings: list[Proceeding] = []

    table = _find_tab_table(soup, "register")
    if table is None:
        table = _find_tab_table(soup, "action")
    if table is None:
        return proceedings

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_val = _parse_date(_text(cells[0]))
        text = _text(cells[1])
        has_doc = bool(cells[1].find("a") or (len(cells) > 2 and cells[2].find("a")))
        fee = _text(cells[-1]) if len(cells) > 2 else None

        if date_val and text:
            proceedings.append(
                Proceeding(
                    date=date_val,
                    text=text,
                    has_document=has_doc,
                    fee=fee or None,
                )
            )

    return proceedings


def parse_documents(soup: BeautifulSoup, base_url: str = "") -> list[Document]:
    """Parse the Documents tab, extracting PDF links."""
    documents: list[Document] = []

    table = _find_tab_table(soup, "document")
    if table is None:
        return documents

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_val = _parse_date(_text(cells[0]))
        desc_cell = cells[1]
        description = _text(desc_cell)

        link = desc_cell.find("a")
        if link is None and len(cells) > 2:
            link = cells[2].find("a")

        filename = None
        if link:
            href = link.get("href", "")
            if href:
                filename = href.split("/")[-1].split("?")[0]
                if not filename.endswith(".pdf"):
                    filename = f"{filename}.pdf"

        if date_val and description:
            documents.append(
                Document(
                    date=date_val,
                    description=description,
                    filename=filename,
                )
            )

    return documents


def parse_pdf_links(soup: BeautifulSoup, base_url: str = "") -> list[str]:
    """Extract all PDF download URLs from the case detail page."""
    urls: list[str] = []
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        text = _text(link).lower()
        if "view" in text or href.lower().endswith(".pdf") or "document" in href.lower():
            full_url = urljoin(base_url, href)
            if full_url not in urls:
                urls.append(full_url)
    return urls


def parse_case_detail(html: str, filing_date: date, base_url: str = "") -> CaseMetadata:
    """Parse a complete case detail page into a CaseMetadata object."""
    soup = BeautifulSoup(html, "lxml")

    header = parse_case_header(soup)
    parties = parse_parties(soup)
    attorneys = parse_attorneys(soup)
    proceedings = parse_proceedings(soup)
    documents = parse_documents(soup, base_url)

    case_number = header.get("case_number") or "UNKNOWN"

    return CaseMetadata(
        case_number=case_number,
        case_title=header.get("case_title") or "",
        cause_of_action=header.get("cause_of_action"),
        filing_date=filing_date,
        parties=parties,
        attorneys=attorneys,
        proceedings=proceedings,
        documents=documents,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tab_table(soup: BeautifulSoup, tab_keyword: str) -> Tag | None:
    """Find the table associated with a tab by searching for a heading or id."""
    # Try by id/class containing the keyword
    for table in soup.find_all("table"):
        table_id = str(table.get("id", "")).lower()
        table_class = " ".join(str(c) for c in table.get("class", []))
        if tab_keyword in table_id or tab_keyword in table_class.lower():
            return table

    # Try finding a heading with the keyword, then the next table
    heading = soup.find(string=re.compile(tab_keyword, re.I))
    if heading:
        parent = heading.parent
        while parent:
            table = parent.find_next_sibling("table")
            if table:
                return table
            parent = parent.parent

    return None
