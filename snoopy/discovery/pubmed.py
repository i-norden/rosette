"""PubMed/PMC E-Utilities client for searching and fetching paper metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import defusedxml.ElementTree as ET  # type: ignore[import-untyped]

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"
DEFAULT_TIMEOUT = 30.0
DEFAULT_LIMIT = 20


@dataclass
class PubMedAuthor:
    """Author metadata from PubMed."""

    last_name: str
    first_name: str
    affiliation: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class PubMedArticle:
    """Article metadata parsed from PubMed efetch XML."""

    pmid: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[PubMedAuthor] = field(default_factory=list)
    journal: Optional[str] = None
    issn: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    publication_year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None


def _build_params(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build query parameters, inserting the API key if available."""
    params: dict[str, Any] = dict(kwargs)
    if api_key:
        params["api_key"] = api_key
    return params


async def search_pubmed(
    query: str,
    limit: int = DEFAULT_LIMIT,
    api_key: Optional[str] = None,
) -> list[str]:
    """Search PubMed via esearch.fcgi and return a list of PMIDs.

    Args:
        query: PubMed search query string.
        limit: Maximum number of PMIDs to return.
        api_key: Optional NCBI API key for higher rate limits.

    Returns:
        List of PMID strings.
    """
    params = _build_params(
        api_key=api_key,
        db="pubmed",
        term=query,
        retmax=limit,
        retmode="json",
        sort="relevance",
    )

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}esearch.fcgi",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "PubMed esearch HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return []
    except httpx.HTTPError as exc:
        logger.error("PubMed esearch request failed: %s", exc)
        return []

    esearch_result = data.get("esearchresult", {})
    return esearch_result.get("idlist", [])


def _parse_article_xml(article_elem: ET.Element) -> Optional[PubMedArticle]:
    """Parse a single PubmedArticle XML element into a PubMedArticle."""
    medline = article_elem.find("MedlineCitation")
    if medline is None:
        return None

    pmid_elem = medline.find("PMID")
    if pmid_elem is None or pmid_elem.text is None:
        return None

    pmid = pmid_elem.text
    article = medline.find("Article")
    if article is None:
        return PubMedArticle(pmid=pmid)

    # Title
    title_elem = article.find("ArticleTitle")
    title = title_elem.text if title_elem is not None else None

    # Abstract
    abstract_elem = article.find("Abstract")
    abstract: Optional[str] = None
    if abstract_elem is not None:
        abstract_parts: list[str] = []
        for abs_text in abstract_elem.findall("AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else None

    # Authors
    authors: list[PubMedAuthor] = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for author_elem in author_list.findall("Author"):
            last_name_elem = author_elem.find("LastName")
            first_name_elem = author_elem.find("ForeName")
            if last_name_elem is None:
                continue
            affiliation_elem = author_elem.find("AffiliationInfo/Affiliation")
            authors.append(
                PubMedAuthor(
                    last_name=last_name_elem.text or "",
                    first_name=(first_name_elem.text or "" if first_name_elem is not None else ""),
                    affiliation=(affiliation_elem.text if affiliation_elem is not None else None),
                )
            )

    # Journal info
    journal_elem = article.find("Journal")
    journal_name: Optional[str] = None
    issn: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pub_year: Optional[int] = None

    if journal_elem is not None:
        journal_title = journal_elem.find("Title")
        if journal_title is not None:
            journal_name = journal_title.text

        issn_elem = journal_elem.find("ISSN")
        if issn_elem is not None:
            issn = issn_elem.text

        ji = journal_elem.find("JournalIssue")
        if ji is not None:
            vol_elem = ji.find("Volume")
            volume = vol_elem.text if vol_elem is not None else None
            issue_elem = ji.find("Issue")
            issue = issue_elem.text if issue_elem is not None else None

            pub_date = ji.find("PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    try:
                        pub_year = int(year_elem.text)
                    except ValueError:
                        pass

    # Pages
    pagination = article.find("Pagination/MedlinePgn")
    pages = pagination.text if pagination is not None else None

    # DOI and PMCID from article IDs
    doi: Optional[str] = None
    pmcid: Optional[str] = None

    # Check ELocationID for DOI
    for eloc in article.findall("ELocationID"):
        if eloc.get("EIdType") == "doi":
            doi = eloc.text

    # Check PubmedData for article IDs
    pubmed_data = article_elem.find("PubmedData")
    if pubmed_data is not None:
        id_list = pubmed_data.find("ArticleIdList")
        if id_list is not None:
            for article_id in id_list.findall("ArticleId"):
                id_type = article_id.get("IdType", "")
                if id_type == "doi" and doi is None:
                    doi = article_id.text
                elif id_type == "pmc":
                    pmcid = article_id.text

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal_name,
        issn=issn,
        doi=doi,
        pmcid=pmcid,
        publication_year=pub_year,
        volume=volume,
        issue=issue,
        pages=pages,
    )


async def fetch_details(
    pmids: list[str],
    api_key: Optional[str] = None,
) -> list[PubMedArticle]:
    """Fetch detailed metadata for a list of PMIDs via efetch.fcgi.

    Args:
        pmids: List of PubMed ID strings.
        api_key: Optional NCBI API key.

    Returns:
        List of PubMedArticle instances.
    """
    if not pmids:
        return []

    params = _build_params(
        api_key=api_key,
        db="pubmed",
        id=",".join(pmids),
        retmode="xml",
        rettype="abstract",
    )

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}efetch.fcgi",
                params=params,
            )
            response.raise_for_status()
            xml_text = response.text
    except httpx.HTTPStatusError as exc:
        logger.error(
            "PubMed efetch HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return []
    except httpx.HTTPError as exc:
        logger.error("PubMed efetch request failed: %s", exc)
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse PubMed efetch XML: %s", exc)
        return []

    articles: list[PubMedArticle] = []
    for article_elem in root.findall("PubmedArticle"):
        parsed = _parse_article_xml(article_elem)
        if parsed is not None:
            articles.append(parsed)

    return articles


async def get_pmc_pdf_url(
    pmcid: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Attempt to resolve an open-access PDF URL from PubMed Central.

    Uses the PMC OA web service to check for a PDF link.

    Args:
        pmcid: PubMed Central ID (e.g. "PMC1234567").
        api_key: Optional NCBI API key.

    Returns:
        URL string for the PDF, or None if unavailable.
    """
    if not pmcid:
        return None

    # Normalize: ensure "PMC" prefix
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    # Try the OA service endpoint
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    params: dict[str, str] = {"id": pmcid}
    if api_key:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(oa_url, params=params)
            response.raise_for_status()
            xml_text = response.text
    except httpx.HTTPStatusError as exc:
        logger.error(
            "PMC OA service HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return None
    except httpx.HTTPError as exc:
        logger.error("PMC OA service request failed: %s", exc)
        return None

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse PMC OA XML: %s", exc)
        return None

    # Look for PDF link in the OA response
    for link in root.iter("link"):
        fmt = link.get("format", "")
        if fmt.lower() == "pdf":
            href = link.get("href")
            if href:
                # Convert FTP links to HTTPS if needed
                if href.startswith("ftp://"):
                    href = href.replace("ftp://", "https://", 1)
                return href

    return None
