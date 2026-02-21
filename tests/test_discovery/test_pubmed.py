"""Tests for PubMed discovery provider."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from rosette.discovery.pubmed import (
    _parse_article_xml,
    fetch_details,
    search_pubmed,
)


SAMPLE_ARTICLE_XML = """\
<PubmedArticle>
  <MedlineCitation>
    <PMID>12345678</PMID>
    <Article>
      <ArticleTitle>Test Article Title</ArticleTitle>
      <Abstract>
        <AbstractText>This is the abstract text.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author>
          <LastName>Smith</LastName>
          <ForeName>John</ForeName>
          <AffiliationInfo>
            <Affiliation>University of Testing</Affiliation>
          </AffiliationInfo>
        </Author>
      </AuthorList>
      <Journal>
        <Title>Journal of Tests</Title>
        <ISSN>1234-5678</ISSN>
        <JournalIssue>
          <Volume>10</Volume>
          <Issue>2</Issue>
          <PubDate><Year>2024</Year></PubDate>
        </JournalIssue>
      </Journal>
      <Pagination><MedlinePgn>100-110</MedlinePgn></Pagination>
      <ELocationID EIdType="doi">10.1234/test.2024</ELocationID>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pmc">PMC9999999</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
"""


class TestParseArticleXml:
    def test_parse_full_article(self) -> None:
        elem = ET.fromstring(SAMPLE_ARTICLE_XML)
        article = _parse_article_xml(elem)
        assert article is not None
        assert article.pmid == "12345678"
        assert article.title == "Test Article Title"
        assert article.abstract == "This is the abstract text."
        assert len(article.authors) == 1
        assert article.authors[0].last_name == "Smith"
        assert article.authors[0].first_name == "John"
        assert article.authors[0].affiliation == "University of Testing"
        assert article.journal == "Journal of Tests"
        assert article.issn == "1234-5678"
        assert article.doi == "10.1234/test.2024"
        assert article.pmcid == "PMC9999999"
        assert article.publication_year == 2024
        assert article.volume == "10"
        assert article.issue == "2"
        assert article.pages == "100-110"

    def test_parse_minimal_article(self) -> None:
        xml = "<PubmedArticle><MedlineCitation><PMID>99999</PMID></MedlineCitation></PubmedArticle>"
        elem = ET.fromstring(xml)
        article = _parse_article_xml(elem)
        assert article is not None
        assert article.pmid == "99999"
        assert article.title is None

    def test_parse_missing_medline(self) -> None:
        xml = "<PubmedArticle></PubmedArticle>"
        elem = ET.fromstring(xml)
        assert _parse_article_xml(elem) is None

    def test_parse_labeled_abstract(self) -> None:
        xml = """\
        <PubmedArticle>
          <MedlineCitation>
            <PMID>11111</PMID>
            <Article>
              <Abstract>
                <AbstractText Label="BACKGROUND">Background text.</AbstractText>
                <AbstractText Label="METHODS">Methods text.</AbstractText>
              </Abstract>
            </Article>
          </MedlineCitation>
        </PubmedArticle>
        """
        elem = ET.fromstring(xml)
        article = _parse_article_xml(elem)
        assert article is not None
        assert "BACKGROUND: Background text." in article.abstract
        assert "METHODS: Methods text." in article.abstract


class TestSearchPubmed:
    @pytest.mark.asyncio
    async def test_search_returns_empty_on_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        results = await search_pubmed("cancer biomarkers")
        assert results == []


class TestFetchDetails:
    @pytest.mark.asyncio
    async def test_fetch_empty_list(self) -> None:
        results = await fetch_details([])
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_on_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        results = await fetch_details(["12345"])
        assert results == []
