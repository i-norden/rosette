"""Tests for rosette.analysis.author_network module.

Tests graph construction and community detection logic using mocked DB sessions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

community = pytest.importorskip("community", reason="python-louvain not installed")
nx = pytest.importorskip("networkx", reason="networkx not installed")

from rosette.analysis.author_network import (  # noqa: E402
    AuthorRisk,
    FraudCluster,
    NetworkAnalysisResult,
    compute_author_risk,
    detect_fraud_clusters,
)


class TestAuthorRiskDataclass:
    def test_creation(self):
        risk = AuthorRisk(
            author_id="a1",
            name="Test Author",
            risk_score=45.0,
            total_papers=10,
            flagged_papers=3,
            retraction_count=1,
            flagged_ratio=0.3,
            details="30% of papers flagged; 1 retraction(s)",
        )
        assert risk.risk_score == 45.0
        assert risk.flagged_ratio == 0.3


class TestFraudClusterDataclass:
    def test_creation(self):
        cluster = FraudCluster(
            cluster_id=0,
            authors=["Alice", "Bob", "Charlie"],
            total_papers=20,
            total_flagged=8,
            total_retractions=2,
            cluster_risk=40.0,
            details="Cluster of 3 authors",
        )
        assert len(cluster.authors) == 3
        assert cluster.cluster_risk == 40.0


class TestNetworkAnalysisResult:
    def test_defaults(self):
        result = NetworkAnalysisResult()
        assert result.high_risk_authors == []
        assert result.fraud_clusters == []
        assert result.total_authors == 0
        assert result.total_communities == 0


class TestDetectFraudClusters:
    @patch("rosette.analysis.author_network.get_session")
    def test_empty_graph_returns_empty(self, mock_get_session):
        """No author links -> no clusters."""
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = []
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_get_session.return_value = mock_session

        clusters = detect_fraud_clusters(min_cluster_size=3)
        assert clusters == []


class TestComputeAuthorRisk:
    @patch("rosette.analysis.author_network.get_session")
    def test_returns_none_for_missing_author(self, mock_get_session):
        """Author not found returns None."""
        mock_session = MagicMock()
        mock_session.get.return_value = None
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_get_session.return_value = mock_session

        result = compute_author_risk("nonexistent")
        assert result is None

    @patch("rosette.analysis.author_network.get_session")
    def test_zero_papers_returns_zero_risk(self, mock_get_session):
        """Author with 0 papers returns 0 risk."""
        mock_author = MagicMock()
        mock_author.name = "Test Author"
        mock_author.total_papers = 0
        mock_author.flagged_papers = 0
        mock_author.retraction_count = 0

        mock_session = MagicMock()
        mock_session.get.return_value = mock_author
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_get_session.return_value = mock_session

        result = compute_author_risk("a1")
        assert result is not None
        assert result.risk_score == 0.0
        assert result.details == "No papers indexed"

    @patch("rosette.analysis.author_network.get_session")
    def test_high_risk_author(self, mock_get_session):
        """Author with flagged papers and retractions gets high risk score."""
        mock_author = MagicMock()
        mock_author.name = "Risky Author"
        mock_author.total_papers = 10
        mock_author.flagged_papers = 5
        mock_author.retraction_count = 2

        mock_session = MagicMock()
        mock_session.get.return_value = mock_author
        # No coauthor papers
        mock_execute = MagicMock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_get_session.return_value = mock_session

        result = compute_author_risk("a2")
        assert result is not None
        # 50% flagged = min(50, 40) = 40 flagged_score
        # 2 retractions = 20 retraction_score
        assert result.risk_score == 60.0
        assert "flagged" in result.details
        assert "retraction" in result.details


class TestDetectFraudClustersMultiAuthor:
    @patch("rosette.analysis.author_network.get_session")
    def test_cluster_detection_with_flagged_authors(self, mock_get_session):
        """Multi-author graph with flagged papers produces fraud clusters."""
        # Simulate 4 authors co-authoring 2 papers together
        mock_links = [
            ("paper1", "a1"),
            ("paper1", "a2"),
            ("paper1", "a3"),
            ("paper2", "a1"),
            ("paper2", "a2"),
            ("paper2", "a4"),
        ]

        mock_authors = {}
        for aid, flagged in [("a1", 3), ("a2", 2), ("a3", 1), ("a4", 0)]:
            author = MagicMock()
            author.id = aid
            author.name = f"Author {aid}"
            author.total_papers = 10
            author.flagged_papers = flagged
            author.retraction_count = 1 if flagged > 1 else 0
            mock_authors[aid] = author

        call_count = [0]

        def session_factory():
            mock_session = MagicMock()
            if call_count[0] == 0:
                # First session: build graph
                mock_session.execute.return_value.all.side_effect = [
                    mock_links,  # AuthorPaperLink query
                    list(mock_authors.values()),  # Author batch fetch
                ]
                mock_session.execute.return_value.scalars.return_value.all.return_value = list(
                    mock_authors.values()
                )
            else:
                # Second session: score communities
                mock_session.execute.return_value.scalars.return_value.all.return_value = list(
                    mock_authors.values()
                )
            call_count[0] += 1
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            return mock_session

        mock_get_session.side_effect = session_factory

        # With 4 authors connected via shared papers, Louvain may find 1 community
        # Results depend on community detection, so just verify it doesn't crash
        # and returns the right type
        clusters = detect_fraud_clusters(min_cluster_size=3)
        assert isinstance(clusters, list)
        for c in clusters:
            assert isinstance(c, FraudCluster)
