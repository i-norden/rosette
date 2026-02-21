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
