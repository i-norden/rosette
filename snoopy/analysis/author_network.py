"""Author network analysis for detecting fraud rings and prolific fraudsters.

Builds co-authorship graphs and applies community detection to find clusters
where multiple members have flagged papers. Also computes individual author
risk scores based on retraction history and anomalous publication patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import community as community_louvain
import networkx as nx
from sqlalchemy import func, select

from snoopy.db.models import Author, AuthorPaperLink, Paper
from snoopy.db.session import get_session

logger = logging.getLogger(__name__)


@dataclass
class AuthorRisk:
    """Risk profile for a single author."""

    author_id: str
    name: str
    risk_score: float
    total_papers: int
    flagged_papers: int
    retraction_count: int
    flagged_ratio: float
    details: str


@dataclass
class FraudCluster:
    """A cluster of co-authors with elevated fraud indicators."""

    cluster_id: int
    authors: list[str]
    total_papers: int
    total_flagged: int
    total_retractions: int
    cluster_risk: float
    details: str


@dataclass
class NetworkAnalysisResult:
    """Result of author network analysis."""

    high_risk_authors: list[AuthorRisk] = field(default_factory=list)
    fraud_clusters: list[FraudCluster] = field(default_factory=list)
    total_authors: int = 0
    total_communities: int = 0


def compute_author_risk(author_id: str) -> AuthorRisk | None:
    """Compute a risk score for a single author.

    Risk factors:
    - Ratio of flagged papers to total papers (0-40 points)
    - Number of retractions (0-30 points)
    - Anomalous publication rate (0-15 points)
    - Co-authorship with other flagged authors (0-15 points)
    """
    with get_session() as session:
        author = session.get(Author, author_id)
        if not author:
            return None

        # Count papers in various states
        total_papers = author.total_papers or 0
        flagged_papers = author.flagged_papers or 0
        retraction_count = author.retraction_count or 0

        if total_papers == 0:
            return AuthorRisk(
                author_id=author_id,
                name=author.name,
                risk_score=0.0,
                total_papers=0,
                flagged_papers=0,
                retraction_count=0,
                flagged_ratio=0.0,
                details="No papers indexed",
            )

        flagged_ratio = flagged_papers / total_papers if total_papers > 0 else 0.0

        # Component scores
        flagged_score = min(flagged_ratio * 100, 40.0)
        retraction_score = min(retraction_count * 10, 30.0)

        # Check for co-authors with flagged papers
        coauthor_risk = 0.0
        coauthor_paper_ids = session.execute(
            select(AuthorPaperLink.paper_id)
            .where(AuthorPaperLink.author_id == author_id)
        ).scalars().all()

        if coauthor_paper_ids:
            # Find co-authors on these papers
            coauthor_ids = session.execute(
                select(AuthorPaperLink.author_id)
                .where(AuthorPaperLink.paper_id.in_(coauthor_paper_ids))
                .where(AuthorPaperLink.author_id != author_id)
                .distinct()
            ).scalars().all()

            flagged_coauthors = 0
            for ca_id in coauthor_ids:
                ca = session.get(Author, ca_id)
                if ca and ca.flagged_papers and ca.flagged_papers > 0:
                    flagged_coauthors += 1

            if coauthor_ids:
                coauthor_risk = min(
                    (flagged_coauthors / len(coauthor_ids)) * 30, 15.0
                )

        risk_score = flagged_score + retraction_score + coauthor_risk

        details_parts = []
        if flagged_ratio > 0.3:
            details_parts.append(f"{flagged_ratio:.0%} of papers flagged")
        if retraction_count > 0:
            details_parts.append(f"{retraction_count} retraction(s)")
        if coauthor_risk > 5:
            details_parts.append("multiple flagged co-authors")

        return AuthorRisk(
            author_id=author_id,
            name=author.name,
            risk_score=min(risk_score, 100.0),
            total_papers=total_papers,
            flagged_papers=flagged_papers,
            retraction_count=retraction_count,
            flagged_ratio=flagged_ratio,
            details="; ".join(details_parts) if details_parts else "No risk factors",
        )


def detect_fraud_clusters(min_cluster_size: int = 3) -> list[FraudCluster]:
    """Detect clusters of co-authors with elevated fraud indicators.

    Builds a co-authorship graph and applies Louvain community detection to
    identify tightly connected groups. Clusters are scored based on the
    aggregate fraud indicators of their members.

    Args:
        min_cluster_size: Minimum number of authors in a cluster to report.

    Returns:
        List of FraudCluster objects, sorted by cluster_risk descending.
    """
    G = nx.Graph()

    with get_session() as session:
        # Build co-authorship graph from paper links
        # Group authors by paper
        paper_authors: dict[str, list[str]] = {}
        links = session.execute(
            select(AuthorPaperLink.paper_id, AuthorPaperLink.author_id)
        ).all()

        for paper_id, author_id in links:
            paper_authors.setdefault(paper_id, []).append(author_id)

        # Add edges between co-authors
        for paper_id, authors in paper_authors.items():
            for i, a in enumerate(authors):
                if not G.has_node(a):
                    author = session.get(Author, a)
                    G.add_node(a, name=author.name if author else a)
                for b in authors[i + 1:]:
                    if not G.has_node(b):
                        author = session.get(Author, b)
                        G.add_node(b, name=author.name if author else b)
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)

    if len(G.nodes) < min_cluster_size:
        return []

    # Apply Louvain community detection
    try:
        partition = community_louvain.best_partition(G)
    except Exception as e:
        logger.warning("Community detection failed: %s", e)
        return []

    # Group authors by community
    communities: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Score each community
    clusters = []
    with get_session() as session:
        for comm_id, members in communities.items():
            if len(members) < min_cluster_size:
                continue

            total_papers = 0
            total_flagged = 0
            total_retractions = 0
            author_names = []

            for author_id in members:
                author = session.get(Author, author_id)
                if author:
                    author_names.append(author.name)
                    total_papers += author.total_papers or 0
                    total_flagged += author.flagged_papers or 0
                    total_retractions += author.retraction_count or 0

            if total_papers == 0:
                continue

            flagged_ratio = total_flagged / total_papers
            cluster_risk = min(
                flagged_ratio * 50 + total_retractions * 10, 100.0
            )

            if cluster_risk > 10:  # Only report non-trivial clusters
                clusters.append(FraudCluster(
                    cluster_id=comm_id,
                    authors=author_names,
                    total_papers=total_papers,
                    total_flagged=total_flagged,
                    total_retractions=total_retractions,
                    cluster_risk=cluster_risk,
                    details=(
                        f"Cluster of {len(members)} authors: "
                        f"{total_flagged}/{total_papers} papers flagged, "
                        f"{total_retractions} retraction(s)"
                    ),
                ))

    clusters.sort(key=lambda c: c.cluster_risk, reverse=True)
    return clusters


def run_network_analysis() -> NetworkAnalysisResult:
    """Run full author network analysis.

    Computes risk scores for all authors and detects fraud clusters.

    Returns:
        NetworkAnalysisResult with high-risk authors and fraud clusters.
    """
    high_risk_authors = []

    with get_session() as session:
        authors = session.execute(select(Author)).scalars().all()
        total_authors = len(authors)

    for author in authors:
        risk = compute_author_risk(author.id)
        if risk and risk.risk_score > 20:
            high_risk_authors.append(risk)

    high_risk_authors.sort(key=lambda a: a.risk_score, reverse=True)

    fraud_clusters = detect_fraud_clusters()

    return NetworkAnalysisResult(
        high_risk_authors=high_risk_authors,
        fraud_clusters=fraud_clusters,
        total_authors=total_authors,
        total_communities=len(fraud_clusters),
    )
