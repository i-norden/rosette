"""Author network analysis for detecting fraud rings and prolific fraudsters.

Builds co-authorship graphs and applies community detection to find clusters
where multiple members have flagged papers. Also computes individual author
risk scores based on retraction history and anomalous publication patterns.
Includes temporal publication pattern analysis for paper mill detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import community as community_louvain
import networkx as nx
from sqlalchemy import select

from rosette.db.models import Author, AuthorPaperLink
from rosette.db.session import get_session

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
                name=author.name,  # type: ignore[arg-type]
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
        coauthor_paper_ids = (
            session.execute(
                select(AuthorPaperLink.paper_id).where(AuthorPaperLink.author_id == author_id)
            )
            .scalars()
            .all()
        )

        if coauthor_paper_ids:
            # Find co-authors on these papers
            coauthor_ids = (
                session.execute(
                    select(AuthorPaperLink.author_id)
                    .where(AuthorPaperLink.paper_id.in_(coauthor_paper_ids))
                    .where(AuthorPaperLink.author_id != author_id)
                    .distinct()
                )
                .scalars()
                .all()
            )

            coauthors = (
                session.execute(select(Author).where(Author.id.in_(coauthor_ids))).scalars().all()
            )
            flagged_coauthors = sum(
                1 for ca in coauthors if ca.flagged_papers and ca.flagged_papers > 0
            )

            if coauthor_ids:
                coauthor_risk = min((flagged_coauthors / len(coauthor_ids)) * 30, 15.0)

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
            name=author.name,  # type: ignore[arg-type]
            risk_score=min(risk_score, 100.0),  # type: ignore[arg-type]
            total_papers=total_papers,  # type: ignore[arg-type]
            flagged_papers=flagged_papers,  # type: ignore[arg-type]
            retraction_count=retraction_count,  # type: ignore[arg-type]
            flagged_ratio=flagged_ratio,  # type: ignore[arg-type]
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
        links = session.execute(select(AuthorPaperLink.paper_id, AuthorPaperLink.author_id)).all()

        for paper_id, author_id in links:
            paper_authors.setdefault(paper_id, []).append(author_id)

        # Batch-fetch all authors referenced in links
        all_author_ids = {aid for aids in paper_authors.values() for aid in aids}
        authors_by_id: dict[str, Author] = {}
        if all_author_ids:
            fetched = (
                session.execute(select(Author).where(Author.id.in_(all_author_ids))).scalars().all()
            )
            authors_by_id = {str(a.id): a for a in fetched}

        # Add edges between co-authors
        for paper_id, authors in paper_authors.items():
            for i, a in enumerate(authors):
                if not G.has_node(a):
                    author = authors_by_id.get(a)
                    G.add_node(a, name=author.name if author else a)
                for b in authors[i + 1 :]:
                    if not G.has_node(b):
                        author = authors_by_id.get(b)
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
        # Batch-fetch all authors in communities that meet min size
        all_community_member_ids = {
            mid
            for members in communities.values()
            if len(members) >= min_cluster_size
            for mid in members
        }
        community_authors: dict[str, Author] = {}
        if all_community_member_ids:
            fetched = (
                session.execute(select(Author).where(Author.id.in_(all_community_member_ids)))
                .scalars()
                .all()
            )
            community_authors = {str(a.id): a for a in fetched}

        for comm_id, members in communities.items():
            if len(members) < min_cluster_size:
                continue

            total_papers = 0
            total_flagged = 0
            total_retractions = 0
            author_names = []

            for author_id in members:
                author = community_authors.get(author_id)
                if author:
                    author_names.append(str(author.name))
                    total_papers += int(author.total_papers or 0)
                    total_flagged += int(author.flagged_papers or 0)
                    total_retractions += int(author.retraction_count or 0)

            if total_papers == 0:
                continue

            flagged_ratio = total_flagged / total_papers
            cluster_risk = min(flagged_ratio * 50 + total_retractions * 10, 100.0)

            if cluster_risk > 10:  # Only report non-trivial clusters
                clusters.append(
                    FraudCluster(
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
                    )
                )

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
        risk = compute_author_risk(str(author.id))
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


# ---------------------------------------------------------------------------
# 2.11 Temporal publication pattern analysis
# ---------------------------------------------------------------------------


@dataclass
class TemporalPatternResult:
    """Result of temporal publication pattern analysis."""

    suspicious: bool
    total_papers: int
    min_interval_days: float | None
    mean_interval_days: float | None
    publication_rate_per_year: float
    volume_spike_detected: bool
    fast_acceptance_count: int
    details: str


def analyze_temporal_patterns(
    author_id: str,
    min_papers: int = 5,
    spike_threshold: float = 3.0,
    min_interval_days: float = 14.0,
) -> TemporalPatternResult | None:
    """Analyze an author's publication timeline for paper mill signals.

    Detects:
    - Implausibly short submission intervals
    - Sudden publication volume spikes
    - Suspiciously fast acceptance times

    Args:
        author_id: The author to analyze.
        min_papers: Minimum number of papers with dates needed for analysis.
        spike_threshold: Multiple of baseline rate to count as a volume spike.
        min_interval_days: Minimum expected days between publications.

    Returns:
        TemporalPatternResult, or None if the author is not found.
    """
    from rosette.db.models import Paper

    with get_session() as session:
        author = session.get(Author, author_id)
        if not author:
            return None

        # Get all papers for this author with publication dates
        paper_ids = (
            session.execute(
                select(AuthorPaperLink.paper_id).where(AuthorPaperLink.author_id == author_id)
            )
            .scalars()
            .all()
        )

        if not paper_ids:
            return TemporalPatternResult(
                suspicious=False,
                total_papers=0,
                min_interval_days=None,
                mean_interval_days=None,
                publication_rate_per_year=0.0,
                volume_spike_detected=False,
                fast_acceptance_count=0,
                details="No papers found for this author",
            )

        papers = session.execute(select(Paper).where(Paper.id.in_(paper_ids))).scalars().all()

        # Extract publication dates
        pub_dates: list[datetime] = []
        for p in papers:
            date_val = getattr(p, "publication_date", None) or getattr(p, "created_at", None)
            if date_val is not None:
                if isinstance(date_val, str):
                    try:
                        date_val = datetime.fromisoformat(date_val)
                    except (ValueError, TypeError):
                        continue
                if isinstance(date_val, datetime):
                    pub_dates.append(date_val)

        total_papers = len(papers)

        if len(pub_dates) < min_papers:
            return TemporalPatternResult(
                suspicious=False,
                total_papers=total_papers,
                min_interval_days=None,
                mean_interval_days=None,
                publication_rate_per_year=0.0,
                volume_spike_detected=False,
                fast_acceptance_count=0,
                details=f"Only {len(pub_dates)} papers with dates (need {min_papers})",
            )

        pub_dates.sort()

        # Compute inter-publication intervals
        intervals: list[float] = []
        for i in range(1, len(pub_dates)):
            delta = (pub_dates[i] - pub_dates[i - 1]).total_seconds() / 86400.0
            intervals.append(delta)

        min_interval = min(intervals) if intervals else None
        mean_interval = sum(intervals) / len(intervals) if intervals else None

        # Publication rate
        date_range = (pub_dates[-1] - pub_dates[0]).days
        years = max(date_range / 365.25, 0.1)
        rate_per_year = len(pub_dates) / years

        # Volume spike detection: check if any 6-month window has >spike_threshold
        # times the baseline rate
        volume_spike = False
        baseline_rate = rate_per_year / 2.0  # per 6 months
        if baseline_rate > 0:
            window = timedelta(days=182)
            for i, date in enumerate(pub_dates):
                window_end = date + window
                count_in_window = sum(1 for d in pub_dates[i:] if d <= window_end)
                if count_in_window > baseline_rate * spike_threshold:
                    volume_spike = True
                    break

        # Fast acceptance count (papers with very short intervals)
        fast_count = sum(1 for iv in intervals if iv < min_interval_days)

        # Determine suspicion
        details_parts: list[str] = []
        suspicious = False

        if min_interval is not None and min_interval < min_interval_days:
            details_parts.append(
                f"Minimum interval between publications: {min_interval:.1f} days "
                f"(threshold: {min_interval_days} days)"
            )
            if fast_count >= 3:
                suspicious = True

        if volume_spike:
            details_parts.append(
                f"Publication volume spike detected (>{spike_threshold}x baseline)"
            )
            suspicious = True

        if rate_per_year > 20:
            details_parts.append(f"Unusually high publication rate: {rate_per_year:.1f}/year")
            suspicious = True

        return TemporalPatternResult(
            suspicious=suspicious,
            total_papers=total_papers,
            min_interval_days=min_interval,
            mean_interval_days=mean_interval,
            publication_rate_per_year=rate_per_year,
            volume_spike_detected=volume_spike,
            fast_acceptance_count=fast_count,
            details="; ".join(details_parts) if details_parts else "No anomalous patterns",
        )
