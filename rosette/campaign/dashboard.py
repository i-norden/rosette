"""Campaign dashboard generation: collects data and renders HTML."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import func, select

from rosette.db.models import (
    Campaign,
    CampaignPaper,
    Figure,
    Finding,
    ImageHashMatch,
    Paper,
    Report,
)
from rosette.db.session import get_session

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "reporting" / "templates"


def _get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=["html.j2"], default_for_string=False),
    )


def generate_campaign_dashboard(campaign_id: str) -> str:
    """Generate an HTML dashboard for a campaign.

    Returns the rendered HTML string.
    """
    data = _collect_dashboard_data(campaign_id)
    env = _get_jinja_env()
    template = env.get_template("campaign_dashboard.html.j2")
    return template.render(**data)


def _collect_dashboard_data(campaign_id: str) -> dict:
    """Collect all data needed for the dashboard template."""
    with get_session() as session:
        campaign = session.get(Campaign, campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Campaign overview
        config_data = {}
        if campaign.config_json:
            try:
                config_data = json.loads(str(campaign.config_json))
            except (json.JSONDecodeError, TypeError):
                pass

        seed_dois = []
        if campaign.seed_dois:
            try:
                seed_dois = json.loads(str(campaign.seed_dois))
            except (json.JSONDecodeError, TypeError):
                pass

        overview = {
            "id": campaign.id,
            "name": campaign.name,
            "mode": campaign.mode,
            "status": campaign.status,
            "created_at": str(campaign.created_at or ""),
            "updated_at": str(campaign.updated_at or ""),
            "max_depth": campaign.max_depth,
            "max_papers": campaign.max_papers,
            "llm_budget": campaign.llm_budget,
            "config": config_data,
            "seed_dois": seed_dois,
        }

        # Funnel numbers
        funnel = {
            "papers_discovered": campaign.papers_discovered or 0,
            "papers_triaged": campaign.papers_triaged or 0,
            "papers_flagged": campaign.papers_flagged or 0,
            "papers_llm_analyzed": campaign.papers_llm_analyzed or 0,
        }

        # Risk distribution
        risk_rows = session.execute(
            select(CampaignPaper.final_risk, func.count(CampaignPaper.id))
            .where(CampaignPaper.campaign_id == campaign_id)
            .where(CampaignPaper.final_risk.isnot(None))
            .group_by(CampaignPaper.final_risk)
        ).all()
        risk_distribution = {str(risk): count for risk, count in risk_rows}

        # Triage status breakdown
        triage_rows = session.execute(
            select(CampaignPaper.triage_status, func.count(CampaignPaper.id))
            .where(CampaignPaper.campaign_id == campaign_id)
            .group_by(CampaignPaper.triage_status)
        ).all()
        triage_breakdown = {str(status): count for status, count in triage_rows}

        # Top findings (papers with highest risk)
        top_papers_query = (
            session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.final_risk.isnot(None))
                .order_by(CampaignPaper.auto_risk_score.desc())
                .limit(20)
            )
            .scalars()
            .all()
        )

        # Batch-fetch papers and reports for top findings
        top_paper_ids = [str(cp.paper_id) for cp in top_papers_query]
        papers_by_id: dict[str, Paper] = {}
        reports_by_paper: dict[str, Report] = {}
        if top_paper_ids:
            fetched_papers = (
                session.execute(select(Paper).where(Paper.id.in_(top_paper_ids))).scalars().all()
            )
            papers_by_id = {str(p.id): p for p in fetched_papers}

            # Get latest report per paper
            all_reports = (
                session.execute(
                    select(Report)
                    .where(Report.paper_id.in_(top_paper_ids))
                    .order_by(Report.created_at.desc())
                )
                .scalars()
                .all()
            )
            for rpt in all_reports:
                pid = str(rpt.paper_id)
                if pid not in reports_by_paper:
                    reports_by_paper[pid] = rpt

        top_findings = []
        for cp in top_papers_query:
            paper = papers_by_id.get(str(cp.paper_id))
            if not paper:
                continue

            report = reports_by_paper.get(str(cp.paper_id))

            top_findings.append(
                {
                    "paper_id": cp.paper_id,
                    "doi": paper.doi,
                    "title": str(paper.title)[:100] if paper.title else "Unknown",
                    "journal": paper.journal,
                    "auto_risk_score": cp.auto_risk_score,
                    "final_risk": cp.final_risk,
                    "llm_promoted": cp.llm_promoted,
                    "source": cp.source,
                    "depth": cp.depth,
                    "auto_findings_count": cp.auto_findings_count,
                    "summary": str(report.summary)[:200] if report and report.summary else "",
                    "num_findings": report.num_findings if report else 0,
                }
            )

        # Cross-paper image hash matches
        hash_matches_query = (
            session.execute(
                select(ImageHashMatch)
                .where(ImageHashMatch.campaign_id == campaign_id)
                .order_by(ImageHashMatch.hash_distance)
                .limit(50)
            )
            .scalars()
            .all()
        )

        # Batch-fetch all papers and figures referenced by hash matches
        match_paper_ids: set[str] = set()
        match_figure_ids: set[str] = set()
        for m in hash_matches_query:
            match_paper_ids.add(str(m.paper_id_a))
            match_paper_ids.add(str(m.paper_id_b))
            match_figure_ids.add(str(m.figure_id_a))
            match_figure_ids.add(str(m.figure_id_b))

        match_papers: dict[str, Paper] = {}
        match_figures: dict[str, Figure] = {}
        if match_paper_ids:
            fetched = (
                session.execute(select(Paper).where(Paper.id.in_(match_paper_ids))).scalars().all()
            )
            match_papers = {str(p.id): p for p in fetched}
        if match_figure_ids:
            fetched = (
                session.execute(select(Figure).where(Figure.id.in_(match_figure_ids)))
                .scalars()
                .all()
            )
            match_figures = {str(f.id): f for f in fetched}

        hash_matches = []
        for m in hash_matches_query:
            paper_a = match_papers.get(str(m.paper_id_a))
            paper_b = match_papers.get(str(m.paper_id_b))
            fig_a = match_figures.get(str(m.figure_id_a))
            fig_b = match_figures.get(str(m.figure_id_b))

            hash_matches.append(
                {
                    "paper_a_doi": paper_a.doi if paper_a else None,
                    "paper_a_title": str(paper_a.title)[:60] if paper_a else "Unknown",
                    "paper_b_doi": paper_b.doi if paper_b else None,
                    "paper_b_title": str(paper_b.title)[:60] if paper_b else "Unknown",
                    "figure_a_label": fig_a.figure_label if fig_a else "",
                    "figure_b_label": fig_b.figure_label if fig_b else "",
                    "figure_a_path": fig_a.image_path if fig_a else None,
                    "figure_b_path": fig_b.image_path if fig_b else None,
                    "hash_distance": m.hash_distance,
                    "hash_type": m.hash_type,
                    "verified": m.verified,
                }
            )

        # Method breakdown
        all_paper_ids = [
            str(row[0])
            for row in session.execute(
                select(CampaignPaper.paper_id).where(CampaignPaper.campaign_id == campaign_id)
            ).all()
        ]

        method_counts: Counter[str] = Counter()
        if all_paper_ids:
            findings = (
                session.execute(
                    select(Finding.analysis_type).where(Finding.paper_id.in_(all_paper_ids))
                )
                .scalars()
                .all()
            )
            for at in findings:
                method_counts[str(at)] += 1

        method_breakdown = [
            {"name": name, "count": count}
            for name, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Expansion tree (for network_expansion mode)
        expansion_tree = []
        if str(campaign.mode) == "network_expansion":
            depth_counts = session.execute(
                select(CampaignPaper.depth, func.count(CampaignPaper.id))
                .where(CampaignPaper.campaign_id == campaign_id)
                .group_by(CampaignPaper.depth)
                .order_by(CampaignPaper.depth)
            ).all()
            for depth_val, count in depth_counts:
                flagged_at_depth = (
                    session.execute(
                        select(func.count(CampaignPaper.id))
                        .where(CampaignPaper.campaign_id == campaign_id)
                        .where(CampaignPaper.depth == depth_val)
                        .where(CampaignPaper.final_risk.in_(["critical", "high", "medium"]))
                    ).scalar()
                    or 0
                )

                expansion_tree.append(
                    {
                        "depth": depth_val,
                        "papers": count,
                        "flagged": flagged_at_depth,
                    }
                )

    import rosette

    return {
        "overview": overview,
        "funnel": funnel,
        "risk_distribution": risk_distribution,
        "triage_breakdown": triage_breakdown,
        "top_findings": top_findings,
        "hash_matches": hash_matches,
        "method_breakdown": method_breakdown,
        "expansion_tree": expansion_tree,
        "version": rosette.__version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
