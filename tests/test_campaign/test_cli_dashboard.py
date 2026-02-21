"""Tests for campaign CLI commands and dashboard generation.

Tests CLI command invocation with Click's test runner and
dashboard data collection with mocked DB data.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from rosette.cli import main
from rosette.config import RosetteConfig
from rosette.db.models import (
    Campaign,
    CampaignPaper,
    Figure,
    Finding,
    ImageHashMatch,
    Paper,
    Report,
)
from rosette.db.session import get_session, init_db


@pytest.fixture
def cli_config(tmp_path) -> RosetteConfig:
    db_path = tmp_path / "cli_test.db"
    return RosetteConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
    )


@pytest.fixture
def cli_db(cli_config) -> RosetteConfig:
    """Initialize DB and return config."""
    init_db(cli_config.storage.database_url)
    return cli_config


@pytest.fixture
def runner():
    return CliRunner()


class TestCampaignCreate:
    def test_create_network_expansion(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(
            main,
            [
                "campaign",
                "create",
                "--mode",
                "network_expansion",
                "--name",
                "Test NE Campaign",
                "--seed-doi",
                "10.1234/test1",
                "--seed-doi",
                "10.1234/test2",
                "--max-depth",
                "3",
                "--max-papers",
                "500",
                "--llm-budget",
                "50",
            ],
        )

        assert result.exit_code == 0
        assert "Campaign created:" in result.output
        assert "Test NE Campaign" in result.output
        assert "Seeds: 2 DOIs" in result.output

    def test_create_domain_scan(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(
            main,
            [
                "campaign",
                "create",
                "--mode",
                "domain_scan",
                "--name",
                "Domain Scan Test",
                "--field",
                "oncology",
            ],
        )

        assert result.exit_code == 0
        assert "Campaign created:" in result.output

    def test_create_network_expansion_requires_seed(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(
            main,
            [
                "campaign",
                "create",
                "--mode",
                "network_expansion",
                "--name",
                "No Seeds",
            ],
        )

        assert result.exit_code != 0
        assert "--seed-doi is required" in result.output

    def test_create_domain_scan_requires_field(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(
            main,
            [
                "campaign",
                "create",
                "--mode",
                "domain_scan",
                "--name",
                "No Field",
            ],
        )

        assert result.exit_code != 0
        assert "--field is required" in result.output


class TestCampaignList:
    def test_list_empty(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(main, ["campaign", "list"])
        assert result.exit_code == 0
        assert "No campaigns found" in result.output

    def test_list_with_campaigns(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        with get_session() as session:
            session.add(
                Campaign(
                    id="cam-001",
                    name="Campaign One",
                    mode="domain_scan",
                    status="completed",
                    papers_discovered=50,
                )
            )

        result = runner.invoke(main, ["campaign", "list"])
        assert result.exit_code == 0
        assert "Campaign One" in result.output
        assert "domain_scan" in result.output


class TestCampaignStatus:
    def test_status_specific_campaign(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        with get_session() as session:
            session.add(
                Campaign(
                    id="cam-status-001",
                    name="Status Campaign",
                    mode="network_expansion",
                    status="auto_analyzing",
                    papers_discovered=20,
                    papers_triaged=15,
                    papers_flagged=5,
                    papers_llm_analyzed=3,
                    llm_budget=50,
                )
            )

        result = runner.invoke(main, ["campaign", "status", "cam-status-001"])
        assert result.exit_code == 0
        assert "Status Campaign" in result.output
        assert "auto_analyzing" in result.output
        assert "20" in result.output

    def test_status_not_found(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(main, ["campaign", "status", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestCampaignPause:
    def test_pause_campaign(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        with get_session() as session:
            session.add(
                Campaign(
                    id="cam-pause-001",
                    name="Pause Test",
                    mode="domain_scan",
                    status="auto_analyzing",
                )
            )

        result = runner.invoke(main, ["campaign", "pause", "cam-pause-001"])
        assert result.exit_code == 0
        assert "paused" in result.output

        with get_session() as session:
            campaign = session.get(Campaign, "cam-pause-001")
            assert campaign.status == "paused"

    def test_pause_not_found(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        result = runner.invoke(main, ["campaign", "pause", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestDashboardGeneration:
    def test_collect_dashboard_data(self, cli_db):
        """Test the dashboard data collection with real DB data."""
        from rosette.campaign.dashboard import _collect_dashboard_data

        campaign_id = "cam-dashboard-001"
        with get_session() as session:
            session.add(
                Campaign(
                    id=campaign_id,
                    name="Dashboard Test",
                    mode="network_expansion",
                    status="completed",
                    config_json=json.dumps({"field": "biomedical"}),
                    seed_dois=json.dumps(["10.1234/seed1"]),
                    max_depth=2,
                    max_papers=100,
                    llm_budget=10,
                    papers_discovered=5,
                    papers_triaged=4,
                    papers_flagged=2,
                    papers_llm_analyzed=1,
                )
            )

            # Paper A: high risk
            session.add(
                Paper(
                    id="dash-paper-a",
                    title="Suspicious Paper A",
                    doi="10.1234/dash.a",
                    source="test",
                    status="complete",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="dash-paper-a",
                    source="seed",
                    depth=0,
                    triage_status="complete",
                    auto_risk_score=75.0,
                    llm_promoted=True,
                    final_risk="high",
                    auto_findings_count=3,
                )
            )
            session.add(
                Finding(
                    paper_id="dash-paper-a",
                    analysis_type="clone_detection",
                    severity="high",
                    confidence=0.9,
                    title="Clone detected",
                )
            )

            # Paper B: clean
            session.add(
                Paper(
                    id="dash-paper-b",
                    title="Clean Paper B",
                    doi="10.1234/dash.b",
                    source="test",
                    status="complete",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="dash-paper-b",
                    source="network_expansion",
                    depth=1,
                    triage_status="dismissed",
                    auto_risk_score=5.0,
                    final_risk="clean",
                )
            )

        data = _collect_dashboard_data(campaign_id)

        assert data["overview"]["name"] == "Dashboard Test"
        assert data["overview"]["mode"] == "network_expansion"
        assert data["funnel"]["papers_discovered"] == 5
        assert "high" in data["risk_distribution"]
        assert "clean" in data["risk_distribution"]
        assert len(data["top_findings"]) >= 1
        assert data["top_findings"][0]["final_risk"] == "high"
        assert len(data["method_breakdown"]) >= 1
        assert len(data["expansion_tree"]) >= 1

    def test_collect_dashboard_nonexistent_campaign(self, cli_db):
        from rosette.campaign.dashboard import _collect_dashboard_data

        with pytest.raises(ValueError, match="not found"):
            _collect_dashboard_data("nonexistent-id")

    def test_generate_dashboard_html(self, cli_db):
        """Test full HTML rendering."""
        from rosette.campaign.dashboard import generate_campaign_dashboard

        campaign_id = "cam-html-001"
        with get_session() as session:
            session.add(
                Campaign(
                    id=campaign_id,
                    name="HTML Test",
                    mode="domain_scan",
                    status="completed",
                    papers_discovered=0,
                )
            )

        html = generate_campaign_dashboard(campaign_id)
        assert "<html" in html
        assert "HTML Test" in html

    def test_dashboard_with_hash_matches(self, cli_db):
        """Test dashboard includes hash match data."""
        from rosette.campaign.dashboard import _collect_dashboard_data

        campaign_id = "cam-hash-dash-001"
        with get_session() as session:
            session.add(
                Campaign(
                    id=campaign_id,
                    name="Hash Match Dashboard",
                    mode="domain_scan",
                    status="completed",
                    papers_discovered=2,
                )
            )

            session.add(
                Paper(
                    id="hd-paper-a",
                    title="Hash Paper A",
                    doi="10.1234/hd.a",
                    source="test",
                    status="complete",
                )
            )
            session.add(
                Paper(
                    id="hd-paper-b",
                    title="Hash Paper B",
                    doi="10.1234/hd.b",
                    source="test",
                    status="complete",
                )
            )
            fig_a = Figure(id="hd-fig-a", paper_id="hd-paper-a", phash="aaaa")
            fig_b = Figure(id="hd-fig-b", paper_id="hd-paper-b", phash="aaab")
            session.add(fig_a)
            session.add(fig_b)

            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="hd-paper-a",
                    source="domain_scan",
                    depth=0,
                    triage_status="complete",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="hd-paper-b",
                    source="domain_scan",
                    depth=0,
                    triage_status="complete",
                )
            )

            session.add(
                ImageHashMatch(
                    campaign_id=campaign_id,
                    figure_id_a="hd-fig-a",
                    figure_id_b="hd-fig-b",
                    paper_id_a="hd-paper-a",
                    paper_id_b="hd-paper-b",
                    hash_type="phash",
                    hash_distance=2,
                )
            )

        data = _collect_dashboard_data(campaign_id)
        assert len(data["hash_matches"]) == 1
        assert data["hash_matches"][0]["hash_distance"] == 2


class TestCampaignExport:
    def test_export_filters_by_risk(self, cli_db, runner, monkeypatch):
        monkeypatch.setattr("rosette.cli.load_config", lambda *a, **kw: cli_db)

        campaign_id = "cam-export-001"
        with get_session() as session:
            session.add(
                Campaign(
                    id=campaign_id,
                    name="Export Test",
                    mode="domain_scan",
                    status="completed",
                )
            )

            session.add(
                Paper(
                    id="exp-paper-h",
                    title="High Risk Paper",
                    source="test",
                    status="complete",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="exp-paper-h",
                    source="domain_scan",
                    depth=0,
                    triage_status="complete",
                    final_risk="high",
                )
            )
            session.add(
                Report(
                    paper_id="exp-paper-h",
                    overall_risk="high",
                    overall_confidence=0.9,
                    report_html="<html>High risk report</html>",
                )
            )

            session.add(
                Paper(
                    id="exp-paper-l",
                    title="Low Risk Paper",
                    source="test",
                    status="complete",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="exp-paper-l",
                    source="domain_scan",
                    depth=0,
                    triage_status="complete",
                    final_risk="low",
                )
            )
            session.add(
                Report(
                    paper_id="exp-paper-l",
                    overall_risk="low",
                    overall_confidence=0.3,
                    report_html="<html>Low risk report</html>",
                )
            )

        import tempfile

        with tempfile.TemporaryDirectory() as export_dir:
            result = runner.invoke(
                main,
                [
                    "campaign",
                    "export",
                    campaign_id,
                    "--min-risk",
                    "high",
                    "--output-dir",
                    export_dir,
                ],
            )

            assert result.exit_code == 0
            assert "Exported 1 evidence packages" in result.output
