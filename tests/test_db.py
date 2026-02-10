"""Tests for database models and session management."""

from snoopy.db.models import Finding, Figure, Paper, Report
from snoopy.db.session import get_session, init_db


class TestDatabaseInit:
    def test_init_creates_tables(self, test_config):
        init_db(test_config.storage.database_url)
        with get_session() as session:
            # Should be able to query without errors
            papers = session.query(Paper).all()
            assert papers == []

    def test_paper_crud(self, db_session):
        paper = Paper(
            id="test-uuid",
            title="Test Paper",
            doi="10.1234/test",
            source="manual",
        )
        db_session.add(paper)
        db_session.flush()

        found = db_session.get(Paper, "test-uuid")
        assert found is not None
        assert found.title == "Test Paper"
        assert found.doi == "10.1234/test"
        assert found.status == "pending"

    def test_figure_relationship(self, db_session):
        paper = Paper(id="paper-1", title="Test Paper")
        figure = Figure(
            id="fig-1",
            paper_id="paper-1",
            page_number=1,
            figure_label="Figure 1",
        )
        db_session.add(paper)
        db_session.add(figure)
        db_session.flush()

        paper = db_session.get(Paper, "paper-1")
        assert len(paper.figures) == 1
        assert paper.figures[0].figure_label == "Figure 1"

    def test_finding_creation(self, db_session):
        paper = Paper(id="paper-2", title="Test Paper 2")
        finding = Finding(
            id="finding-1",
            paper_id="paper-2",
            analysis_type="ela",
            severity="medium",
            confidence=0.75,
            title="ELA anomaly",
        )
        db_session.add(paper)
        db_session.add(finding)
        db_session.flush()

        found = db_session.get(Finding, "finding-1")
        assert found.severity == "medium"
        assert found.confidence == 0.75

    def test_report_creation(self, db_session):
        paper = Paper(id="paper-3", title="Test Paper 3")
        report = Report(
            id="report-1",
            paper_id="paper-3",
            overall_risk="low",
            overall_confidence=0.3,
            num_findings=2,
        )
        db_session.add(paper)
        db_session.add(report)
        db_session.flush()

        found = db_session.get(Report, "report-1")
        assert found.overall_risk == "low"
