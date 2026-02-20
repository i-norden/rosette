"""Tests for database models, session management, and migrations."""

import pytest
from sqlalchemy import select

from rosette.db.migrations import check_schema, create_all_tables, get_paper_counts, reset_database
from rosette.db.models import Base, Finding, Figure, Paper, Report
from rosette.db.session import get_engine, get_session, init_async_db, init_db, reset_db


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


class TestResetDb:
    def test_reset_db_clears_state(self, test_config):
        """reset_db should clear the global engine and session factory."""
        init_db(test_config.storage.database_url)
        reset_db()
        # After reset, get_session should raise
        with pytest.raises(RuntimeError, match="not initialized"):
            with get_session():
                pass

    def test_reset_db_allows_reinit(self, test_config):
        """After reset_db, init_db should work again."""
        init_db(test_config.storage.database_url)
        reset_db()
        init_db(test_config.storage.database_url)
        with get_session() as session:
            papers = session.query(Paper).all()
            assert papers == []

    def test_get_engine_before_init_raises(self):
        """get_engine before init_db raises RuntimeError."""
        reset_db()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_engine()

    def test_reset_db_with_async_engine(self, test_config):
        """reset_db also clears async engine/session."""
        init_db(test_config.storage.database_url)
        init_async_db(test_config.storage.database_url)
        reset_db()
        with pytest.raises(RuntimeError, match="not initialized"):
            with get_session():
                pass


class TestMigrations:
    def test_create_all_tables(self, test_config):
        init_db(test_config.storage.database_url)
        schema = check_schema()
        expected_tables = set(Base.metadata.tables.keys())
        for table in expected_tables:
            assert schema.get(table) is True, f"Table {table} was not created"

    def test_create_all_tables_idempotent(self, test_config):
        init_db(test_config.storage.database_url)
        create_all_tables()
        schema = check_schema()
        assert all(schema.values())

    def test_check_schema_returns_booleans(self, test_config):
        init_db(test_config.storage.database_url)
        schema = check_schema()
        assert isinstance(schema, dict)
        for table, exists in schema.items():
            assert isinstance(table, str)
            assert isinstance(exists, bool)

    def test_reset_database(self, test_config):
        init_db(test_config.storage.database_url)

        with get_session() as session:
            paper = Paper(id="reset-test-1", title="Test", source="test", status="pending")
            session.add(paper)

        reset_database()
        schema = check_schema()
        assert all(schema.values())

        with get_session() as session:
            result = session.execute(select(Paper)).scalars().all()
            assert len(result) == 0

    def test_get_paper_counts_empty(self, test_config):
        init_db(test_config.storage.database_url)
        counts = get_paper_counts()
        assert counts == {}

    def test_get_paper_counts_by_status(self, test_config):
        init_db(test_config.storage.database_url)

        with get_session() as session:
            for i in range(3):
                session.add(Paper(id=f"pending-{i}", title="P", source="test", status="pending"))
            for i in range(2):
                session.add(Paper(id=f"complete-{i}", title="C", source="test", status="complete"))

        counts = get_paper_counts()
        assert counts.get("pending") == 3
        assert counts.get("complete") == 2
