"""Tests for API route handlers."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from snoopy.api.app import create_app
from snoopy.config import SnoopyConfig
from snoopy.db.models import Author, Base, Finding, Paper, Report
from snoopy.db.session import get_async_session, get_session, init_async_db, init_db


@pytest.fixture
def app(tmp_path):
    """Create a test FastAPI app with a temporary database."""
    db_path = tmp_path / "test.db"
    config = SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    init_db(config.storage.database_url)
    init_async_db(config.storage.database_url)

    application = create_app(config)
    return application


@pytest.fixture
def client(app):
    """Provide a test client for the app."""
    return TestClient(app)


@pytest.fixture
def seeded_paper(tmp_path):
    """Seed a paper into the database and return its id."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    init_db(db_url)
    paper_id = str(uuid.uuid4())
    with get_session() as session:
        paper = Paper(
            id=paper_id,
            doi="10.1234/test.paper",
            title="Test Paper",
            source="test",
            status="complete",
        )
        session.add(paper)
    return paper_id


class TestHealthCheck:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSubmitPaper:
    def test_submit_with_doi(self, client):
        response = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/test.paper"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "paper_id" in data
        assert data["status"] == "pending"

    def test_submit_without_doi_or_pdf(self, client):
        response = client.post(
            "/api/v1/papers",
            json={},
        )
        assert response.status_code == 422

    def test_submit_with_invalid_doi(self, client):
        response = client.post(
            "/api/v1/papers",
            json={"doi": "not-a-doi"},
        )
        assert response.status_code == 422
        assert "Invalid DOI" in response.json()["detail"]

    def test_submit_duplicate_doi_returns_existing(self, client):
        # First submission
        resp1 = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/duplicate.test"},
        )
        assert resp1.status_code == 202
        paper_id_1 = resp1.json()["paper_id"]

        # Second submission with same DOI
        resp2 = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/duplicate.test"},
        )
        assert resp2.status_code == 202
        assert resp2.json()["paper_id"] == paper_id_1


class TestGetPaperStatus:
    def test_get_existing_paper(self, client):
        # Create paper
        resp = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/status.test"},
        )
        paper_id = resp.json()["paper_id"]

        # Check status
        status_resp = client.get(f"/api/v1/papers/{paper_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["paper_id"] == paper_id

    def test_get_nonexistent_paper(self, client):
        response = client.get(f"/api/v1/papers/{uuid.uuid4()}")
        assert response.status_code == 404


class TestGetPaperReport:
    def test_get_report_no_paper(self, client):
        response = client.get(f"/api/v1/papers/{uuid.uuid4()}/report")
        assert response.status_code == 404

    def test_get_report_no_report_yet(self, client):
        # Create paper
        resp = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/report.test"},
        )
        paper_id = resp.json()["paper_id"]

        # Try to get report before analysis
        response = client.get(f"/api/v1/papers/{paper_id}/report")
        assert response.status_code == 404
        assert "not yet available" in response.json()["detail"]


class TestSubmitBatch:
    def test_submit_batch(self, client):
        response = client.post(
            "/api/v1/batch",
            json={"dois": ["10.1234/batch.1", "10.1234/batch.2"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert len(data["papers"]) == 2

    def test_submit_empty_batch(self, client):
        response = client.post(
            "/api/v1/batch",
            json={"dois": []},
        )
        assert response.status_code == 422

    def test_submit_batch_with_invalid_doi(self, client):
        response = client.post(
            "/api/v1/batch",
            json={"dois": ["not-a-doi"]},
        )
        assert response.status_code == 422
        assert "Invalid DOI" in response.json()["detail"]

    def test_batch_size_limit(self, client):
        dois = [f"10.1234/batch.{i}" for i in range(101)]
        response = client.post(
            "/api/v1/batch",
            json={"dois": dois},
        )
        assert response.status_code == 422


class TestGetAuthorRisk:
    def test_get_nonexistent_author(self, client):
        response = client.get(f"/api/v1/authors/{uuid.uuid4()}/risk")
        assert response.status_code == 404


class TestApiKeyAuth:
    def test_no_auth_when_no_keys_configured(self, client):
        """When no api_keys are configured, all requests should pass."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_auth_rejects_missing_key(self, app):
        """When api_keys are set on config, missing key is rejected."""
        app.state.config.api_keys = ["test-key-123"]
        test_client = TestClient(app)
        response = test_client.get("/api/v1/papers/some-id")
        assert response.status_code == 401

    def test_auth_accepts_valid_key(self, app):
        """When api_keys are set on config, valid key is accepted."""
        app.state.config.api_keys = ["test-key-123"]
        test_client = TestClient(app)
        # Create a paper first to avoid 404
        resp = test_client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/auth.test"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 202

    def test_auth_rejects_invalid_key(self, app):
        """When api_keys are set, wrong key is rejected."""
        app.state.config.api_keys = ["test-key-123"]
        test_client = TestClient(app)
        response = test_client.get(
            "/api/v1/papers/some-id",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
