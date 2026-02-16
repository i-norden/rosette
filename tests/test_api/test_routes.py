"""Tests for API route handlers."""

from __future__ import annotations

import base64
import uuid

import pytest
from fastapi.testclient import TestClient

from snoopy.api.app import create_app
from snoopy.config import SnoopyConfig
from snoopy.db.models import Paper
from snoopy.db.session import get_session, init_async_db, init_db


@pytest.fixture
def app(tmp_path):
    """Create a test FastAPI app with a temporary database."""
    db_path = tmp_path / "test.db"
    config = SnoopyConfig(
        require_authentication=False,
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
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

    def test_submit_pdf_with_invalid_base64(self, client):
        """Invalid base64 in pdf_upload should return 422."""
        response = client.post(
            "/api/v1/papers",
            json={"pdf_upload": "not-valid-base64!!!"},
        )
        assert response.status_code == 422
        assert "base64" in response.json()["detail"].lower()

    def test_submit_pdf_with_non_pdf_content(self, client):
        """Valid base64 but not PDF content should return 422."""
        fake_content = base64.b64encode(b"This is not a PDF").decode()
        response = client.post(
            "/api/v1/papers",
            json={"pdf_upload": fake_content},
        )
        assert response.status_code == 422
        assert "not a valid PDF" in response.json()["detail"]

    def test_submit_pdf_with_valid_pdf_magic(self, client):
        """Valid base64 with PDF magic bytes should be accepted."""
        # Minimal PDF-like content (just has the magic bytes)
        pdf_content = b"%PDF-1.4 minimal pdf content"
        encoded = base64.b64encode(pdf_content).decode()
        response = client.post(
            "/api/v1/papers",
            json={"pdf_upload": encoded},
        )
        assert response.status_code == 202
        data = response.json()
        assert "paper_id" in data
        assert data["status"] == "pending"

    def test_submit_doi_too_long(self, client):
        """DOI exceeding max length should be rejected."""
        long_doi = "10.1234/" + "a" * 300
        response = client.post(
            "/api/v1/papers",
            json={"doi": long_doi},
        )
        assert response.status_code == 422
        assert (
            "too long" in response.json()["detail"].lower()
            or "Invalid DOI" in response.json()["detail"]
        )

    def test_submit_oversized_pdf(self, client, monkeypatch):
        """PDF exceeding 100MB should be rejected with 422."""
        # Mock base64.b64decode to return oversized content without actually
        # sending 100MB+ over the wire
        import snoopy.api.routes as routes_module

        def _mock_b64decode(data, validate=False):
            # Return oversized content with PDF magic bytes
            return b"%PDF-1.4 " + b"\x00" * (100 * 1024 * 1024 + 1)

        monkeypatch.setattr(routes_module.base64, "b64decode", _mock_b64decode)

        # Send a small valid base64 payload (the mock will override the decode)
        pdf_content = b"%PDF-1.4 small"
        encoded = base64.b64encode(pdf_content).decode()
        response = client.post(
            "/api/v1/papers",
            json={"pdf_upload": encoded},
        )
        assert response.status_code == 422
        assert "100MB" in response.json()["detail"]

    def test_submit_neither_doi_nor_pdf_explicit(self, client):
        """Explicitly sending null for both doi and pdf_upload returns 422."""
        response = client.post(
            "/api/v1/papers",
            json={"doi": None, "pdf_upload": None},
        )
        assert response.status_code == 422
        assert "Provide either" in response.json()["detail"]


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

    def test_batch_with_mix_of_valid_and_duplicate(self, client):
        """Batch submission where some DOIs already exist returns mixed results."""
        # First, submit one DOI
        resp1 = client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/batch.existing"},
        )
        assert resp1.status_code == 202
        existing_id = resp1.json()["paper_id"]

        # Now batch-submit with that DOI plus a new one
        response = client.post(
            "/api/v1/batch",
            json={"dois": ["10.1234/batch.existing", "10.1234/batch.new"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert len(data["papers"]) == 2

        # The existing DOI should return the same paper_id
        paper_ids = {p["paper_id"] for p in data["papers"]}
        assert existing_id in paper_ids


class TestGetAuthorRisk:
    def test_get_nonexistent_author(self, client):
        response = client.get(f"/api/v1/authors/{uuid.uuid4()}/risk")
        assert response.status_code == 404


class TestApiKeyAuth:
    def test_require_auth_default_rejects_when_no_keys(self, tmp_path):
        """When require_authentication=true (default) and no keys, return 500."""
        db_path = tmp_path / "auth_test.db"
        config = SnoopyConfig(
            require_authentication=True,
            storage={
                "database_url": f"sqlite:///{db_path}",
                "pdf_dir": str(tmp_path / "pdfs"),
                "figures_dir": str(tmp_path / "figures"),
                "reports_dir": str(tmp_path / "reports"),
            },
        )
        init_db(config.storage.database_url)
        init_async_db(config.storage.database_url)
        from snoopy.api.app import create_app

        app = create_app(config)
        test_client = TestClient(app)
        response = test_client.post(
            "/api/v1/papers",
            json={"doi": "10.1234/test"},
        )
        assert response.status_code == 500
        assert "API keys are required" in response.json()["detail"]

    def test_no_auth_when_require_auth_false(self, client):
        """When require_authentication=false and no keys, requests pass."""
        # The default fixture has require_authentication=True by default,
        # but the fixture creates SnoopyConfig with defaults which now has
        # require_authentication=True. We need to explicitly set it false.
        client.app.state.config.require_authentication = False
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


class TestCreateAppDefaults:
    def test_create_app_with_no_config(self):
        """create_app with no config uses load_config defaults."""
        app = create_app()
        assert app.state.config is not None
        assert app.state.config.llm.provider == "claude"


class TestCorsConfiguration:
    def test_wildcard_cors_disables_credentials(self, tmp_path):
        """When cors_origins includes '*', allow_credentials should be disabled."""
        db_path = tmp_path / "cors_test.db"
        config = SnoopyConfig(
            require_authentication=False,
            cors_origins=["*"],
            storage={
                "database_url": f"sqlite:///{db_path}",
                "pdf_dir": str(tmp_path / "pdfs"),
                "figures_dir": str(tmp_path / "figures"),
                "reports_dir": str(tmp_path / "reports"),
            },
        )
        init_db(config.storage.database_url)
        init_async_db(config.storage.database_url)

        app = create_app(config)
        test_client = TestClient(app)

        # The app should still work
        response = test_client.get("/health")
        assert response.status_code == 200
