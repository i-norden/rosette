"""Shared test fixtures for snoopy tests."""

from __future__ import annotations


import numpy as np
import pytest
from PIL import Image

from snoopy.config import SnoopyConfig
from snoopy.db.session import init_db, get_session


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def test_config(tmp_path):
    """Provide a test configuration with temp paths."""
    db_path = tmp_path / "test.db"
    return SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )


@pytest.fixture
def db_session(test_config):
    """Provide a clean database session for testing."""
    init_db(test_config.storage.database_url)
    with get_session() as session:
        yield session


@pytest.fixture
def sample_image(tmp_path) -> str:
    """Create a simple test image and return its path."""
    img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
    path = str(tmp_path / "test_image.png")
    img.save(path)
    return path


@pytest.fixture
def sample_jpeg(tmp_path) -> str:
    """Create a test JPEG image and return its path."""
    img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
    path = str(tmp_path / "test_image.jpg")
    img.save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def clean_image(tmp_path) -> str:
    """Create a clean (unmanipulated) gradient image."""
    arr = np.zeros((200, 300, 3), dtype=np.uint8)
    for i in range(200):
        arr[i, :, :] = int(i / 200 * 255)
    img = Image.fromarray(arr)
    path = str(tmp_path / "clean_image.jpg")
    img.save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def manipulated_image(tmp_path) -> str:
    """Create an image with a cloned region (for testing clone detection)."""
    arr = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    # Clone a region from top-left to bottom-right
    arr[250:350, 250:350, :] = arr[50:150, 50:150, :]
    img = Image.fromarray(arr)
    path = str(tmp_path / "manipulated_image.png")
    img.save(path)
    return path
