"""Shared fixtures for campaign test package.

Auto-patches ClaudeProvider so PipelineOrchestrator can be instantiated
without an Anthropic API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_claude_provider():
    """Globally mock ClaudeProvider for all campaign tests."""
    with patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls
