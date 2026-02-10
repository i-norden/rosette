"""Pipeline stage definitions for paper analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StageStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


PIPELINE_STAGES = [
    "discover",
    "prioritize",
    "download",
    "extract_text",
    "extract_figures",
    "extract_stats",
    "classify_figures",
    "analyze_images",
    "analyze_stats",
    "aggregate",
    "report",
]


@dataclass
class StageResult:
    stage: str
    status: StageStatus
    details: str = ""
    error: str | None = None


def get_next_stage(last_completed: str | None) -> str | None:
    """Return the next stage after the last completed one, or the first stage."""
    if last_completed is None:
        return PIPELINE_STAGES[0]
    try:
        idx = PIPELINE_STAGES.index(last_completed)
        if idx + 1 < len(PIPELINE_STAGES):
            return PIPELINE_STAGES[idx + 1]
        return None
    except ValueError:
        return PIPELINE_STAGES[0]


def get_remaining_stages(last_completed: str | None) -> list[str]:
    """Return all stages remaining after the last completed one."""
    if last_completed is None:
        return list(PIPELINE_STAGES)
    try:
        idx = PIPELINE_STAGES.index(last_completed)
        return PIPELINE_STAGES[idx + 1:]
    except ValueError:
        return list(PIPELINE_STAGES)
