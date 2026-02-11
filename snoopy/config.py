"""Configuration management with YAML files and environment variable overrides."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Pattern for ${ENV_VAR} and ${ENV_VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}:]+)(?::-([^}]*))?\}$")


class LLMConfig(BaseModel):
    provider: str = "claude"
    model_screen: str = "claude-haiku-4-5-20251001"
    model_analyze: str = "claude-sonnet-4-5-20250929"
    model_proof: str = "claude-opus-4-6"
    use_batch_api: bool = True
    max_concurrent_requests: int = 5


class DiscoveryConfig(BaseModel):
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    ncbi_api_key: str | None = None
    unpaywall_email: str | None = None
    scimago_csv_path: str = "data/scimago.csv"


class PriorityConfig(BaseModel):
    min_citations: int = 50
    journal_quartile_filter: str | None = "Q1"
    min_priority_score: float = 40.0


class AnalysisConfig(BaseModel):
    ela_quality: int = 95
    clone_min_matches: int = 10
    noise_block_size: int = 64
    llm_screening_confidence_threshold: float = 0.5
    convergence_required: bool = True

    # Method weights for composite scoring (research-based)
    weight_clone_detection: float = 0.85
    weight_phash: float = 0.90
    weight_pvalue_check: float = 0.80
    weight_grim: float = 0.60
    weight_noise: float = 0.50
    weight_ela: float = 0.35
    weight_benford: float = 0.30
    weight_duplicate_check: float = 0.25
    weight_llm_vision: float = 0.70

    # Convergence thresholds (previously magic numbers)
    convergence_confidence_threshold: float = 0.6
    convergence_confidence_boost: float = 0.1
    single_method_max_severity: str = "medium"

    # ELA severity thresholds
    ela_high_threshold: float = 60.0
    ela_medium_threshold: float = 40.0
    ela_low_threshold: float = 25.0

    # Clone severity thresholds
    clone_high_inliers: int = 60
    clone_high_ratio: float = 0.35
    clone_medium_inliers: int = 40
    clone_medium_ratio: float = 0.25
    clone_low_inliers: int = 20
    clone_low_ratio: float = 0.15

    # Noise severity thresholds
    noise_high_ratio: float = 20.0
    noise_medium_ratio: float = 10.0
    noise_low_ratio: float = 5.0

    # Noise analysis
    noise_intensity_bin_width: int = 32

    # Figure extraction
    min_figure_width: int = 50
    min_figure_height: int = 50


class StorageConfig(BaseModel):
    database_url: str = "sqlite:///snoopy.db"
    pdf_dir: str = "data/pdfs"
    figures_dir: str = "data/figures"
    reports_dir: str = "data/reports"


class SnoopyConfig(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api_keys: list[str] | None = Field(
        default=None,
        description="List of valid API keys. When None, all requests are allowed.",
    )

    model_config = {"env_prefix": "SNOOPY_", "env_nested_delimiter": "__"}


def _resolve_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ${ENV_VAR} and ${ENV_VAR:-default} references in config values."""
    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, str):
            match = _ENV_VAR_PATTERN.match(value)
            if match:
                env_name = match.group(1)
                default_value = match.group(2)  # None if no default specified
                env_value = os.environ.get(env_name)
                if env_value is not None:
                    resolved[key] = env_value
                elif default_value is not None:
                    resolved[key] = default_value
                else:
                    logger.warning(
                        "Environment variable '%s' referenced in config key '%s' is not set. "
                        "Use ${%s:-default} syntax to provide a default value.",
                        env_name,
                        key,
                        env_name,
                    )
                    resolved[key] = None
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def _resolve_paths(data: dict, config_dir: Path) -> dict:
    """Resolve relative storage paths to absolute paths relative to config directory."""
    resolved = dict(data)
    if "storage" in resolved and isinstance(resolved["storage"], dict):
        storage = resolved["storage"]
        for key in ("pdf_dir", "figures_dir", "reports_dir"):
            if key in storage and isinstance(storage[key], str):
                path = Path(storage[key])
                if not path.is_absolute():
                    storage[key] = str(config_dir / path)
    return resolved


def load_config(config_path: str | Path | None = None) -> SnoopyConfig:
    """Load configuration from YAML file with environment variable overrides.

    Resolution order (later wins):
    1. Default values in Pydantic models
    2. YAML config file values
    3. Environment variables (SNOOPY__LLM__PROVIDER, etc.)
    """
    yaml_data = {}
    config_dir = Path.cwd()

    if config_path is None:
        default_path = Path("config/default.yaml")
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            config_dir = path.resolve().parent
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            yaml_data = _resolve_env_vars(raw)
            yaml_data = _resolve_paths(yaml_data, config_dir)

    return SnoopyConfig(**yaml_data)
