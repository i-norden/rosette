"""Configuration management with YAML files and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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
    scimago_csv_path: str = "data/scimago.csv"


class PriorityConfig(BaseModel):
    min_citations: int = 50
    journal_quartile_filter: str | None = "Q1"
    min_priority_score: float = 40.0


class AnalysisConfig(BaseModel):
    ela_quality: int = 95
    clone_min_matches: int = 30
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

    model_config = {"env_prefix": "SNOOPY_", "env_nested_delimiter": "__"}


def _resolve_env_vars(data: dict) -> dict:
    """Recursively resolve ${ENV_VAR} references in config values."""
    resolved = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_name = value[2:-1]
            resolved[key] = os.environ.get(env_name)
        else:
            resolved[key] = value
    return resolved


def load_config(config_path: str | Path | None = None) -> SnoopyConfig:
    """Load configuration from YAML file with environment variable overrides.

    Resolution order (later wins):
    1. Default values in Pydantic models
    2. YAML config file values
    3. Environment variables (SNOOPY__LLM__PROVIDER, etc.)
    """
    yaml_data = {}
    if config_path is None:
        default_path = Path("config/default.yaml")
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            yaml_data = _resolve_env_vars(raw)

    return SnoopyConfig(**yaml_data)
