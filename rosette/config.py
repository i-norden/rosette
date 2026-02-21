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
    cache_enabled: bool = True
    cache_dir: str = "data/llm_cache"


class DiscoveryConfig(BaseModel):
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    ncbi_api_key: str | None = None
    unpaywall_email: str | None = None
    contact_email: str = "research@example.com"
    scimago_csv_path: str = "data/scimago.csv"


class PriorityConfig(BaseModel):
    min_citations: int = 50
    journal_quartile_filter: str | None = "Q1"
    min_priority_score: float = 40.0


class ELAConfig(BaseModel):
    """Error Level Analysis configuration."""

    quality: int = 80  # Quality 80 per forensics literature (75-85 range)
    high_threshold: float = 60.0
    medium_threshold: float = 40.0
    low_threshold: float = 25.0
    min_max_diff: float = 15.0


class CloneConfig(BaseModel):
    """Clone / copy-move detection configuration."""

    min_matches: int = 10
    high_inliers: int = 60
    high_ratio: float = 0.35
    medium_inliers: int = 40
    medium_ratio: float = 0.25
    low_inliers: int = 20
    low_ratio: float = 0.15
    min_inlier_ratio: float = 0.15
    spatial_distance: float = 20.0
    ransac_threshold: float = 5.0
    cluster_radius: float = 50.0
    feature_extractor: str = "sift"


class NoiseConfig(BaseModel):
    """Noise analysis configuration."""

    block_size: int = 64
    intensity_bin_width: int = 32
    high_ratio: float = 50.0
    medium_ratio: float = 25.0
    low_ratio: float = 10.0
    max_ratio_threshold: float = 25.0


class StatisticalConfig(BaseModel):
    """Statistical test configuration."""

    terminal_digit_uniformity_alpha: float = 0.01
    variance_ratio_min_sds: int = 3
    tortured_phrase_min_matches: int = 2


class WesternBlotConfig(BaseModel):
    """Western blot analysis configuration."""

    lane_threshold_multiplier: float = 0.7
    duplicate_correlation: float = 0.95
    splice_border_px: int = 3


class AnalysisConfig(BaseModel):
    # Nested sub-configs
    ela: ELAConfig = Field(default_factory=ELAConfig)
    clone: CloneConfig = Field(default_factory=CloneConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    western_blot: WesternBlotConfig = Field(default_factory=WesternBlotConfig)

    llm_screening_confidence_threshold: float = 0.5
    convergence_required: bool = True

    # Method weights for composite scoring (calibrated against demo benchmark)
    # High weights = highly specific methods (clone detection, phash, p-value)
    # Lower weights = methods prone to PDF compression false positives
    weight_clone_detection: float = Field(default=0.85, ge=0.0, le=1.0)
    weight_phash: float = Field(default=0.90, ge=0.0, le=1.0)
    weight_pvalue_check: float = Field(default=0.80, ge=0.0, le=1.0)
    weight_grim: float = Field(default=0.60, ge=0.0, le=1.0)
    weight_noise: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_ela: float = Field(default=0.70, ge=0.0, le=1.0)
    weight_benford: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_duplicate_check: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_llm_vision: float = Field(default=0.70, ge=0.0, le=1.0)
    weight_metadata_forensics: float = Field(default=0.3, ge=0.0, le=1.0)

    # Convergence thresholds
    convergence_confidence_threshold: float = 0.60
    convergence_confidence_boost: float = 0.1
    single_method_max_severity: str = "medium"
    single_method_max_confidence: float = 0.7

    # Figure extraction
    min_figure_width: int = 50
    min_figure_height: int = 50

    # Phase 2 new detection method weights.
    # Compression-sensitive methods (DCT, JPEG ghost, FFT) get weight <= 0.30
    # so they contribute to scoring but NOT to convergence determination
    # (convergence requires weight > 0.3).
    weight_dct_analysis: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_jpeg_ghost: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_fft_analysis: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_grimmer: float = Field(default=0.60, ge=0.0, le=1.0)
    weight_terminal_digit: float = Field(default=0.45, ge=0.0, le=1.0)
    weight_distribution_fit: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_variance_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    weight_tortured_phrases: float = Field(default=0.80, ge=0.0, le=1.0)
    weight_temporal_patterns: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_sprite: float = Field(default=0.65, ge=0.0, le=1.0)

    # DCT analysis
    dct_periodicity_threshold: float = 0.3

    # JPEG ghost detection
    jpeg_ghost_quality_range_start: int = 50
    jpeg_ghost_quality_range_end: int = 95
    jpeg_ghost_step: int = 5

    # FFT frequency analysis
    fft_spectral_anomaly_threshold: float = 2.5

    # SSIM cross-reference
    ssim_duplicate_threshold: float = 0.95

    @property
    def method_weights(self) -> dict[str, float]:
        """Build a method-name → weight mapping for evidence aggregation."""
        return {
            "ela": self.weight_ela,
            "clone_detection": self.weight_clone_detection,
            "noise_analysis": self.weight_noise,
            "dct_analysis": self.weight_dct_analysis,
            "jpeg_ghost": self.weight_jpeg_ghost,
            "fft_analysis": self.weight_fft_analysis,
            "metadata_forensics": self.weight_metadata_forensics,
            "phash": self.weight_phash,
            "grim": self.weight_grim,
            "pvalue_check": self.weight_pvalue_check,
            "benford": self.weight_benford,
            "duplicate_check": self.weight_duplicate_check,
            "llm_vision": self.weight_llm_vision,
            "llm_screening": self.weight_llm_vision,
            "grimmer": self.weight_grimmer,
            "terminal_digit": self.weight_terminal_digit,
            "distribution_fit": self.weight_distribution_fit,
            "variance_ratio": self.weight_variance_ratio,
            "tortured_phrases": self.weight_tortured_phrases,
            "sprite": self.weight_sprite,
        }


class CampaignConfig(BaseModel):
    auto_risk_promotion_threshold: float = 30.0
    max_authors_per_paper: int = 20  # cap on co-authors to expand per paper
    max_papers_per_author: int = 50  # cap on papers to fetch per author
    hash_match_max_distance: int = 10  # perceptual hash Hamming distance threshold
    network_min_cluster_size: int = 3  # minimum authors for fraud cluster reporting
    batch_concurrency: int = 5  # concurrent papers in batch processing
    hash_prefix_length: int = 2  # prefix length for hash bucketing (256 buckets)


class StorageConfig(BaseModel):
    database_url: str = "sqlite:///rosette.db"
    pdf_dir: str = "data/pdfs"
    figures_dir: str = "data/figures"
    reports_dir: str = "data/reports"


class RosetteConfig(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    campaign: CampaignConfig = Field(default_factory=CampaignConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    require_authentication: bool = Field(
        default=True,
        description="When true, API keys must be configured. Set to false for dev mode.",
    )
    api_keys: list[str] | None = Field(
        default=None,
        description="List of valid API keys. When None, all requests are allowed.",
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Empty list blocks all cross-origin requests.",
    )
    rate_limit: str = Field(
        default="60/minute",
        description="Default rate limit (slowapi format, e.g. '60/minute').",
    )

    model_config = {"env_prefix": "ROSETTE_", "env_nested_delimiter": "__"}


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


def load_config(config_path: str | Path | None = None) -> RosetteConfig:
    """Load configuration from YAML file with environment variable overrides.

    Resolution order (later wins):
    1. Default values in Pydantic models
    2. YAML config file values
    3. Environment variables (ROSETTE__LLM__PROVIDER, etc.)
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

    return RosetteConfig(**yaml_data)
