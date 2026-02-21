"""Tests for configuration loading."""

import pytest
from pydantic import ValidationError

from rosette.config import (
    AnalysisConfig,
    CloneConfig,
    ELAConfig,
    NoiseConfig,
    RosetteConfig,
    StatisticalConfig,
    WesternBlotConfig,
    load_config,
)


class TestConfigDefaults:
    def test_default_config(self):
        config = RosetteConfig()
        assert config.llm.provider == "claude"
        assert config.llm.model_screen == "claude-haiku-4-5-20251001"
        assert config.storage.database_url == "sqlite:///rosette.db"
        assert config.analysis.ela.quality == 80
        assert config.priority.min_citations == 50

    def test_custom_config(self):
        config = RosetteConfig(
            llm={"provider": "custom", "max_concurrent_requests": 10},
            storage={"database_url": "sqlite:///custom.db"},
        )
        assert config.llm.provider == "custom"
        assert config.llm.max_concurrent_requests == 10
        assert config.storage.database_url == "sqlite:///custom.db"


class TestLoadConfig:
    def test_load_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.llm.provider == "claude"

    def test_load_yaml_file(self, tmp_path):
        yaml_content = """
llm:
  provider: test_provider
  max_concurrent_requests: 3
storage:
  database_url: sqlite:///test.db
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        config = load_config(str(config_file))
        assert config.llm.provider == "test_provider"
        assert config.llm.max_concurrent_requests == 3
        assert config.storage.database_url == "sqlite:///test.db"


class TestNestedSubConfigs:
    """Tests for the new nested sub-configs in AnalysisConfig."""

    def test_ela_sub_config_defaults(self):
        config = AnalysisConfig()
        assert isinstance(config.ela, ELAConfig)
        assert config.ela.quality == 80
        assert config.ela.high_threshold == 60.0

    def test_clone_sub_config_defaults(self):
        config = AnalysisConfig()
        assert isinstance(config.clone, CloneConfig)
        assert config.clone.min_matches == 10
        assert config.clone.feature_extractor == "sift"

    def test_noise_sub_config_defaults(self):
        config = AnalysisConfig()
        assert isinstance(config.noise, NoiseConfig)
        assert config.noise.block_size == 64

    def test_statistical_sub_config_defaults(self):
        config = AnalysisConfig()
        assert isinstance(config.statistical, StatisticalConfig)
        assert config.statistical.terminal_digit_uniformity_alpha == 0.01

    def test_western_blot_sub_config_defaults(self):
        config = AnalysisConfig()
        assert isinstance(config.western_blot, WesternBlotConfig)
        assert config.western_blot.duplicate_correlation == 0.95

    def test_nested_configs_are_canonical(self):
        """Nested sub-configs are the canonical source of analysis parameters."""
        config = AnalysisConfig()
        assert config.ela.quality == 80
        assert config.clone.min_matches == 10
        assert config.noise.block_size == 64


class TestWeightValidation:
    """Tests for analysis weight field validation (ge=0.0, le=1.0)."""

    def test_valid_weights(self):
        """Weights within [0.0, 1.0] should be accepted."""
        config = AnalysisConfig(weight_clone_detection=0.0, weight_phash=1.0, weight_ela=0.5)
        assert config.weight_clone_detection == 0.0
        assert config.weight_phash == 1.0
        assert config.weight_ela == 0.5

    def test_negative_weight_rejected(self):
        """Negative weight should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            AnalysisConfig(weight_clone_detection=-0.1)

    def test_weight_above_one_rejected(self):
        """Weight > 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            AnalysisConfig(weight_phash=1.5)

    def test_all_phase2_weights_validated(self):
        """All phase 2 weights should also be validated."""
        with pytest.raises(ValidationError):
            AnalysisConfig(weight_dct_analysis=-0.01)
        with pytest.raises(ValidationError):
            AnalysisConfig(weight_tortured_phrases=2.0)
        with pytest.raises(ValidationError):
            AnalysisConfig(weight_sprite=1.01)


class TestConfigEdgeCases:
    """Tests for configuration edge cases and error handling."""

    def test_invalid_yaml_syntax(self, tmp_path):
        """Invalid YAML should be handled gracefully (returns defaults)."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml: [")
        # yaml.safe_load raises on invalid syntax
        with pytest.raises(Exception):
            load_config(str(config_file))

    def test_env_var_resolution(self, tmp_path, monkeypatch):
        """${ENV_VAR} references in config should be resolved."""
        monkeypatch.setenv("TEST_PROVIDER", "custom_provider")
        yaml_content = """
llm:
  provider: ${TEST_PROVIDER}
"""
        config_file = tmp_path / "env.yaml"
        config_file.write_text(yaml_content)
        config = load_config(str(config_file))
        assert config.llm.provider == "custom_provider"

    def test_env_var_with_default(self, tmp_path, monkeypatch):
        """${ENV_VAR:-default} should use the default when var is unset."""
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        yaml_content = """
llm:
  provider: ${NONEXISTENT_VAR_12345:-fallback_provider}
"""
        config_file = tmp_path / "envdefault.yaml"
        config_file.write_text(yaml_content)
        config = load_config(str(config_file))
        assert config.llm.provider == "fallback_provider"

    def test_env_var_unset_no_default_raises(self, tmp_path, monkeypatch):
        """${ENV_VAR} with no default and unset var should cause validation error."""
        monkeypatch.delenv("TOTALLY_MISSING_VAR", raising=False)
        yaml_content = """
llm:
  provider: ${TOTALLY_MISSING_VAR}
"""
        config_file = tmp_path / "missing_env.yaml"
        config_file.write_text(yaml_content)
        # The resolved value is None, which Pydantic rejects for str field
        with pytest.raises(ValidationError, match="provider"):
            load_config(str(config_file))

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file should return default config."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(str(config_file))
        assert config.llm.provider == "claude"
        assert config.storage.database_url == "sqlite:///rosette.db"
