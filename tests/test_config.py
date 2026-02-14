"""Tests for configuration loading."""

from snoopy.config import (
    AnalysisConfig,
    CloneConfig,
    ELAConfig,
    NoiseConfig,
    SnoopyConfig,
    StatisticalConfig,
    WesternBlotConfig,
    load_config,
)


class TestConfigDefaults:
    def test_default_config(self):
        config = SnoopyConfig()
        assert config.llm.provider == "claude"
        assert config.llm.model_screen == "claude-haiku-4-5-20251001"
        assert config.storage.database_url == "sqlite:///snoopy.db"
        assert config.analysis.ela_quality == 80
        assert config.priority.min_citations == 50

    def test_custom_config(self):
        config = SnoopyConfig(
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

    def test_backward_compatible_flat_fields(self):
        """Flat fields should still work for backward compat."""
        config = AnalysisConfig()
        assert config.ela_quality == 80
        assert config.clone_min_matches == 10
        assert config.noise_block_size == 64
