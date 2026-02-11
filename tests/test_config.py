"""Tests for configuration loading."""

from snoopy.config import SnoopyConfig, load_config


class TestConfigDefaults:
    def test_default_config(self):
        config = SnoopyConfig()
        assert config.llm.provider == "claude"
        assert config.llm.model_screen == "claude-haiku-4-5-20251001"
        assert config.storage.database_url == "sqlite:///snoopy.db"
        assert config.analysis.ela_quality == 95
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
