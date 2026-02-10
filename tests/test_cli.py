"""Tests for the snoopy CLI, focusing on the demo command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from snoopy.cli import main


class TestDemoCommand:
    """Tests for the `snoopy demo` CLI command."""

    def test_demo_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()
        assert "--download-only" in result.output
        assert "--skip-llm" in result.output
        assert "--output-dir" in result.output
        assert "--download-rsiil" in result.output

    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_invokes_run_demo(
        self, mock_run_demo: MagicMock, mock_load_config: MagicMock, mock_init_db: MagicMock
    ) -> None:
        mock_load_config.return_value = MagicMock()
        mock_run_demo.return_value = []
        runner = CliRunner()
        result = runner.invoke(main, ["demo"])
        assert result.exit_code == 0
        mock_run_demo.assert_called_once_with(
            download_only=False,
            skip_llm=True,
            output_dir=None,
            download_rsiil=False,
        )

    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_download_only_flag(
        self, mock_run_demo: MagicMock, mock_load_config: MagicMock, mock_init_db: MagicMock
    ) -> None:
        mock_load_config.return_value = MagicMock()
        mock_run_demo.return_value = []
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--download-only"])
        assert result.exit_code == 0
        mock_run_demo.assert_called_once_with(
            download_only=True,
            skip_llm=True,
            output_dir=None,
            download_rsiil=False,
        )

    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_output_dir_option(
        self, mock_run_demo: MagicMock, mock_load_config: MagicMock, mock_init_db: MagicMock
    ) -> None:
        mock_load_config.return_value = MagicMock()
        mock_run_demo.return_value = []
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--output-dir", "/tmp/reports"])
        assert result.exit_code == 0
        mock_run_demo.assert_called_once_with(
            download_only=False,
            skip_llm=True,
            output_dir="/tmp/reports",
            download_rsiil=False,
        )


class TestMainGroup:
    """Basic CLI group tests."""

    def test_help_shows_all_commands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "demo" in result.output
        assert "analyze" in result.output
        assert "discover" in result.output
        assert "batch" in result.output
        assert "report" in result.output
        assert "status" in result.output
        assert "config" in result.output

    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    def test_config_command(
        self, mock_load_config: MagicMock, mock_init_db: MagicMock
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.model_dump_json.return_value = '{"llm": {}}'
        mock_load_config.return_value = mock_cfg
        runner = CliRunner()
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert '{"llm": {}}' in result.output
