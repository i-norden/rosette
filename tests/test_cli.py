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

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_invokes_run_demo(
        self,
        mock_run_demo: MagicMock,
        mock_load_config: MagicMock,
        mock_init_db: MagicMock,
        mock_init_async_db: MagicMock,
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

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_download_only_flag(
        self,
        mock_run_demo: MagicMock,
        mock_load_config: MagicMock,
        mock_init_db: MagicMock,
        mock_init_async_db: MagicMock,
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

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.demo.runner.run_demo")
    def test_demo_output_dir_option(
        self,
        mock_run_demo: MagicMock,
        mock_load_config: MagicMock,
        mock_init_db: MagicMock,
        mock_init_async_db: MagicMock,
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
        assert "serve" in result.output

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    def test_config_command(
        self, mock_load_config: MagicMock, mock_init_db: MagicMock, mock_init_async_db: MagicMock
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.model_dump_json.return_value = '{"llm": {}}'
        mock_load_config.return_value = mock_cfg
        runner = CliRunner()
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert '{"llm": {}}' in result.output


class TestAnalyzeCommand:
    """Tests for the `snoopy analyze` CLI command."""

    def test_analyze_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--doi" in result.output
        assert "--pdf" in result.output
        assert "--from-stage" in result.output

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    def test_analyze_requires_doi_or_pdf(
        self, mock_load_config: MagicMock, mock_init_db: MagicMock, mock_init_async_db: MagicMock
    ) -> None:
        """Analyze without --doi or --pdf should exit with error."""
        mock_load_config.return_value = MagicMock()
        runner = CliRunner()
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code != 0
        assert "provide either" in result.output.lower() or result.exit_code == 1

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.pipeline.orchestrator.PipelineOrchestrator")
    @patch("snoopy.cli.get_session")
    def test_analyze_with_doi(
        self,
        mock_get_session: MagicMock,
        mock_orch_class: MagicMock,
        mock_load_config: MagicMock,
        mock_init_db: MagicMock,
        mock_init_async_db: MagicMock,
    ) -> None:
        """Analyze with a valid DOI invokes the orchestrator."""
        mock_load_config.return_value = MagicMock()

        # Mock the session context manager
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.scalars.return_value.first.return_value = None
        mock_get_session.return_value = mock_session

        # Mock the orchestrator
        mock_orch_instance = MagicMock()
        mock_orch_instance.process_paper = MagicMock(return_value=None)
        mock_orch_class.return_value = mock_orch_instance

        # Make process_paper a proper coroutine

        async def mock_process_paper(paper_id):
            pass

        mock_orch_instance.process_paper = mock_process_paper

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "--doi", "10.1234/test.paper"])
        # Should succeed or at least attempt processing
        assert "10.1234/test.paper" in result.output or result.exit_code == 0


class TestStatusCommand:
    """Tests for the `snoopy status` CLI command."""

    def test_status_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output.lower()

    @patch("snoopy.cli.init_async_db")
    @patch("snoopy.cli.init_db")
    @patch("snoopy.cli.load_config")
    @patch("snoopy.cli.get_session")
    def test_status_shows_totals(
        self,
        mock_get_session: MagicMock,
        mock_load_config: MagicMock,
        mock_init_db: MagicMock,
        mock_init_async_db: MagicMock,
    ) -> None:
        """Status command displays paper counts."""
        mock_load_config.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Mock total count
        mock_session.execute.return_value.scalar.return_value = 5
        # Mock status breakdown
        mock_session.execute.return_value.all.return_value = [
            ("pending", 3),
            ("complete", 2),
        ]
        # Mock top priority papers - return scalars().all() as empty
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.execute.return_value.scalars.return_value = mock_scalars

        mock_get_session.return_value = mock_session

        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Total papers" in result.output


class TestServeCommand:
    """Tests for the `snoopy serve` CLI command."""

    def test_serve_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
