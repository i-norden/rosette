"""Main pipeline orchestrator with resumability and bounded concurrency."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from snoopy.analysis.cross_reference import compute_ahash, compute_phash
from snoopy.analysis.evidence import aggregate_findings
from snoopy.analysis.llm_vision import analyze_figure_detailed, classify_figure, screen_figure
from snoopy.analysis.run_analysis import (
    run_dct_analysis,
    run_frequency_analysis,
    run_image_forensics,
    run_jpeg_ghost_analysis,
    run_sprite_analysis,
    run_statistical_tests,
    run_tortured_phrases,
    run_western_blot_analysis,
)
from snoopy.analysis.statistical import benford_test, duplicate_value_check, grim_test, pvalue_check
from snoopy.config import SnoopyConfig
from snoopy.db.models import Figure, Finding, Paper, ProcessingLog, Report
from snoopy.db.session import get_async_session
from snoopy.extraction.figure_extractor import extract_figures
from snoopy.extraction.pdf_parser import download_pdf, extract_text
from snoopy.extraction.stats_extractor import extract_means_and_ns, extract_test_statistics
from snoopy.extraction.table_extractor import extract_tables
from snoopy.llm.claude import ClaudeProvider
from snoopy.pipeline.stages import PIPELINE_STAGES, _LEGACY_STAGE_MAP
from snoopy.reporting.proof import generate_html_report, generate_markdown_report

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages the full analysis pipeline with resumability."""

    def __init__(self, config: SnoopyConfig):
        self.config = config
        self.llm_provider = ClaudeProvider(
            default_model=config.llm.model_analyze,
        )
        # NOTE: Callers must call init_async_db() before constructing the orchestrator.
        # The FastAPI lifespan and CLI entry points handle this.
        # Explicit stage handler dispatch table
        self._stage_handlers: dict[str, Any] = {
            "download": self._run_download,
            "extract_text": self._run_extract_text,
            "extract_figures": self._run_extract_figures,
            "extract_stats": self._run_extract_stats,
            "classify_figures": self._run_classify_figures,
            "analyze_images_auto": self._run_analyze_images_auto,
            "analyze_images_llm": self._run_analyze_images_llm,
            "analyze_stats": self._run_analyze_stats,
            "aggregate": self._run_aggregate,
            "report": self._run_report,
        }

    async def _log_stage(self, paper_id: str, stage: str, status: str, details: str = "") -> None:
        async with get_async_session() as session:
            now = datetime.now(timezone.utc)
            log = ProcessingLog(
                paper_id=paper_id,
                stage=stage,
                status=status,
                details=details,
                started_at=now if status == "started" else None,
                completed_at=now if status in ("completed", "failed") else None,
            )
            session.add(log)

    async def _get_last_completed_stage(self, paper_id: str) -> str | None:
        async with get_async_session() as session:
            result = await session.execute(
                select(ProcessingLog)
                .where(ProcessingLog.paper_id == paper_id)
                .where(ProcessingLog.status == "completed")
                .order_by(ProcessingLog.completed_at.desc())
            )
            log = result.scalars().first()
            return str(log.stage) if log else None

    async def _update_paper_status(
        self, paper_id: str, status: str, error: str | None = None
    ) -> None:
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if paper:
                paper.status = status
                paper.error_message = error
                paper.updated_at = datetime.now(timezone.utc)

    async def process_paper(self, paper_id: str) -> None:
        """Process a single paper through all remaining stages."""
        last_completed = await self._get_last_completed_stage(paper_id)

        if last_completed:
            try:
                start_idx = PIPELINE_STAGES.index(last_completed) + 1
            except ValueError:
                start_idx = 0
        else:
            start_idx = 0

        # Skip discovery/prioritize for individual paper processing
        paper_stages = [
            s for s in PIPELINE_STAGES[start_idx:] if s not in ("discover", "prioritize")
        ]

        await self._update_paper_status(paper_id, "analyzing")

        for stage in paper_stages:
            await self._log_stage(paper_id, stage, "started")
            try:
                handler = self._stage_handlers.get(stage)
                if handler is None:
                    raise ValueError(f"Unknown pipeline stage: {stage!r}")
                await handler(paper_id)
                await self._log_stage(paper_id, stage, "completed")
            except Exception as e:
                logger.error(f"Stage {stage} failed for paper {paper_id}: {e}")
                await self._log_stage(paper_id, stage, "failed", str(e))
                await self._update_paper_status(paper_id, "error", str(e))
                raise

        await self._update_paper_status(paper_id, "complete")

    async def process_paper_stages(
        self,
        paper_id: str,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force_stages: list[str] | None = None,
    ) -> None:
        """Process specific pipeline stages for a paper.

        Args:
            paper_id: The paper to process.
            from_stage: Start from this stage (inclusive). Defaults to next unfinished.
            to_stage: Stop after this stage (inclusive). Defaults to last stage.
            force_stages: If provided, run only these specific stages regardless of
                completion status.
        """
        if force_stages:
            # Expand legacy stage names (e.g. "analyze_images" -> auto + llm)
            expanded: list[str] = []
            for s in force_stages:
                if s in _LEGACY_STAGE_MAP:
                    expanded.extend(_LEGACY_STAGE_MAP[s])
                else:
                    expanded.append(s)
            paper_stages = [s for s in expanded if s in PIPELINE_STAGES]
        else:
            if from_stage:
                try:
                    start_idx = PIPELINE_STAGES.index(from_stage)
                except ValueError:
                    start_idx = 0
            else:
                last_completed = await self._get_last_completed_stage(paper_id)
                if last_completed:
                    try:
                        start_idx = PIPELINE_STAGES.index(last_completed) + 1
                    except ValueError:
                        start_idx = 0
                else:
                    start_idx = 0

            if to_stage:
                try:
                    end_idx = PIPELINE_STAGES.index(to_stage) + 1
                except ValueError:
                    end_idx = len(PIPELINE_STAGES)
            else:
                end_idx = len(PIPELINE_STAGES)

            paper_stages = [
                s for s in PIPELINE_STAGES[start_idx:end_idx] if s not in ("discover", "prioritize")
            ]

        await self._update_paper_status(paper_id, "analyzing")

        for stage in paper_stages:
            await self._log_stage(paper_id, stage, "started")
            try:
                handler = self._stage_handlers.get(stage)
                if handler is None:
                    raise ValueError(f"Unknown pipeline stage: {stage!r}")
                await handler(paper_id)
                await self._log_stage(paper_id, stage, "completed")
            except Exception as e:
                logger.error(f"Stage {stage} failed for paper {paper_id}: {e}")
                await self._log_stage(paper_id, stage, "failed", str(e))
                await self._update_paper_status(paper_id, "error", str(e))
                raise

        await self._update_paper_status(paper_id, "complete")

    async def _run_download(self, paper_id: str) -> None:
        """Download the PDF for a paper."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or paper.pdf_path:
                return

            if not paper.doi:
                logger.warning(f"Paper {paper_id} has no DOI, skipping download")
                return

            from snoopy.discovery.unpaywall import get_pdf_url

            email = self.config.discovery.unpaywall_email
            if not email:
                logger.warning(
                    "unpaywall_email not configured, skipping PDF download for %s", paper_id
                )
                return
            pdf_url = await get_pdf_url(str(paper.doi), email)
            if not pdf_url:
                logger.warning(f"No OA PDF found for {paper.doi}")
                return

            pdf_dir = Path(self.config.storage.pdf_dir)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(pdf_dir / f"{paper_id}.pdf")

            sha256 = await download_pdf(pdf_url, output_path)
            paper.pdf_path = output_path
            paper.pdf_sha256 = sha256

    async def _run_extract_text(self, paper_id: str) -> None:
        """Extract text from the paper's PDF and cache it on the Paper model."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return
            pages = await asyncio.to_thread(extract_text, str(paper.pdf_path))
            paper.full_text = "\n".join(p.text for p in pages)
            logger.info(f"Extracted {len(pages)} pages from {paper_id}")

    async def _run_extract_figures(self, paper_id: str) -> None:
        """Extract figures from the paper's PDF and store perceptual hashes."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return

            fig_dir = Path(self.config.storage.figures_dir) / paper_id
            fig_dir.mkdir(parents=True, exist_ok=True)

            figures = await asyncio.to_thread(extract_figures, str(paper.pdf_path), str(fig_dir))
            for fig_info in figures:
                # Compute and store perceptual hashes at extraction time
                phash_val = await asyncio.to_thread(compute_phash, fig_info.image_path)
                ahash_val = await asyncio.to_thread(compute_ahash, fig_info.image_path)

                figure = Figure(
                    paper_id=paper_id,
                    page_number=fig_info.page_number,
                    figure_label=fig_info.figure_label,
                    caption=fig_info.caption,
                    image_path=fig_info.image_path,
                    image_sha256=fig_info.image_sha256,
                    width=fig_info.width,
                    height=fig_info.height,
                    phash=phash_val,
                    ahash=ahash_val,
                )
                session.add(figure)

    async def _run_extract_stats(self, paper_id: str) -> None:
        """Extract statistical values from paper text."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return
            full_text = str(paper.full_text or "")
            if not full_text:
                pages = await asyncio.to_thread(extract_text, str(paper.pdf_path))
                full_text = "\n".join(p.text for p in pages)
                paper.full_text = full_text
            means = extract_means_and_ns(full_text)
            stats = extract_test_statistics(full_text)
            logger.info(
                f"Paper {paper_id}: found {len(means)} mean reports, {len(stats)} test statistics"
            )

    async def _run_classify_figures(self, paper_id: str) -> None:
        """Classify figure types using LLM.

        Gracefully degrades if the LLM provider is unavailable.
        """
        async with get_async_session() as session:
            result = await session.execute(select(Figure).where(Figure.paper_id == paper_id))
            figures = result.scalars().all()

            for figure in figures:
                if figure.image_path and Path(str(figure.image_path)).exists():
                    try:
                        fig_type = await classify_figure(str(figure.image_path), self.llm_provider)
                        figure.image_type = fig_type
                    except Exception as e:
                        logger.warning(
                            "LLM classification unavailable for figure %s, skipping: %s",
                            figure.id,
                            e,
                        )

    @staticmethod
    def _create_finding_from_analysis(
        session: Any,
        paper_id: str,
        figure: Any,
        analysis_type: str,
        finding_dict: dict,
    ) -> None:
        """Create and add a Finding to the session from an analysis result dict."""
        finding = Finding(
            paper_id=paper_id,
            figure_id=figure.id,
            analysis_type=analysis_type,
            severity=finding_dict["severity"],
            confidence=finding_dict["confidence"],
            title=finding_dict["title"],
            description=finding_dict["description"],
            evidence_json=json.dumps(finding_dict.get("evidence", {})),
        )
        session.add(finding)

    async def _run_analyze_images_auto(self, paper_id: str) -> None:
        """Run automated image forensics on all figures.

        Uses ``run_image_forensics`` from ``run_analysis.py`` for consistent
        severity/confidence thresholds, and runs ELA, clone, noise, DCT,
        JPEG ghost, and FFT in parallel per figure.
        """
        async with get_async_session() as session:
            result = await session.execute(select(Figure).where(Figure.paper_id == paper_id))
            figures = result.scalars().all()

            for figure in figures:
                img_path = str(figure.image_path or "")
                if not img_path or not Path(img_path).exists():
                    continue

                fig_id = str(figure.figure_label or figure.id)

                # Run all forensics methods in parallel via asyncio.gather
                tasks: list[Any] = [
                    asyncio.to_thread(
                        run_image_forensics,
                        img_path,
                        figure_id=fig_id,
                        config=self.config.analysis,
                    ),
                    asyncio.to_thread(
                        run_dct_analysis,
                        img_path,
                        figure_id=fig_id,
                        config=self.config.analysis,
                    ),
                    asyncio.to_thread(
                        run_jpeg_ghost_analysis,
                        img_path,
                        figure_id=fig_id,
                        config=self.config.analysis,
                    ),
                    asyncio.to_thread(
                        run_frequency_analysis,
                        img_path,
                        figure_id=fig_id,
                        config=self.config.analysis,
                    ),
                ]

                # Wire western blot analysis for appropriate figure types (6.2)
                is_western = str(figure.image_type or "").lower() in (
                    "western_blot",
                    "gel",
                )
                if is_western:
                    tasks.append(
                        asyncio.to_thread(
                            run_western_blot_analysis,
                            img_path,
                            figure_id=fig_id,
                        )
                    )

                results = await asyncio.gather(*tasks)

                all_findings: list[dict] = []
                for r in results:
                    all_findings.extend(r)
                for fd in all_findings:
                    self._create_finding_from_analysis(
                        session, paper_id, figure, fd.get("analysis_type", "unknown"), fd
                    )

    async def _run_analyze_images_llm(self, paper_id: str) -> None:
        """Run LLM-based image analysis (screening + detailed) on all figures.

        Gracefully degrades if the LLM provider is unavailable (no API key,
        rate limited, network error): logs a warning and continues with
        CV-only results.
        """
        async with get_async_session() as session:
            result = await session.execute(select(Figure).where(Figure.paper_id == paper_id))
            figures = result.scalars().all()

            for figure in figures:
                img_path = str(figure.image_path or "")
                if not img_path or not Path(img_path).exists():
                    continue

                try:
                    # LLM Vision - Stage 1 screening
                    screening = await screen_figure(
                        img_path, self.llm_provider, caption=str(figure.caption or "")
                    )

                    if (
                        screening.suspicious
                        and screening.confidence
                        >= self.config.analysis.llm_screening_confidence_threshold
                    ):
                        # Stage 2 detailed analysis
                        detailed = await analyze_figure_detailed(
                            img_path,
                            self.llm_provider,
                            caption=str(figure.caption or ""),
                            figure_type=str(figure.image_type or ""),
                        )

                        for vf in detailed.findings:
                            severity = (
                                "high"
                                if vf.confidence > 0.7
                                else "medium"
                                if vf.confidence > 0.4
                                else "low"
                            )
                            finding = Finding(
                                paper_id=paper_id,
                                figure_id=figure.id,
                                analysis_type="llm_vision",
                                severity=severity,
                                confidence=vf.confidence,
                                title=vf.finding_type,
                                description=vf.description,
                                evidence_json=json.dumps({"location": vf.location}),
                                model_used=detailed.model_used,
                                raw_response=detailed.raw_response,
                            )
                            session.add(finding)
                except Exception as e:
                    logger.warning(
                        "LLM analysis unavailable for figure %s in paper %s, "
                        "continuing with CV-only results: %s",
                        figure.id,
                        paper_id,
                        e,
                    )

    async def _run_analyze_stats(self, paper_id: str) -> None:
        """Run statistical integrity checks."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return

            full_text = str(paper.full_text or "")
            if not full_text:
                pages = await asyncio.to_thread(extract_text, str(paper.pdf_path))
                full_text = "\n".join(p.text for p in pages)
                paper.full_text = full_text

            # GRIM test
            means = extract_means_and_ns(full_text)
            for mr in means:
                grim_result = grim_test(mr.mean, mr.n)
                if not grim_result.consistent:
                    finding = Finding(
                        paper_id=paper_id,
                        analysis_type="grim",
                        severity="medium",
                        confidence=0.8,
                        title=f"GRIM inconsistency: M={mr.mean}, N={mr.n}",
                        description=(
                            f"Mean of {mr.mean} with N={mr.n} is not mathematically possible. "
                            f"Product {mr.mean}*{mr.n} = {grim_result.product:.2f}, "
                            f"nearest integer = {grim_result.nearest_integer}, "
                            f"difference = {grim_result.difference:.4f}. "
                            f"Context: {mr.context[:200]}"
                        ),
                        evidence_json=json.dumps(
                            {
                                "mean": mr.mean,
                                "n": mr.n,
                                "product": grim_result.product,
                            }
                        ),
                    )
                    session.add(finding)

            # P-value check
            test_stats = extract_test_statistics(full_text)
            for ts in test_stats:
                if ts.p_value is not None and ts.df is not None:
                    pval_result = pvalue_check(ts.test_type, ts.statistic, ts.df, ts.p_value)
                    if not pval_result.consistent:
                        finding = Finding(
                            paper_id=paper_id,
                            analysis_type="pvalue_check",
                            severity="high" if pval_result.significance_changed else "medium",
                            confidence=0.85,
                            title=f"P-value inconsistency: {ts.test_type} test",
                            description=(
                                f"Reported p={pval_result.reported_p}, computed p={pval_result.computed_p:.6f} "
                                f"(difference: {pval_result.difference:.6f}). "
                                f"{'Significance conclusion changed!' if pval_result.significance_changed else ''} "
                                f"Context: {ts.context[:200]}"
                            ),
                            evidence_json=json.dumps(
                                {
                                    "test_type": ts.test_type,
                                    "statistic": ts.statistic,
                                    "df": ts.df,
                                    "reported_p": pval_result.reported_p,
                                    "computed_p": pval_result.computed_p,
                                }
                            ),
                        )
                        session.add(finding)

            # Benford's law on tables
            tables = await asyncio.to_thread(extract_tables, str(paper.pdf_path))
            all_values = []
            for table in tables:
                for row in table.rows:
                    for cell in row:
                        try:
                            val = float(cell)
                            if val > 0:
                                all_values.append(val)
                        except (ValueError, TypeError):
                            pass

            if len(all_values) >= 50:
                bf = benford_test(all_values)
                if not bf.conforms:
                    finding = Finding(
                        paper_id=paper_id,
                        analysis_type="benford",
                        severity="low",
                        confidence=min(1.0 - bf.p_value, 0.95),
                        title="Benford's law deviation in numerical data",
                        description=(
                            f"Leading digit distribution of {bf.n_values} values deviates from "
                            f"Benford's law (chi2={bf.chi_squared:.2f}, p={bf.p_value:.6f})"
                        ),
                        evidence_json=json.dumps(
                            {
                                "chi_squared": bf.chi_squared,
                                "p_value": bf.p_value,
                                "observed": bf.observed_distribution,
                                "expected": bf.expected_distribution,
                            }
                        ),
                    )
                    session.add(finding)

            # Duplicate value check on tables
            for table in tables:
                dup = duplicate_value_check(table.rows)
                if dup.suspicious:
                    finding = Finding(
                        paper_id=paper_id,
                        analysis_type="duplicate_values",
                        severity="low",
                        confidence=0.5,
                        title=f"Suspicious value patterns in table (page {table.page_number})",
                        description=dup.details,
                        evidence_json=json.dumps(
                            {
                                "duplicate_count": dup.duplicate_count,
                                "total_values": dup.total_values,
                                "duplicate_ratio": dup.duplicate_ratio,
                            }
                        ),
                    )
                    session.add(finding)

            # New statistical tests (GRIMMER, variance ratio, terminal digit)
            stat_findings = await asyncio.to_thread(
                run_statistical_tests, full_text, config=self.config.analysis
            )
            for fd in stat_findings:
                finding = Finding(
                    paper_id=paper_id,
                    analysis_type=fd["analysis_type"],
                    severity=fd["severity"],
                    confidence=fd["confidence"],
                    title=fd["title"],
                    description=fd["description"],
                    evidence_json=json.dumps(fd.get("evidence", {})),
                )
                session.add(finding)

            # SPRITE test on mean/SD/N reports
            from snoopy.extraction.stats_extractor import extract_means_sds_and_ns

            mean_sd_reports = extract_means_sds_and_ns(full_text)
            for mr in mean_sd_reports:
                if mr.sd is not None and mr.n >= 2:
                    sprite_findings = await asyncio.to_thread(
                        run_sprite_analysis,
                        mr.mean,
                        mr.sd,
                        mr.n,
                        context=mr.context,
                    )
                    for fd in sprite_findings:
                        finding = Finding(
                            paper_id=paper_id,
                            analysis_type="sprite",
                            severity=fd["severity"],
                            confidence=fd["confidence"],
                            title=fd["title"],
                            description=fd["description"],
                            evidence_json=json.dumps(fd.get("evidence", {})),
                        )
                        session.add(finding)

            # Tortured phrase detection
            tp_findings = await asyncio.to_thread(
                run_tortured_phrases, full_text, config=self.config.analysis
            )
            for fd in tp_findings:
                finding = Finding(
                    paper_id=paper_id,
                    analysis_type="tortured_phrases",
                    severity=fd["severity"],
                    confidence=fd["confidence"],
                    title=fd["title"],
                    description=fd["description"],
                    evidence_json=json.dumps(fd.get("evidence", {})),
                )
                session.add(finding)

    def _build_method_weights(self) -> dict[str, float]:
        """Build a mapping of analysis_type -> weight from config."""
        cfg = self.config.analysis
        return {
            "clone_detection": cfg.weight_clone_detection,
            "phash": cfg.weight_phash,
            "pvalue_check": cfg.weight_pvalue_check,
            "grim": cfg.weight_grim,
            "noise_analysis": cfg.weight_noise,
            "ela": cfg.weight_ela,
            "benford": cfg.weight_benford,
            "duplicate_values": cfg.weight_duplicate_check,
            "duplicate_check": cfg.weight_duplicate_check,
            "llm_vision": cfg.weight_llm_vision,
            "llm_screening": cfg.weight_llm_vision,
            "dct_analysis": cfg.weight_dct_analysis,
            "jpeg_ghost": cfg.weight_jpeg_ghost,
            "fft_analysis": cfg.weight_fft_analysis,
            "grimmer": cfg.weight_grimmer,
            "terminal_digit": cfg.weight_terminal_digit,
            "distribution_fit": cfg.weight_distribution_fit,
            "variance_ratio": cfg.weight_variance_ratio,
            "tortured_phrases": cfg.weight_tortured_phrases,
            "sprite": cfg.weight_sprite,
            "temporal_patterns": cfg.weight_temporal_patterns,
        }

    async def _run_aggregate(self, paper_id: str) -> None:
        """Aggregate findings and compute overall risk."""
        async with get_async_session() as session:
            result = await session.execute(select(Finding).where(Finding.paper_id == paper_id))
            findings = result.scalars().all()

            finding_dicts = []
            for f in findings:
                finding_dicts.append(
                    {
                        "figure_id": f.figure_id,
                        "analysis_type": f.analysis_type,
                        "method": f.analysis_type,  # Normalize key for evidence module
                        "severity": f.severity,
                        "confidence": f.confidence,
                        "title": f.title,
                        "description": f.description,
                    }
                )

            method_weights = self._build_method_weights()
            evidence = aggregate_findings(finding_dicts, method_weights=method_weights)
            # Store aggregated result as paper metadata for report stage
            paper = await session.get(Paper, paper_id)
            if paper:
                paper.status = "analyzed"
                paper.risk_level = evidence.paper_risk
                paper.overall_confidence = evidence.overall_confidence
                paper.converging_evidence = evidence.converging_evidence

    async def _run_report(self, paper_id: str) -> None:
        """Generate the final proof report."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper:
                return

            findings_result = await session.execute(
                select(Finding).where(Finding.paper_id == paper_id)
            )
            findings = findings_result.scalars().all()
            figures_result = await session.execute(
                select(Figure).where(Figure.paper_id == paper_id)
            )
            figures = figures_result.scalars().all()

            finding_dicts = [
                {
                    "figure_id": f.figure_id,
                    "analysis_type": f.analysis_type,
                    "method": f.analysis_type,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "title": f.title,
                    "description": f.description,
                    "evidence_json": f.evidence_json,
                    "model_used": f.model_used,
                }
                for f in findings
            ]
            figure_dict: dict[str, dict[str, str | None]] = {
                str(fig.id): {
                    "figure_label": str(fig.figure_label) if fig.figure_label else None,
                    "caption": str(fig.caption) if fig.caption else None,
                    "image_type": str(fig.image_type) if fig.image_type else None,
                }
                for fig in figures
            }

            method_weights = self._build_method_weights()
            evidence = aggregate_findings(finding_dicts, method_weights=method_weights)

            # Generate LLM summary
            summary = f"Analysis of this paper identified {evidence.total_findings} findings."
            if evidence.paper_risk in ("critical", "high"):
                summary += (
                    f" {evidence.critical_count} critical issues were detected"
                    f"{' with converging evidence from multiple independent methods' if evidence.converging_evidence else ''}."
                )
            elif evidence.paper_risk == "clean":
                summary = "No integrity concerns were identified in this paper."

            # Try LLM summarization for non-clean papers
            if evidence.paper_risk != "clean" and finding_dicts:
                try:
                    from snoopy.llm.prompts import PROMPT_SUMMARIZE_EVIDENCE, SYSTEM_PROOF_WRITER

                    result = await self.llm_provider.analyze_text(
                        text=json.dumps(finding_dicts, indent=2),
                        prompt=PROMPT_SUMMARIZE_EVIDENCE.format(
                            findings_json=json.dumps(finding_dicts, indent=2)
                        ),
                        system=SYSTEM_PROOF_WRITER,
                    )
                    if result.get("parsed") and result["parsed"].get("summary"):
                        summary = result["parsed"]["summary"]
                    elif result.get("content"):
                        summary = result["content"][:500]
                except Exception as e:
                    logger.warning(f"LLM summarization failed: {e}")

            paper_dict = {
                "title": paper.title,
                "doi": paper.doi,
                "journal": paper.journal,
                "citation_count": paper.citation_count,
                "priority_score": paper.priority_score,
                "publication_year": paper.publication_year,
                "authors_json": paper.authors_json,
            }

            markdown = generate_markdown_report(
                paper=paper_dict,
                findings=finding_dicts,
                figures=figure_dict,
                summary=summary,
                overall_risk=evidence.paper_risk,
                overall_confidence=evidence.overall_confidence,
                converging_evidence=evidence.converging_evidence,
            )
            html = generate_html_report(
                paper=paper_dict,
                findings=finding_dicts,
                figures=figure_dict,
                summary=summary,
                overall_risk=evidence.paper_risk,
                overall_confidence=evidence.overall_confidence,
                converging_evidence=evidence.converging_evidence,
            )

            report = Report(
                paper_id=paper_id,
                overall_risk=evidence.paper_risk,
                overall_confidence=evidence.overall_confidence,
                summary=summary,
                report_markdown=markdown,
                report_html=html,
                num_findings=evidence.total_findings,
                num_critical=evidence.critical_count,
                converging_evidence=evidence.converging_evidence,
            )
            session.add(report)

            # Save to files
            reports_dir = Path(self.config.storage.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / f"{paper_id}.md").write_text(markdown)
            (reports_dir / f"{paper_id}.html").write_text(html)

    async def run_batch(self, limit: int = 100) -> list[str]:
        """Process top-priority pending papers."""
        async with get_async_session() as session:
            result = await session.execute(
                select(Paper)
                .where(Paper.status == "pending")
                .where(Paper.pdf_path.isnot(None))
                .order_by(Paper.priority_score.desc())
                .limit(limit)
            )
            papers = result.scalars().all()
            paper_ids = [str(p.id) for p in papers]

        results = []
        concurrency = 5
        # SQLite cannot handle concurrent writes; force serial processing
        db_url = self.config.storage.database_url
        if db_url.startswith("sqlite"):
            logger.warning(
                "SQLite detected — forcing batch concurrency to 1 to avoid write contention"
            )
            concurrency = 1
        sem = asyncio.Semaphore(concurrency)

        async def _process(pid: str) -> str:
            async with sem:
                try:
                    await self.process_paper(pid)
                    return pid
                except Exception as e:
                    logger.error(f"Failed to process paper {pid}: {e}")
                    return pid

        tasks = [_process(pid) for pid in paper_ids]
        results = await asyncio.gather(*tasks)
        return list(results)
