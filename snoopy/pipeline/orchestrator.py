"""Main pipeline orchestrator with resumability and bounded concurrency."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from snoopy.analysis.evidence import aggregate_findings
from snoopy.analysis.image_forensics import clone_detection, error_level_analysis, noise_analysis
from snoopy.analysis.llm_vision import analyze_figure_detailed, classify_figure, screen_figure
from snoopy.analysis.statistical import benford_test, duplicate_value_check, grim_test, pvalue_check
from snoopy.config import SnoopyConfig
from snoopy.db.models import Figure, Finding, Paper, ProcessingLog, Report
from snoopy.db.session import get_session
from snoopy.extraction.figure_extractor import extract_figures
from snoopy.extraction.pdf_parser import download_pdf, extract_text
from snoopy.extraction.stats_extractor import extract_means_and_ns, extract_test_statistics
from snoopy.extraction.table_extractor import extract_tables
from snoopy.llm.claude import ClaudeProvider
from snoopy.pipeline.stages import PIPELINE_STAGES, StageResult, StageStatus
from snoopy.reporting.proof import generate_html_report, generate_markdown_report

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages the full analysis pipeline with resumability."""

    def __init__(self, config: SnoopyConfig):
        self.config = config
        self.llm_provider = ClaudeProvider(
            default_model=config.llm.model_analyze,
        )
        self.semaphore = asyncio.Semaphore(config.llm.max_concurrent_requests)

    def _log_stage(self, paper_id: str, stage: str, status: str, details: str = "") -> None:
        with get_session() as session:
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

    def _get_last_completed_stage(self, paper_id: str) -> str | None:
        with get_session() as session:
            result = session.execute(
                select(ProcessingLog)
                .where(ProcessingLog.paper_id == paper_id)
                .where(ProcessingLog.status == "completed")
                .order_by(ProcessingLog.completed_at.desc())
            )
            log = result.scalars().first()
            return log.stage if log else None

    def _update_paper_status(self, paper_id: str, status: str, error: str | None = None) -> None:
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if paper:
                paper.status = status
                paper.error_message = error
                paper.updated_at = datetime.now(timezone.utc)

    async def process_paper(self, paper_id: str) -> None:
        """Process a single paper through all remaining stages."""
        last_completed = self._get_last_completed_stage(paper_id)

        if last_completed:
            try:
                start_idx = PIPELINE_STAGES.index(last_completed) + 1
            except ValueError:
                start_idx = 0
        else:
            start_idx = 0

        # Skip discovery/prioritize for individual paper processing
        paper_stages = [s for s in PIPELINE_STAGES[start_idx:] if s not in ("discover", "prioritize")]

        self._update_paper_status(paper_id, "analyzing")

        for stage in paper_stages:
            self._log_stage(paper_id, stage, "started")
            try:
                handler = getattr(self, f"_run_{stage}", None)
                if handler:
                    await handler(paper_id)
                self._log_stage(paper_id, stage, "completed")
            except Exception as e:
                logger.error(f"Stage {stage} failed for paper {paper_id}: {e}")
                self._log_stage(paper_id, stage, "failed", str(e))
                self._update_paper_status(paper_id, "error", str(e))
                raise

        self._update_paper_status(paper_id, "complete")

    async def _run_download(self, paper_id: str) -> None:
        """Download the PDF for a paper."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper or paper.pdf_path:
                return

            if not paper.doi:
                logger.warning(f"Paper {paper_id} has no DOI, skipping download")
                return

            from snoopy.discovery.unpaywall import get_pdf_url

            pdf_url = await get_pdf_url(paper.doi)
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
        """Extract text from the paper's PDF."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return
            # Text extraction is used by downstream stages via the PDF path
            pages = extract_text(paper.pdf_path)
            logger.info(f"Extracted {len(pages)} pages from {paper_id}")

    async def _run_extract_figures(self, paper_id: str) -> None:
        """Extract figures from the paper's PDF."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return

            fig_dir = Path(self.config.storage.figures_dir) / paper_id
            fig_dir.mkdir(parents=True, exist_ok=True)

            figures = extract_figures(paper.pdf_path, str(fig_dir))
            for fig_info in figures:
                figure = Figure(
                    paper_id=paper_id,
                    page_number=fig_info.page_number,
                    figure_label=fig_info.figure_label,
                    caption=fig_info.caption,
                    image_path=fig_info.image_path,
                    image_sha256=fig_info.image_sha256,
                    width=fig_info.width,
                    height=fig_info.height,
                )
                session.add(figure)

    async def _run_extract_stats(self, paper_id: str) -> None:
        """Extract statistical values from paper text."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return
            # Stats are extracted and used in analyze_stats stage
            pages = extract_text(paper.pdf_path)
            full_text = "\n".join(p.text for p in pages)
            means = extract_means_and_ns(full_text)
            stats = extract_test_statistics(full_text)
            logger.info(
                f"Paper {paper_id}: found {len(means)} mean reports, {len(stats)} test statistics"
            )

    async def _run_classify_figures(self, paper_id: str) -> None:
        """Classify figure types using LLM."""
        with get_session() as session:
            figures = session.execute(
                select(Figure).where(Figure.paper_id == paper_id)
            ).scalars().all()

            for figure in figures:
                if figure.image_path and Path(figure.image_path).exists():
                    async with self.semaphore:
                        fig_type = await classify_figure(figure.image_path, self.llm_provider)
                        figure.image_type = fig_type

    async def _run_analyze_images(self, paper_id: str) -> None:
        """Run image forensics on all figures."""
        with get_session() as session:
            figures = session.execute(
                select(Figure).where(Figure.paper_id == paper_id)
            ).scalars().all()

            for figure in figures:
                if not figure.image_path or not Path(figure.image_path).exists():
                    continue

                # ELA
                ela = error_level_analysis(
                    figure.image_path, quality=self.config.analysis.ela_quality
                )
                if ela.suspicious:
                    finding = Finding(
                        paper_id=paper_id,
                        figure_id=figure.id,
                        analysis_type="ela",
                        severity="medium",
                        confidence=min(ela.max_difference / 100, 1.0),
                        title=f"ELA anomaly in {figure.figure_label or 'figure'}",
                        description=(
                            f"Error Level Analysis detected inconsistent compression levels. "
                            f"Max difference: {ela.max_difference:.1f}, "
                            f"Mean: {ela.mean_difference:.1f}, Std: {ela.std_difference:.1f}"
                        ),
                        evidence_json=json.dumps({
                            "max_difference": ela.max_difference,
                            "mean_difference": ela.mean_difference,
                            "ela_image_path": ela.ela_image_path,
                        }),
                    )
                    session.add(finding)

                # Clone detection
                clone = clone_detection(
                    figure.image_path, min_matches=self.config.analysis.clone_min_matches
                )
                if clone.suspicious:
                    finding = Finding(
                        paper_id=paper_id,
                        figure_id=figure.id,
                        analysis_type="clone_detection",
                        severity="high",
                        confidence=min(clone.num_matches / 50, 1.0),
                        title=f"Potential clone region in {figure.figure_label or 'figure'}",
                        description=(
                            f"Copy-move detection found {clone.num_matches} geometrically "
                            f"consistent keypoint matches (inlier ratio: {clone.inlier_ratio:.2f})"
                        ),
                        evidence_json=json.dumps({
                            "num_matches": clone.num_matches,
                            "inlier_ratio": clone.inlier_ratio,
                            "clusters": clone.match_clusters,
                        }),
                    )
                    session.add(finding)

                # Noise analysis
                noise = noise_analysis(
                    figure.image_path, block_size=self.config.analysis.noise_block_size
                )
                if noise.suspicious:
                    finding = Finding(
                        paper_id=paper_id,
                        figure_id=figure.id,
                        analysis_type="noise_analysis",
                        severity="medium",
                        confidence=min(noise.max_ratio / 10, 1.0),
                        title=f"Noise inconsistency in {figure.figure_label or 'figure'}",
                        description=(
                            f"Noise analysis detected inconsistent noise levels across image "
                            f"regions. Max noise ratio: {noise.max_ratio:.1f}x"
                        ),
                        evidence_json=json.dumps({
                            "max_ratio": noise.max_ratio,
                            "mean_noise": noise.mean_noise,
                        }),
                    )
                    session.add(finding)

                # LLM Vision - Stage 1 screening
                async with self.semaphore:
                    screening = await screen_figure(
                        figure.image_path, self.llm_provider, caption=figure.caption or ""
                    )

                if screening.suspicious and screening.confidence >= self.config.analysis.llm_screening_confidence_threshold:
                    # Stage 2 detailed analysis
                    async with self.semaphore:
                        detailed = await analyze_figure_detailed(
                            figure.image_path,
                            self.llm_provider,
                            caption=figure.caption or "",
                            figure_type=figure.image_type or "",
                        )

                    for vf in detailed.findings:
                        severity = "high" if vf.confidence > 0.7 else "medium" if vf.confidence > 0.4 else "low"
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

    async def _run_analyze_stats(self, paper_id: str) -> None:
        """Run statistical integrity checks."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper or not paper.pdf_path:
                return

            pages = extract_text(paper.pdf_path)
            full_text = "\n".join(p.text for p in pages)

            # GRIM test
            means = extract_means_and_ns(full_text)
            for mr in means:
                result = grim_test(mr.mean, mr.n)
                if not result.consistent:
                    finding = Finding(
                        paper_id=paper_id,
                        analysis_type="grim",
                        severity="medium",
                        confidence=0.8,
                        title=f"GRIM inconsistency: M={mr.mean}, N={mr.n}",
                        description=(
                            f"Mean of {mr.mean} with N={mr.n} is not mathematically possible. "
                            f"Product {mr.mean}*{mr.n} = {result.product:.2f}, "
                            f"nearest integer = {result.nearest_integer}, "
                            f"difference = {result.difference:.4f}. "
                            f"Context: {mr.context[:200]}"
                        ),
                        evidence_json=json.dumps({
                            "mean": mr.mean,
                            "n": mr.n,
                            "product": result.product,
                        }),
                    )
                    session.add(finding)

            # P-value check
            test_stats = extract_test_statistics(full_text)
            for ts in test_stats:
                if ts.p_value is not None:
                    result = pvalue_check(ts.test_type, ts.statistic, ts.df, ts.p_value)
                    if not result.consistent:
                        finding = Finding(
                            paper_id=paper_id,
                            analysis_type="pvalue_check",
                            severity="high" if result.significance_changed else "medium",
                            confidence=0.85,
                            title=f"P-value inconsistency: {ts.test_type} test",
                            description=(
                                f"Reported p={result.reported_p}, computed p={result.computed_p:.6f} "
                                f"(difference: {result.difference:.6f}). "
                                f"{'Significance conclusion changed!' if result.significance_changed else ''} "
                                f"Context: {ts.context[:200]}"
                            ),
                            evidence_json=json.dumps({
                                "test_type": ts.test_type,
                                "statistic": ts.statistic,
                                "df": ts.df,
                                "reported_p": result.reported_p,
                                "computed_p": result.computed_p,
                            }),
                        )
                        session.add(finding)

            # Benford's law on tables
            tables = extract_tables(paper.pdf_path)
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
                        evidence_json=json.dumps({
                            "chi_squared": bf.chi_squared,
                            "p_value": bf.p_value,
                            "observed": bf.observed_distribution,
                            "expected": bf.expected_distribution,
                        }),
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
                        evidence_json=json.dumps({
                            "duplicate_count": dup.duplicate_count,
                            "total_values": dup.total_values,
                            "duplicate_ratio": dup.duplicate_ratio,
                        }),
                    )
                    session.add(finding)

    async def _run_aggregate(self, paper_id: str) -> None:
        """Aggregate findings and compute overall risk."""
        with get_session() as session:
            findings = session.execute(
                select(Finding).where(Finding.paper_id == paper_id)
            ).scalars().all()

            finding_dicts = []
            for f in findings:
                finding_dicts.append({
                    "figure_id": f.figure_id,
                    "analysis_type": f.analysis_type,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "title": f.title,
                    "description": f.description,
                })

            evidence = aggregate_findings(finding_dicts)
            # Store aggregated result as paper metadata for report stage
            paper = session.get(Paper, paper_id)
            if paper:
                paper.status = "analyzed"

    async def _run_report(self, paper_id: str) -> None:
        """Generate the final proof report."""
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            if not paper:
                return

            findings = session.execute(
                select(Finding).where(Finding.paper_id == paper_id)
            ).scalars().all()
            figures = session.execute(
                select(Figure).where(Figure.paper_id == paper_id)
            ).scalars().all()

            finding_dicts = [
                {
                    "figure_id": f.figure_id,
                    "analysis_type": f.analysis_type,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "title": f.title,
                    "description": f.description,
                    "evidence_json": f.evidence_json,
                    "model_used": f.model_used,
                }
                for f in findings
            ]
            figure_dict = {
                fig.id: {
                    "figure_label": fig.figure_label,
                    "caption": fig.caption,
                    "image_type": fig.image_type,
                }
                for fig in figures
            }

            evidence = aggregate_findings(finding_dicts)

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
        with get_session() as session:
            papers = session.execute(
                select(Paper)
                .where(Paper.status == "pending")
                .where(Paper.pdf_path.isnot(None))
                .order_by(Paper.priority_score.desc())
                .limit(limit)
            ).scalars().all()
            paper_ids = [p.id for p in papers]

        results = []
        sem = asyncio.Semaphore(5)

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
