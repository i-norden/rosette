"""forge-world protocol implementations for rosette.

Implements the forge-world protocols directly in the rosette package,
enabling benchmark-modify-benchmark cycles via ``forge bench -m rosette.forge_world``.

RosetteDataset is seed-aware:
  - seed=None  → only fixed items (synthetic, RSIIL GitHub, retracted, survey,
                  retraction_watch, clean PDFs)
  - seed=N     → fixed items + ``sample_rsiil_images(sample_size, seed=N)``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forge_world.core.protocols import (
    AggregatedResult,
    Finding,
    LabeledItem,
    PassFailRule,
    Severity,
)

logger = logging.getLogger(__name__)

# Resolve project root and fixture directories
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"

# Default RSIIL Zenodo sample size per seed
_DEFAULT_SAMPLE_SIZE = 50


def _severity_from_str(s: str) -> Severity:
    """Convert rosette severity string to forge-world Severity enum."""
    mapping = {
        "clean": Severity.CLEAN,
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }
    return mapping.get(s, Severity.LOW)


def _finding_from_rosette_dict(d: dict[str, Any]) -> Finding:
    """Convert a rosette finding dict to a forge-world Finding."""
    return Finding(
        title=d.get("title", ""),
        method=d.get("method") or d.get("analysis_type", ""),
        severity=_severity_from_str(d.get("severity", "low")),
        confidence=float(d.get("confidence", 0.0)),
        description=d.get("description", ""),
        item_id=d.get("figure_id", ""),
        evidence=d.get("evidence", {}),
    )


class RosettePipeline:
    """Rosette's analysis pipeline as a forge-world Pipeline.

    Supports both image files (RSIIL, synthetic) and PDFs (retracted papers).
    """

    def __init__(self):
        from rosette.config import AnalysisConfig

        self._config = AnalysisConfig()

    def analyze(self, item: Any) -> list[Finding]:
        """Run rosette analysis on an item (image path or PDF path)."""
        path = Path(item) if isinstance(item, str) else item
        if not path.exists():
            logger.warning("Item not found: %s", path)
            return []

        if path.suffix.lower() == ".pdf":
            return self._analyze_pdf(path)
        return self._analyze_image(path)

    def _analyze_image(self, path: Path) -> list[Finding]:
        from rosette.analysis.run_analysis import run_image_forensics

        try:
            rosette_findings = run_image_forensics(
                str(path), figure_id=path.name, config=self._config,
            )
            return [_finding_from_rosette_dict(f) for f in rosette_findings]
        except Exception as exc:
            logger.warning("Image analysis failed for %s: %s", path.name, exc)
            return []

    def _analyze_pdf(self, path: Path) -> list[Finding]:
        """Run full PDF analysis (figures + text + tables + cross-ref)."""
        from rosette.demo.runner import _analyze_pdf as rosette_analyze_pdf

        figures_dir = _PROJECT_ROOT / "data" / "figures" / "forge_world"
        figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = rosette_analyze_pdf(path, figures_dir)
            return [_finding_from_rosette_dict(f) for f in result.get("findings", [])]
        except Exception as exc:
            logger.warning("PDF analysis failed for %s: %s", path.name, exc)
            return []

    def get_config(self) -> Any:
        return self._config

    def set_config(self, config: Any) -> None:
        self._config = config

    def get_config_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the analysis config."""
        from rosette.config import AnalysisConfig

        return AnalysisConfig.model_json_schema()


class RosetteAggregator:
    """Rosette's aggregate_findings as a forge-world Aggregator."""

    def __init__(self):
        from rosette.config import AnalysisConfig

        self._config = AnalysisConfig()

    def aggregate(self, findings: list[Finding]) -> AggregatedResult:
        from rosette.analysis.evidence import aggregate_findings

        rosette_dicts = []
        for f in findings:
            rosette_dicts.append({
                "title": f.title,
                "method": f.method,
                "analysis_type": f.method,
                "severity": f.severity.value,
                "confidence": f.confidence,
                "description": f.description,
                "figure_id": f.item_id,
                "evidence": f.evidence,
            })

        aggregated = aggregate_findings(
            rosette_dicts,
            method_weights=self._config.method_weights,
            single_method_max_severity=self._config.single_method_max_severity,
            single_method_max_confidence=self._config.single_method_max_confidence,
            convergence_confidence_threshold=self._config.convergence_confidence_threshold,
        )

        methods_flagged: set[str] = set()
        for fe in aggregated.figure_evidence:
            methods_flagged.update(fe.methods_flagged)

        return AggregatedResult(
            risk_level=_severity_from_str(aggregated.paper_risk),
            overall_confidence=aggregated.overall_confidence,
            converging_evidence=aggregated.converging_evidence,
            total_findings=aggregated.total_findings,
            methods_flagged=methods_flagged,
            findings=findings,
        )


class RosetteDataset:
    """Rosette's benchmark dataset as a forge-world LabeledDataset.

    Fixed items (always returned):
      - Synthetic forgeries (tests/fixtures/synthetic/)
      - RSIIL GitHub images (tests/fixtures/rsiil/)
      - Retracted PDFs (tests/fixtures/retracted/)
      - Survey paper (tests/fixtures/survey/)
      - Retraction Watch PDFs (tests/fixtures/retraction_watch/)
      - Clean control PDFs (tests/fixtures/clean/)

    Seed-sampled items (returned when seed is provided):
      - RSIIL Zenodo images sampled via sample_rsiil_images(sample_size, seed)
    """

    def __init__(self):
        self._fixed_items: list[LabeledItem] | None = None

    def _load_fixed(self) -> list[LabeledItem]:
        """Load fixed items (deterministic, same every run)."""
        if self._fixed_items is not None:
            return self._fixed_items

        items: list[LabeledItem] = []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        # Synthetic forgeries
        synth_dir = _FIXTURES_DIR / "synthetic"
        if synth_dir.exists():
            for img in sorted(synth_dir.iterdir()):
                if img.suffix.lower() in exts:
                    items.append(LabeledItem(
                        id=img.name,
                        category="synthetic",
                        expected_label="findings",
                        data=img,
                    ))

        # RSIIL GitHub images (small fixed set)
        rsiil_dir = _FIXTURES_DIR / "rsiil"
        if rsiil_dir.exists():
            try:
                from rosette.demo.fixtures import (
                    RSIIL_CLEAN_IMAGES,
                    _is_ground_truth,
                    _is_pristine_ref,
                )

                rsiil_clean_names = {name for _, name in RSIIL_CLEAN_IMAGES}
                for img in sorted(rsiil_dir.iterdir()):
                    if img.suffix.lower() not in exts:
                        continue
                    if _is_ground_truth(img):
                        continue
                    if img.name in rsiil_clean_names or _is_pristine_ref(img):
                        items.append(LabeledItem(
                            id=img.name,
                            category="rsiil_clean",
                            expected_label="clean",
                            data=img,
                        ))
                    else:
                        items.append(LabeledItem(
                            id=img.name,
                            category="rsiil",
                            expected_label="findings",
                            data=img,
                        ))
            except ImportError:
                logger.warning("Could not import rosette fixtures module")

        # Retracted papers
        retracted_dir = _FIXTURES_DIR / "retracted"
        if retracted_dir.exists():
            for pdf in sorted(retracted_dir.iterdir()):
                if pdf.suffix.lower() == ".pdf":
                    items.append(LabeledItem(
                        id=pdf.name,
                        category="retracted",
                        expected_label="findings",
                        data=pdf,
                    ))

        # Survey paper
        survey_dir = _FIXTURES_DIR / "survey"
        if survey_dir.exists():
            for pdf in sorted(survey_dir.iterdir()):
                if pdf.suffix.lower() == ".pdf":
                    items.append(LabeledItem(
                        id=pdf.name,
                        category="survey",
                        expected_label="informational",
                        data=pdf,
                    ))

        # Retraction Watch papers
        rw_dir = _FIXTURES_DIR / "retraction_watch"
        if rw_dir.exists():
            for pdf in sorted(rw_dir.iterdir()):
                if pdf.suffix.lower() == ".pdf":
                    items.append(LabeledItem(
                        id=pdf.name,
                        category="retraction_watch",
                        expected_label="findings",
                        data=pdf,
                    ))

        # Clean control papers
        clean_dir = _FIXTURES_DIR / "clean"
        if clean_dir.exists():
            for pdf in sorted(clean_dir.iterdir()):
                if pdf.suffix.lower() == ".pdf":
                    items.append(LabeledItem(
                        id=pdf.name,
                        category="clean",
                        expected_label="clean",
                        data=pdf,
                    ))

        self._fixed_items = items
        return items

    def items(
        self, seed: int | None = None, sample_size: int | None = None
    ) -> list[LabeledItem]:
        """Return benchmark items.

        seed=None: fixed items only.
        seed=N: fixed items + RSIIL Zenodo samples for that seed.
        sample_size: how many pristine + how many tampered from Zenodo per seed.
        """
        result = list(self._load_fixed())

        if seed is not None:
            size = sample_size if sample_size is not None else _DEFAULT_SAMPLE_SIZE
            result.extend(self._sample_zenodo(seed, size))

        return result

    def _sample_zenodo(self, seed: int, sample_size: int) -> list[LabeledItem]:
        """Sample RSIIL Zenodo images for a given seed."""
        try:
            from rosette.demo.fixtures import sample_rsiil_images
        except ImportError:
            logger.warning("Cannot import sample_rsiil_images — Zenodo data unavailable")
            return []

        pristine, tampered = sample_rsiil_images(sample_size=sample_size, seed=seed)
        items: list[LabeledItem] = []

        for path in pristine:
            items.append(LabeledItem(
                id=f"zenodo_pristine_{path.name}",
                category="rsiil_zenodo_clean",
                expected_label="clean",
                data=path,
                metadata={"source": "zenodo", "seed": seed},
            ))

        for path in tampered:
            items.append(LabeledItem(
                id=f"zenodo_tampered_{path.name}",
                category="rsiil_zenodo",
                expected_label="findings",
                data=path,
                metadata={"source": "zenodo", "seed": seed},
            ))

        return items

    def categories(self) -> list[str]:
        cats = {item.category for item in self._load_fixed()}
        cats.update(["rsiil_zenodo", "rsiil_zenodo_clean"])
        return sorted(cats)

    def tiers(self) -> dict[str, list[str]]:
        """Return tier definitions for tiered evaluation.

        Smoke: ~15 items, ~30s.  Standard: ~68 items, ~3 min.
        Full: all categories including Zenodo, ~168 items, ~25 min.
        """
        return {
            "smoke": ["synthetic"],
            "standard": [
                "synthetic", "rsiil", "rsiil_clean", "retracted",
                "survey", "retraction_watch", "clean",
            ],
            "full": [
                "synthetic", "rsiil", "rsiil_clean", "retracted",
                "survey", "retraction_watch", "clean",
                "rsiil_zenodo", "rsiil_zenodo_clean",
            ],
        }


class RosetteRules:
    """Pass/fail rules matching rosette's demo runner logic.

    - Items expected to have findings: pass if risk >= MEDIUM
    - Clean items: pass if risk <= MEDIUM (fail only on HIGH/CRITICAL)
    - Informational: always pass
    """

    def get_rule(self, expected_label: str) -> PassFailRule:
        if expected_label == "findings":
            return PassFailRule(
                expected_label="findings",
                min_risk_for_pass=Severity.MEDIUM,
            )
        elif expected_label == "clean":
            return PassFailRule(
                expected_label="clean",
                max_risk_for_pass=Severity.MEDIUM,
            )
        else:
            return PassFailRule(expected_label=expected_label)


# --- Module-level factory functions (used by forge CLI module loading) ---


def create_pipeline() -> RosettePipeline:
    return RosettePipeline()


def create_aggregator() -> RosetteAggregator:
    return RosetteAggregator()


def create_dataset() -> RosetteDataset:
    return RosetteDataset()


def create_rules() -> RosetteRules:
    return RosetteRules()
