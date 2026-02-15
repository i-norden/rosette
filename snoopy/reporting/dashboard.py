"""Generate an HTML dashboard for demo results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import snoopy

TEMPLATE_DIR = Path(__file__).parent / "templates"

CATEGORY_META: dict[str, dict[str, str]] = {
    "synthetic": {
        "display_name": "Synthetic Forgeries",
        "description": "Programmatically generated manipulated images with known copy-move, splicing, and ELA artifacts.",
        "expected": "findings",
    },
    "rsiil": {
        "display_name": "RSIIL Benchmark",
        "description": "Images from the Realistic Synthetic Image Integrity Loss benchmark dataset.",
        "expected": "findings",
    },
    "retracted": {
        "display_name": "Retracted Papers",
        "description": "Figures extracted from papers that were retracted due to image manipulation concerns.",
        "expected": "findings",
    },
    "retraction_watch": {
        "display_name": "Retraction Watch",
        "description": "Papers flagged by Retraction Watch for integrity issues.",
        "expected": "findings",
    },
    "survey": {
        "display_name": "Survey / Informational",
        "description": "Bik survey paper — analyzed for informational purposes, no pass/fail expectation.",
        "expected": "informational",
    },
    "clean": {
        "display_name": "Clean Controls",
        "description": "Unmanipulated papers used to measure false positive rate.",
        "expected": "clean",
    },
    "rsiil_clean": {
        "display_name": "RSIIL Clean Controls",
        "description": "Pristine images from the RSIIL dataset used to measure false positive rate.",
        "expected": "clean",
    },
}


def _get_jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)


def _methods_for_result(result: dict) -> list[str]:
    """Extract unique analysis method names from a result's findings."""
    # Prefer pre-computed methods_used if available
    if result.get("methods_used"):
        return result["methods_used"]
    methods: list[str] = []
    seen: set[str] = set()
    for f in result.get("findings", []):
        method = f.get("method") or f.get("analysis_type", "")
        if method and method not in seen:
            methods.append(method)
            seen.add(method)
    return methods


def _report_filename(name: str) -> str:
    """Compute the expected individual report filename for a result."""
    return Path(name).stem + "_report.html"


def _build_category(cat_key: str, results: list[dict]) -> dict:
    """Build a category context dict for template rendering."""
    meta = CATEGORY_META.get(
        cat_key,
        {
            "display_name": cat_key.replace("_", " ").title(),
            "description": "",
            "expected": "findings",
        },
    )

    total = len(results)
    expected = meta["expected"]

    # Enrich each result with methods list and report link
    enriched: list[dict] = []
    for r in results:
        enriched.append(
            {
                **r,
                "methods": _methods_for_result(r),
                "report_link": "./" + _report_filename(r["name"])
                if r.get("findings_count", 0) > 0
                else "",
            }
        )

    # Per-method breakdown across all results in this category
    method_counts: dict[str, int] = {}
    for r in results:
        for f in r.get("findings", []):
            m = f.get("method") or f.get("analysis_type", "")
            if m:
                method_counts[m] = method_counts.get(m, 0) + 1

    # Convergence and confidence aggregates
    converging_count = sum(1 for r in results if r.get("converging_evidence"))
    confidences = [r.get("overall_confidence", 0.0) for r in results if r.get("overall_confidence")]
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

    cat: dict = {
        "key": cat_key,
        "display_name": meta["display_name"],
        "description": meta["description"],
        "expected": expected,
        "total": total,
        "results": enriched,
        "method_counts": method_counts,
        "converging_count": converging_count,
        "avg_confidence": avg_confidence,
    }

    if expected == "findings":
        detected = sum(1 for r in results if r.get("pass_fail"))
        missed = total - detected
        accuracy = (detected / total * 100) if total > 0 else 0.0
        cat.update({"detected": detected, "missed": missed, "accuracy": accuracy})
    elif expected == "clean":
        false_positives = sum(1 for r in results if not r.get("pass_fail"))
        fp_rate = (false_positives / total * 100) if total > 0 else 0.0
        cat.update({"false_positives": false_positives, "fp_rate": fp_rate})
    else:
        # Informational — just count total findings
        cat["findings_total"] = sum(r.get("findings_count", 0) for r in results)

    return cat


def generate_dashboard(demo_results: list[dict], report_dir: Path) -> Path:
    """Generate an HTML dashboard from demo results.

    Args:
        demo_results: List of result dicts from run_demo(), each containing
            name, category, expected, actual_risk, findings_count, pass_fail,
            and findings.
        report_dir: Directory to write index.html into.

    Returns:
        Path to the generated index.html file.
    """
    # Group results by category, preserving order of first appearance
    groups: dict[str, list[dict]] = {}
    for r in demo_results:
        cat = r.get("category", "unknown")
        groups.setdefault(cat, []).append(r)

    # Build per-category template context
    categories = [_build_category(key, results) for key, results in groups.items()]

    # Compute overall stats
    total_tested = len(demo_results)
    total_passed = sum(1 for r in demo_results if r.get("pass_fail"))
    pass_rate = (total_passed / total_tested * 100) if total_tested > 0 else 0.0

    # True positive rate: across categories that expect findings
    tp_results = [r for r in demo_results if r.get("expected") == "findings"]
    tp_detected = sum(1 for r in tp_results if r.get("pass_fail"))
    tp_rate = (tp_detected / len(tp_results) * 100) if tp_results else 0.0

    # False positive rate: across clean-expected categories
    fp_results = [r for r in demo_results if r.get("expected") == "clean"]
    fp_count = sum(1 for r in fp_results if not r.get("pass_fail"))
    fp_rate = (fp_count / len(fp_results) * 100) if fp_results else 0.0

    env = _get_jinja_env()
    template = env.get_template("dashboard.html.j2")

    html = template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        version=snoopy.__version__,
        total_tested=total_tested,
        pass_rate=pass_rate,
        tp_rate=tp_rate,
        fp_rate=fp_rate,
        categories=categories,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "index.html"
    out_path.write_text(html)
    return out_path
