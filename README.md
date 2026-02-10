# Sniffer

LLM-powered academic integrity analyzer. Detects image manipulation, statistical anomalies, and figure duplication in scientific papers using computer vision, statistical tests, and Claude-based analysis.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the demo (downloads test fixtures, runs forensics, prints results):

```bash
sniffer demo
```

## Architecture

```
sniffer/
├── analysis/          # Detection methods
│   ├── image_forensics.py   # ELA, clone detection, noise analysis
│   ├── statistical.py       # GRIM test, Benford's law, p-value checks
│   ├── llm_vision.py        # Claude-based figure screening
│   ├── cross_reference.py   # Perceptual hashing for duplicate figures
│   └── evidence.py          # Multi-method evidence aggregation
├── discovery/         # Paper sourcing (OpenAlex, PubMed, Semantic Scholar, CrossRef)
├── extraction/        # PDF text, figure, table, and statistics extraction
├── llm/               # Claude API provider with retry and batch support
├── pipeline/          # Multi-stage orchestrator with resumability
├── reporting/         # Markdown/HTML report generation + Rich terminal output
├── db/                # SQLAlchemy models (papers, figures, findings, reports)
├── cli.py             # Click CLI entry point
└── config.py          # Pydantic config with YAML + env var support
```

## CLI Commands

```
sniffer discover    Search academic APIs for papers to analyze
sniffer analyze     Analyze a single paper by DOI or local PDF
sniffer batch       Process top-priority pending papers
sniffer report      Display a paper's analysis report
sniffer status      Show pipeline status and queue depth
sniffer demo        Run demo with test fixtures and pretty output
sniffer config      Show current configuration
```

### sniffer demo

Runs the full demo pipeline: downloads test fixtures, runs image forensics on all test cases, prints Rich-formatted results, and generates HTML reports.

```bash
sniffer demo                        # Full demo
sniffer demo --download-only        # Only download fixtures
sniffer demo --output-dir ./out     # Custom report output directory
```

### sniffer analyze

```bash
sniffer analyze --doi 10.1234/example
sniffer analyze --pdf path/to/paper.pdf
```

## Analysis Methods

| Method | Module | What It Detects |
|--------|--------|-----------------|
| Error Level Analysis | `image_forensics.py` | JPEG re-compression artifacts from editing |
| Clone Detection | `image_forensics.py` | Copy-move forgery via ORB feature matching |
| Noise Analysis | `image_forensics.py` | Noise inconsistencies from splicing |
| GRIM Test | `statistical.py` | Impossible means given sample size |
| Benford's Law | `statistical.py` | Unnatural leading digit distributions |
| P-Value Check | `statistical.py` | Reported p-values inconsistent with test statistics |
| LLM Vision | `llm_vision.py` | Visual anomalies via Claude (requires API key) |
| Cross-Reference | `cross_reference.py` | Duplicate figures across papers via perceptual hashing |

Findings from multiple methods are aggregated in `evidence.py`. Converging evidence (2+ methods flagging the same figure) boosts confidence and severity.

## Configuration

Copy and edit `config/default.yaml`, or override with environment variables:

```bash
export SNIFFER__LLM__PROVIDER=claude
export SNIFFER__STORAGE__DATABASE_URL=sqlite:///my.db
export ANTHROPIC_API_KEY=sk-ant-...
```

Key settings:

```yaml
analysis:
  ela_quality: 95              # JPEG quality for ELA re-compression
  clone_min_matches: 10        # Minimum ORB matches to flag cloning
  noise_block_size: 64         # Block size for noise analysis
  convergence_required: true   # Require multi-method agreement

priority:
  min_citations: 50            # Minimum citations to queue a paper
  min_priority_score: 40       # Minimum priority score to process
```

## Testing

```bash
# Unit tests (fast, no network)
pytest tests/ -v -m "not integration"

# Integration tests (requires downloaded fixtures)
python scripts/download_fixtures.py
pytest tests/test_integration/ -v -m integration

# All tests
pytest tests/ -v
```

### Test Fixtures

`scripts/download_fixtures.py` downloads/generates six categories of test data into `tests/fixtures/`:

| Category | Source | Purpose |
|----------|--------|---------|
| `rsiil/` | RSIIL benchmark (GitHub) | Known manipulated images with ground truth |
| `synthetic/` | Generated locally | Copy-move, spliced, and retouched forgeries |
| `retracted/` | PMC Open Access | Retracted papers from Bik MCB study |
| `survey/` | PMC Open Access | Bik et al. 2016 survey on image duplication |
| `retraction_watch/` | PMC Open Access | Papers flagged for image manipulation |
| `clean/` | PMC Open Access | High-impact, never-questioned papers (false positive control) |

The script is idempotent -- it skips already-downloaded files.

## Project Layout

```
sniffer/             # Main package (installed via pip)
scripts/             # Standalone scripts (demo, fixture download)
tests/               # Test suite
  test_analysis/     #   Analysis method tests
  test_discovery/    #   Priority scoring tests
  test_extraction/   #   Stats extraction tests
  test_llm/          #   Prompt validation tests
  test_reporting/    #   Pretty reporter tests
  test_scripts/      #   Script helper function tests
  test_integration/  #   Integration tests (requires fixtures)
  fixtures/          #   Downloaded test data (not checked in)
config/              # YAML configuration
data/                # Runtime data (PDFs, figures, reports, DB)
```
