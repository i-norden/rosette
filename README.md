# Rosette

Academic integrity analyzer. Detects image manipulation, statistical anomalies, and figure duplication in scientific papers using computer vision, statistical tests, and (optionally) LLM-based analysis.

The analysis pipeline was iteratively refined using an agentic evolutionary metaprogramming harness and a dataset of known manipulated papers, retracted papers, and clean controls.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the demo (downloads test fixtures, runs forensics, prints results):

```bash
rosette demo
```

## Architecture

```
rosette/
├── analysis/          # Detection methods
│   ├── image_forensics.py   # ELA, clone detection, noise, DCT, JPEG ghost, FFT
│   ├── statistical.py       # GRIM test, Benford's law, p-value checks
│   ├── llm_vision.py        # Claude-based figure screening
│   ├── cross_reference.py   # Perceptual hashing for duplicate figures
│   ├── evidence.py          # Multi-method evidence aggregation
│   └── author_network.py    # Co-author graph analysis (Louvain clustering)
├── api/               # FastAPI REST server
├── calibration/       # Benchmark and accuracy metrics
├── campaign/          # Multi-paper investigation system
│   ├── orchestrator.py      # Campaign execution engine (3 modes)
│   ├── triage.py            # Two-tier funnel (auto → LLM)
│   ├── expander.py          # Co-author network expansion
│   ├── hash_scanner.py      # Cross-paper image hash matching
│   └── dashboard.py         # Campaign HTML dashboard
├── discovery/         # Paper sourcing + external signals
│   ├── openalex.py          # OpenAlex API
│   ├── pubmed.py / crossref.py / semantic_scholar.py / unpaywall.py
│   ├── retraction_watch.py  # Retraction status lookups
│   └── pubpeer.py           # PubPeer comment checks
├── extraction/        # PDF text, figure, table, stats extraction
├── llm/               # Claude API provider with retry and batch support
├── pipeline/          # Multi-stage orchestrator with resumability
├── reporting/         # Reports, dashboards, Rich terminal output
├── db/                # SQLAlchemy models + Alembic migrations
├── cli.py             # Click CLI entry point
├── cli_campaign.py    # Campaign CLI commands
├── config.py          # Pydantic config with YAML + env var support
└── validation.py      # Input validation utilities
```

## CLI Commands

```
rosette discover    Search academic APIs for papers to analyze
rosette analyze     Analyze a single paper by DOI or local PDF
rosette batch       Process top-priority pending papers
rosette report      Display a paper's analysis report
rosette status      Show pipeline status and queue depth
rosette demo        Run demo with test fixtures and pretty output
rosette serve       Start the REST API server
rosette config      Show current configuration
rosette db          Database migration commands (upgrade/downgrade/current)

rosette campaign create     Create a new investigation campaign
rosette campaign run        Start or resume a campaign
rosette campaign pause      Pause a running campaign
rosette campaign status     Show campaign progress
rosette campaign list       List all campaigns
rosette campaign dashboard  Generate HTML dashboard
rosette campaign export     Export evidence packages
```

### rosette demo

End-to-end showcase of the forensic analysis pipeline. Downloads ~3 GB of test fixtures across six categories, runs multi-method analysis on each, and generates an interactive HTML dashboard that opens in your browser.

**Fixture categories:**

| Category | Count | Source | Purpose |
|----------|-------|--------|---------|
| Synthetic forgeries | 10 images | Generated locally | Copy-move, splicing, retouching |
| RSIIL benchmark | 3 images | RSIIL GitHub | Known manipulations with ground truth |
| Retracted papers | 10 PDFs | PMC Open Access | Papers retracted for image issues |
| Bik survey | 1 PDF | PMC Open Access | Reference study on image duplication |
| Retraction Watch | 2 PDFs | PMC Open Access | Papers flagged for manipulation |
| Clean controls | 21 PDFs | PMC Open Access | Landmark papers (false positive control) |

**Analysis applied per item:**
- **Images:** ELA, clone detection (ORB + SIFT), block-based clone detection, noise analysis, DCT analysis, JPEG ghost detection, FFT frequency analysis, metadata forensics, perceptual hashing
- **PDFs:** Figure extraction + all image forensics, text extraction + GRIM/GRIMMER/Benford/p-value/terminal digit/variance ratio/SPRITE tests + tortured phrase detection, table extraction + duplicate value checks, intra-paper cross-reference
- **LLM (opt-in):** Claude vision screening and detailed analysis

```bash
rosette demo                        # Full demo (no LLM)
rosette demo --download-only        # Only download fixtures
rosette demo --use-llm              # Enable LLM analysis (needs ANTHROPIC_API_KEY)
rosette demo --output-dir ./out     # Custom report output directory
rosette demo --download-rsiil       # Also download full RSIIL dataset (~57 GB)
```

### rosette analyze

```bash
rosette analyze --doi 10.1234/example
rosette analyze --pdf path/to/paper.pdf
```

### rosette campaign

Large-scale investigation across multiple papers. Three investigation modes:

```bash
# Network expansion: follow co-author networks from suspicious papers
rosette campaign create --mode network_expansion --name "Dana-Farber follow-up" \
  --seed-doi 10.1234/suspicious1 --seed-doi 10.1234/suspicious2 \
  --max-depth 2 --max-papers 500 --llm-budget 50
rosette campaign run <campaign-id>

# Domain scan: systematic sweep of a research field
rosette campaign create --mode domain_scan --name "Stem cell survey" \
  --field "stem cell" --min-citations 100 --max-papers 200

# Paper mill detection: follow image reuse connectivity
rosette campaign create --mode paper_mill --name "Mill cluster" \
  --seed-doi 10.1234/mill1 --max-papers 1000
```

## Analysis Methods

| Method | Module | What It Detects |
|--------|--------|-----------------|
| Error Level Analysis | `image_forensics.py` | JPEG re-compression artifacts from editing |
| Clone Detection | `image_forensics.py` | Copy-move forgery via ORB/SIFT feature matching + RANSAC |
| Block-Based Clone Detection | `image_forensics.py` | Fridrich-style copy-move detection for smooth regions |
| Noise Analysis | `image_forensics.py` | Noise inconsistencies from splicing |
| DCT Analysis | `image_forensics.py` | Double JPEG compression via DCT coefficient periodicity |
| JPEG Ghost Detection | `image_forensics.py` | Mixed compression history regions in images |
| FFT Frequency Analysis | `image_forensics.py` | Frequency-domain manipulation artifacts |
| Metadata Forensics | `metadata_forensics.py` | Software mismatch and ICC profile anomalies |
| GRIM Test | `statistical.py` | Impossible means given sample size |
| GRIMMER Test | `statistical.py` | Impossible SD given mean and sample size |
| Benford's Law | `statistical.py` | Unnatural leading digit distributions |
| P-Value Check | `statistical.py` | Reported p-values inconsistent with test statistics |
| Terminal Digit Test | `statistical.py` | Non-uniform last-digit distributions |
| Variance Ratio Test | `statistical.py` | Suspiciously uniform standard deviations across groups |
| Tortured Phrase Detection | `text_forensics.py` | Paper-mill tortured phrases (e.g., "profound learning" for "deep learning") |
| SPRITE Consistency | `sprite.py` | Impossible mean/SD combinations on Likert scales |
| Western Blot Analysis | `western_blot.py` | Duplicate lanes, splice boundaries, uniform profiles |
| LLM Vision | `llm_vision.py` | Visual anomalies via Claude (requires API key) |
| Cross-Reference | `cross_reference.py` | Duplicate figures across papers via perceptual hashing |
| Author Network | `author_network.py` | Suspicious co-author clusters via Louvain community detection |

Findings from multiple methods are aggregated in `evidence.py`. Converging evidence (2+ methods with confidence >= 0.6 and weight >= 0.3 flagging the same figure) boosts confidence and severity. Compression-sensitive methods (DCT, JPEG ghost, FFT) have weights <= 0.30 so they contribute to scoring but cannot trigger convergence on their own.

## Configuration

Copy and edit `config/default.yaml`, or override with environment variables:

```bash
export ROSETTE__LLM__PROVIDER=claude
export ROSETTE__STORAGE__DATABASE_URL=sqlite:///my.db
export ANTHROPIC_API_KEY=sk-ant-...
```

Key settings:

```yaml
analysis:
  ela:
    quality: 80                # JPEG quality for ELA re-compression (75-85 range)
  clone:
    min_matches: 10            # Minimum ORB matches to flag cloning
  noise:
    block_size: 64             # Block size for noise analysis
  convergence_required: true   # Require multi-method agreement

priority:
  min_citations: 50            # Minimum citations to queue a paper
  min_priority_score: 40       # Minimum priority score to process

campaign:
  auto_risk_promotion_threshold: 30  # Min auto-risk to promote to LLM tier
  max_authors_per_paper: 20
  max_papers_per_author: 50
  hash_match_max_distance: 10
  batch_concurrency: 5
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
rosette/             # Main package (installed via pip)
scripts/             # Standalone scripts (demo, fixture download)
tests/               # Test suite
  test_analysis/     #   Analysis + author network tests
  test_api/          #   REST API tests
  test_calibration/  #   Benchmark accuracy tests
  test_campaign/     #   Campaign system tests (77 tests)
  test_discovery/    #   Priority scoring tests
  test_extraction/   #   PDF/stats extraction tests
  test_llm/          #   Prompt validation tests
  test_pipeline/     #   Pipeline orchestrator tests
  test_reporting/    #   Report generation tests
  test_scripts/      #   Script helper function tests
  test_integration/  #   Integration tests (requires fixtures)
  fixtures/          #   Downloaded test data (not checked in)
config/              # YAML configuration
data/                # Runtime data (PDFs, figures, reports, DB)
```

## API Server

`rosette serve` starts a FastAPI server for submitting papers programmatically. Papers are queued for background analysis via the pipeline orchestrator; clients poll for results.

```bash
rosette serve                       # Start on 0.0.0.0:8000
rosette serve --host 127.0.0.1 --port 9000
```

Endpoints:

- `POST /api/v1/papers` — Submit a paper for analysis (DOI or base64 PDF)
- `GET /api/v1/papers/{id}` — Get paper status and metadata
- `GET /api/v1/papers/{id}/report` — Get analysis report
- `POST /api/v1/batch` — Submit a batch of papers
- `GET /api/v1/authors/{id}/risk` — Get author risk profile
- `GET /health` — Health check

Authentication via `X-API-Key` header (configurable, can be disabled for development).
