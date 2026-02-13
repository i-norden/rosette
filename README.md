# Snoopy

LLM-powered academic integrity analyzer. Detects image manipulation, statistical anomalies, and figure duplication in scientific papers using computer vision, statistical tests, and Claude-based analysis.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the demo (downloads test fixtures, runs forensics, prints results):

```bash
snoopy demo
```

## Architecture

```
snoopy/
├── analysis/          # Detection methods
│   ├── image_forensics.py   # ELA, clone detection, noise analysis
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
├── notifications/     # Webhook notifications
├── db/                # SQLAlchemy models + Alembic migrations
├── cli.py             # Click CLI entry point
├── cli_campaign.py    # Campaign CLI commands
├── config.py          # Pydantic config with YAML + env var support
└── validation.py      # Input validation utilities
```

## CLI Commands

```
snoopy discover    Search academic APIs for papers to analyze
snoopy analyze     Analyze a single paper by DOI or local PDF
snoopy batch       Process top-priority pending papers
snoopy report      Display a paper's analysis report
snoopy status      Show pipeline status and queue depth
snoopy demo        Run demo with test fixtures and pretty output
snoopy serve       Start the REST API server
snoopy config      Show current configuration
snoopy db          Database migration commands (upgrade/downgrade/current)

snoopy campaign create     Create a new investigation campaign
snoopy campaign run        Start or resume a campaign
snoopy campaign pause      Pause a running campaign
snoopy campaign status     Show campaign progress
snoopy campaign list       List all campaigns
snoopy campaign dashboard  Generate HTML dashboard
snoopy campaign export     Export evidence packages
```

### snoopy demo

Runs the full demo pipeline: downloads test fixtures, runs image forensics on all test cases, prints Rich-formatted results, and generates HTML reports.

```bash
snoopy demo                        # Full demo
snoopy demo --download-only        # Only download fixtures
snoopy demo --output-dir ./out     # Custom report output directory
```

### snoopy analyze

```bash
snoopy analyze --doi 10.1234/example
snoopy analyze --pdf path/to/paper.pdf
```

### snoopy campaign

Large-scale investigation across multiple papers. Three investigation modes:

```bash
# Network expansion: follow co-author networks from suspicious papers
snoopy campaign create --mode network_expansion --name "Dana-Farber follow-up" \
  --seed-doi 10.1234/suspicious1 --seed-doi 10.1234/suspicious2 \
  --max-depth 2 --max-papers 500 --llm-budget 50
snoopy campaign run <campaign-id>

# Domain scan: systematic sweep of a research field
snoopy campaign create --mode domain_scan --name "Stem cell survey" \
  --field "stem cell" --min-citations 100 --max-papers 200

# Paper mill detection: follow image reuse connectivity
snoopy campaign create --mode paper_mill --name "Mill cluster" \
  --seed-doi 10.1234/mill1 --max-papers 1000
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
| Author Network | `author_network.py` | Suspicious co-author clusters via Louvain community detection |

Findings from multiple methods are aggregated in `evidence.py`. Converging evidence (2+ methods flagging the same figure) boosts confidence and severity.

## Configuration

Copy and edit `config/default.yaml`, or override with environment variables:

```bash
export SNOOPY__LLM__PROVIDER=claude
export SNOOPY__STORAGE__DATABASE_URL=sqlite:///my.db
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

campaign:
  auto_risk_promotion_threshold: 30  # Min auto-risk to promote to LLM tier
  max_authors_per_paper: 20
  max_papers_per_author: 50
  hash_match_max_distance: 10
  batch_concurrency: 5
```

## API Server

Start the REST API with `snoopy serve`. Key endpoints:

- `POST /papers` — Submit a paper for analysis
- `GET /papers/{id}` — Get paper status and metadata
- `GET /papers/{id}/report` — Get analysis report
- `POST /batch` — Submit a batch of papers
- `GET /authors/{id}/risk` — Get author risk profile
- `GET /health` — Health check

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
snoopy/             # Main package (installed via pip)
scripts/             # Standalone scripts (demo, fixture download)
tests/               # Test suite
  test_analysis/     #   Analysis + author network tests
  test_api/          #   REST API tests
  test_calibration/  #   Benchmark accuracy tests
  test_campaign/     #   Campaign system tests (77 tests)
  test_discovery/    #   Priority scoring tests
  test_extraction/   #   PDF/stats extraction tests
  test_llm/          #   Prompt validation tests
  test_notifications/#   Webhook tests
  test_pipeline/     #   Pipeline orchestrator tests
  test_reporting/    #   Report generation tests
  test_scripts/      #   Script helper function tests
  test_integration/  #   Integration tests (requires fixtures)
  fixtures/          #   Downloaded test data (not checked in)
config/              # YAML configuration
data/                # Runtime data (PDFs, figures, reports, DB)
```
