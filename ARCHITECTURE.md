# Snoopy Architecture

## Overview

Snoopy is an academic integrity analysis system that detects image manipulation, statistical fabrication, and data recycling in scientific papers. It operates in two modes:

- **Without AI** (default, `--skip-llm`): Runs all deterministic/CV-based methods
- **With AI** (`--use-llm` + `ANTHROPIC_API_KEY`): Adds LLM vision screening and statistical analysis

There are two execution paths: the **demo pipeline** (`snoopy demo`) for benchmarking against known fixtures, and the **production pipeline** (`snoopy analyze`/`snoopy batch`) for real papers with full DB persistence.

---

## Full Data Processing Pipeline

```
                          ┌─────────────────────────────────────────┐
                          │              USER INPUT                  │
                          │   CLI: demo / analyze / batch / discover │
                          └──────────────┬──────────────────────────┘
                                         │
                    ┌────────────────────┬┴───────────────────────┐
                    │                    │                        │
                    ▼                    ▼                        ▼
           ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
           │  Demo Runner  │   │    Orchestrator   │   │    Discovery    │
           │  runner.py    │   │  orchestrator.py  │   │  openalex.py   │
           │  (standalone) │   │  (DB-backed)      │   │  pubmed.py     │
           └──────┬───────┘   └────────┬──────────┘   │  crossref.py   │
                  │                    │               │  semantic_      │
                  │                    │               │   scholar.py   │
                  │                    │               │  unpaywall.py  │
                  │                    │               └───────┬────────┘
                  │                    │                       │
                  │                    │               ┌───────▼────────┐
                  │                    │               │   Priority     │
                  │                    │               │  Scoring       │
                  │                    │               │  priority.py   │
                  │                    │               └───────┬────────┘
                  │                    │                       │
                  │                    │               ┌───────▼────────┐
                  │                    │               │   Database     │
                  │                    ├──────────────►│  models.py     │
                  │                    │               │  session.py    │
                  │                    │               └────────────────┘
                  │                    │
                  ▼                    ▼
     ┌────────────────────────────────────────────────────────────┐
     │                      EXTRACTION LAYER                      │
     │                                                            │
     │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
     │  │ PDF Parser   │  │   Figure     │  │  Table           │ │
     │  │ pdf_parser.py│  │  Extractor   │  │  Extractor       │ │
     │  │              │  │  figure_     │  │  table_          │ │
     │  │ - extract_   │  │  extractor.py│  │  extractor.py    │ │
     │  │   text()     │  │              │  │                  │ │
     │  │ - extract_   │  │ - extract_   │  │ - extract_       │ │
     │  │   metadata() │  │   figures()  │  │   tables()       │ │
     │  │ - download_  │  │ - associate_ │  │                  │ │
     │  │   pdf()      │  │   captions() │  │                  │ │
     │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
     │         │                 │                    │           │
     │  ┌──────▼───────┐        │                    │           │
     │  │ Stats        │        │                    │           │
     │  │ Extractor    │        │                    │           │
     │  │ stats_       │        │                    │           │
     │  │ extractor.py │        │                    │           │
     │  │              │        │                    │           │
     │  │ - extract_   │        │                    │           │
     │  │   means_     │        │                    │           │
     │  │   and_ns()   │        │                    │           │
     │  │ - extract_   │        │                    │           │
     │  │   test_      │        │                    │           │
     │  │   statistics()│       │                    │           │
     │  │ - extract_   │        │                    │           │
     │  │   p_values() │        │                    │           │
     │  │ - extract_   │        │                    │           │
     │  │   numerical_ │        │                    │           │
     │  │   values()   │        │                    │           │
     │  └──────┬───────┘        │                    │           │
     └─────────┼────────────────┼────────────────────┼───────────┘
               │                │                    │
               ▼                ▼                    ▼
     ┌────────────────────────────────────────────────────────────┐
     │                      ANALYSIS LAYER                        │
     │                    (Detection Methods)                     │
     │                                                            │
     │  ┌─────────────────────────────────────────────────────┐   │
     │  │             IMAGE FORENSICS (per figure)            │   │
     │  │  image_forensics.py                                 │   │
     │  │  - error_level_analysis()  [ELA]                    │   │
     │  │  - clone_detection()       [ORB + RANSAC]           │   │
     │  │  - noise_analysis()        [Laplacian variance]     │   │
     │  └─────────────────────────────────────────────────────┘   │
     │                                                            │
     │  ┌─────────────────────────────────────────────────────┐   │
     │  │           STATISTICAL INTEGRITY (per paper)         │   │
     │  │  statistical.py                                     │   │
     │  │  - grim_test()             [mean/N consistency]     │   │
     │  │  - pvalue_check()          [p-value recalculation]  │   │
     │  │  - benford_test()          [leading digit law]      │   │
     │  │  - duplicate_value_check() [table pattern analysis] │   │
     │  └─────────────────────────────────────────────────────┘   │
     │                                                            │
     │  ┌─────────────────────────────────────────────────────┐   │
     │  │          CROSS-REFERENCE (per paper)                │   │
     │  │  cross_reference.py                                 │   │
     │  │  - compute_phash() / compute_ahash()                │   │
     │  │  - hash_distance()                                  │   │
     │  │  - Intra-paper: pairwise figure comparison          │   │
     │  │  - Cross-paper: find_cross_paper_duplicates()  [DB] │   │
     │  └─────────────────────────────────────────────────────┘   │
     │                                                            │
     │  ┌─────────────────────────────────────────────────────┐   │
     │  │          LLM VISION (opt-in, per figure)            │   │
     │  │  llm_vision.py + claude.py                          │   │
     │  │  - screen_figure()           [Haiku, fast pass]     │   │
     │  │  - analyze_figure_detailed() [Sonnet, deep pass]    │   │
     │  │  - classify_figure()         [figure type]          │   │
     │  │  - analyze_text() w/ PROMPT_ANALYZE_STATISTICS      │   │
     │  └─────────────────────────────────────────────────────┘   │
     │                                                            │
     └─────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
     ┌────────────────────────────────────────────────────────────┐
     │                   EVIDENCE AGGREGATION                     │
     │  evidence.py                                               │
     │                                                            │
     │  - Group findings by figure_id                             │
     │  - Detect converging evidence (>=2 methods, conf > 0.6)    │
     │  - Downgrade single-method high/critical → medium          │
     │  - Boost severity when methods converge (+1 level)         │
     │  - Compute weighted confidence (avg + convergence bonus)   │
     │  - Determine paper_risk: clean/low/medium/high/critical    │
     │                                                            │
     │  Returns: AggregatedEvidence                               │
     │    .paper_risk          .overall_confidence                │
     │    .converging_evidence .figure_evidence[]                 │
     │    .total_findings      .critical_count                    │
     └─────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
     ┌────────────────────────────────────────────────────────────┐
     │                     REPORTING LAYER                        │
     │                                                            │
     │  ┌─────────────────┐  ┌───────────────┐  ┌─────────────┐  │
     │  │  Dashboard      │  │  Proof Report │  │   Pretty     │  │
     │  │  dashboard.py   │  │  proof.py     │  │   CLI        │  │
     │  │  + template     │  │  + template   │  │   pretty.py  │  │
     │  │                 │  │               │  │              │  │
     │  │  Per-category   │  │  Per-paper    │  │  Rich-based  │  │
     │  │  HTML overview  │  │  HTML/MD      │  │  terminal    │  │
     │  │  with stats     │  │  evidence     │  │  output      │  │
     │  └─────────────────┘  └───────────────┘  └─────────────┘  │
     └────────────────────────────────────────────────────────────┘
```

---

## Detection Pipeline Detail

The detection pipeline runs all applicable methods on each input, then aggregates the results. Methods are **not strictly ordered** — image forensics methods run in parallel on each figure, statistical methods run in parallel on extracted text/tables. The only ordering constraint is that extraction must complete before analysis.

```
                         ┌──────────────┐
                         │  Input Item  │
                         │ (image/PDF)  │
                         └──────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
            ┌───▼───┐      ┌───▼───┐       ┌───▼───┐
            │ IMAGE │      │  PDF  │       │ IMAGE │
            │ only  │      │ only  │       │ only  │
            └───┬───┘      └───┬───┘       └───┬───┘
                │              │               │
                │         ┌────┴────┐          │
                │         ▼         ▼          │
                │    ┌─────────┐ ┌──────────┐  │
                │    │ Extract │ │ Extract  │  │
                │    │ Figures │ │ Text     │  │
                │    │         │ │ + Tables │  │
                │    └────┬────┘ └────┬─────┘  │
                │         │          │         │
                │    ┌────▼────┐     │         │
                │    │ Per-fig │     │         │
                ├────►analysis │     │         │
                │    └────┬────┘     │         │
                │         │          │         │
                ▼         ▼          ▼         ▼
┌──────────────────────────────────────────────────────────┐
│                 DETECTION METHODS (parallel)              │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║          PER-FIGURE METHODS (on each image)         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.35           ║ │
│  ║  │ 1. ELA                 │  Tiered thresholds:     ║ │
│  ║  │    error_level_        │  max_diff >=25 → low    ║ │
│  ║  │    analysis()          │  max_diff >=40 → medium ║ │
│  ║  │                        │  max_diff >=60 → high   ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.85           ║ │
│  ║  │ 2. Clone Detection     │  Tiered thresholds:     ║ │
│  ║  │    clone_detection()   │  inliers >=20 → low     ║ │
│  ║  │    ORB + RANSAC        │  inliers >=40 → medium  ║ │
│  ║  │                        │  inliers >=60 → high    ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.50           ║ │
│  ║  │ 3. Noise Analysis      │  Tiered thresholds:     ║ │
│  ║  │    noise_analysis()    │  ratio >5  → low        ║ │
│  ║  │    Laplacian variance  │  ratio >10 → medium     ║ │
│  ║  │                        │  ratio >20 → high       ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  (computed, not scored  ║ │
│  ║  │ 4. Perceptual Hash     │   individually per fig) ║ │
│  ║  │    compute_phash()     │  Stored for cross-ref   ║ │
│  ║  │    compute_ahash()     │  comparison step        ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      INTRA-PAPER CROSS-REFERENCE (PDF only)         ║ │
│  ║                                               W:0.90║ │
│  ║  Compare all figure phash pairs within same paper:  ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐                         ║ │
│  ║  │ 5. Hash Distance       │  dist <=5  → critical   ║ │
│  ║  │    Pairwise comparison │  dist 6-10 → high       ║ │
│  ║  │    of all figures      │  dist 11-15→ medium     ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      STATISTICAL METHODS (PDF only, from text)      ║ │
│  ║                                                     ║ │
│  ║  text → stats_extractor → statistical tests:        ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.60           ║ │
│  ║  │ 6. GRIM Test           │  1 failure  → low       ║ │
│  ║  │    Mean/N consistency  │  2 failures → medium    ║ │
│  ║  │    extract_means_      │  3+failures → high      ║ │
│  ║  │    and_ns() → grim()   │                         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.80           ║ │
│  ║  │ 7. P-value Recheck     │  diff >0.01  → low     ║ │
│  ║  │    extract_test_       │  sig changed → medium   ║ │
│  ║  │    statistics() →      │  both + >0.05→ high     ║ │
│  ║  │    pvalue_check()      │                         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.30           ║ │
│  ║  │ 8. Benford's Law       │  Requires n >= 50       ║ │
│  ║  │    extract_numerical_  │  p<0.001   → low        ║ │
│  ║  │    values() → benford()│  p<0.0001  → medium     ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      TABLE METHODS (PDF only, from tables)          ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.25           ║ │
│  ║  │ 9. Duplicate Value     │  dup_ratio >0.3 → low  ║ │
│  ║  │    Check               │  + round_ratio  → med   ║ │
│  ║  │    extract_tables() →  │                         ║ │
│  ║  │    duplicate_value_    │                         ║ │
│  ║  │    check()             │                         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      LLM METHODS (opt-in only, --use-llm)          ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.70           ║ │
│  ║  │ 10. LLM Screening     │  Haiku (fast)           ║ │
│  ║  │     screen_figure()    │  suspicious + conf>0.5  ║ │
│  ║  │                        │  triggers detailed:     ║ │
│  ║  │     ─────────────────  │                         ║ │
│  ║  │                        │                         ║ │
│  ║  │ 11. LLM Detailed      │  Sonnet (thorough)      ║ │
│  ║  │     analyze_figure_    │  Per-anomaly findings   ║ │
│  ║  │     detailed()         │  with locations         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │       EVIDENCE AGGREGATION          │
         │       evidence.py                   │
         │                                     │
         │  1. Group findings by figure_id     │
         │                                     │
         │  2. Per-figure:                     │
         │     - Count methods with conf > 0.6 │
         │     - Converging = 2+ methods       │
         │     - Compute severity (max + boost │
         │       if converging)                │
         │     - Single-method: cap at medium  │
         │                                     │
         │  3. Paper-level risk:               │
         │     critical + converging → critical │
         │     2+ figures flagged   → high     │
         │     1 fig + convergence  → medium   │
         │     1 fig, single method → low      │
         │     nothing flagged      → clean    │
         │                                     │
         │  4. Overall confidence:             │
         │     weighted avg + 0.1 if converging│
         └──────────────────┬──────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   RESULT / REPORT        │
              │                          │
              │  - paper_risk level      │
              │  - overall_confidence    │
              │  - converging_evidence   │
              │  - per-figure breakdown  │
              │  - per-method breakdown  │
              │  - statistical_summary   │
              │  - phash_matches         │
              │  - HTML dashboard        │
              │  - individual reports    │
              └──────────────────────────┘
```

### Method Execution Order

Within the detection layer, methods are **not** sequentially ordered — they run as soon as their inputs are available:

| Phase | What runs | Depends on |
|-------|-----------|------------|
| **Extraction** | `extract_figures()`, `extract_text()`, `extract_tables()` | PDF input |
| **Per-figure analysis** | ELA, Clone Detection, Noise Analysis, phash/ahash | Extracted figures |
| **Statistical analysis** | GRIM, p-value recheck, Benford's Law | Extracted text → stats |
| **Table analysis** | Duplicate value check | Extracted tables |
| **Cross-reference** | Intra-paper phash pairwise comparison | Per-figure phash values |
| **LLM analysis** (opt-in) | Screening → Detailed (sequential per figure) | Extracted figures |
| **Aggregation** | `aggregate_findings()` | All findings collected |

For standalone images (not from PDFs), only per-figure methods (ELA, clone detection, noise analysis, phash) apply.

---

## Module Map

```
snoopy/
├── cli.py                    # Click CLI: demo, analyze, batch, discover, status, config
├── config.py                 # Pydantic config: LLM, analysis weights, storage, discovery
│
├── demo/
│   ├── runner.py             # Standalone demo pipeline (no DB, benchmarks fixtures)
│   └── fixtures.py           # Fixture download: synthetic, RSIIL, retracted, clean
│
├── pipeline/
│   ├── orchestrator.py       # Production pipeline (DB-backed, resumable, async)
│   └── stages.py             # Stage definitions and ordering
│
├── extraction/
│   ├── pdf_parser.py         # PyMuPDF: text, metadata, PDF download
│   ├── figure_extractor.py   # PyMuPDF: embedded image extraction + caption association
│   ├── stats_extractor.py    # Regex: means/N, test statistics, p-values, decimals
│   └── table_extractor.py    # pdfplumber: table extraction
│
├── analysis/
│   ├── image_forensics.py    # OpenCV: ELA, clone detection (ORB+RANSAC), noise analysis
│   ├── statistical.py        # scipy: GRIM, Benford, p-value recheck, duplicate check
│   ├── cross_reference.py    # imagehash: phash, ahash, distance, cross-paper lookup
│   ├── llm_vision.py         # LLM vision: screening, detailed analysis, classification
│   └── evidence.py           # Aggregation: convergence detection, risk assessment
│
├── llm/
│   ├── base.py               # LLMProvider protocol / LLMResponse type
│   ├── claude.py             # Anthropic Claude: vision, text, batch, retry logic
│   └── prompts.py            # All prompt templates (screening, analysis, stats, proof)
│
├── discovery/
│   ├── openalex.py           # OpenAlex API
│   ├── pubmed.py             # PubMed/NCBI API
│   ├── semantic_scholar.py   # Semantic Scholar API
│   ├── crossref.py           # Crossref API
│   ├── unpaywall.py          # Unpaywall (OA PDF URLs)
│   └── priority.py           # Priority scoring algorithm
│
├── db/
│   ├── models.py             # SQLAlchemy: Paper, Figure, Finding, Report, ProcessingLog
│   ├── session.py            # Session management
│   └── migrations.py         # Schema migrations
│
└── reporting/
    ├── dashboard.py          # HTML dashboard generation (demo results overview)
    ├── proof.py              # Per-paper HTML/Markdown evidence reports
    ├── pretty.py             # Rich terminal output
    └── templates/
        ├── dashboard.html.j2 # Dashboard Jinja2 template
        └── report.html.j2    # Individual report template
```

---

## Cross-Image Comparison: Current State

**Implemented and wired up:**
- **Intra-paper comparison** (demo pipeline): After extracting all figures from a PDF, `_analyze_pdf()` computes perceptual hashes for each figure, then does pairwise `hash_distance()` comparisons to find duplicated/recycled figures within the same paper. Matches with distance <= 15 generate findings.

**Implemented but NOT wired into the demo pipeline:**
- **Cross-paper comparison** (`find_cross_paper_duplicates()`, `build_hash_index()`): These functions exist in `cross_reference.py` and query the SQLAlchemy database to compare a paper's figures against all other papers in the DB. They work in the production pipeline (orchestrator) context where figures are persisted to the database, but the demo runner is a standalone pipeline that doesn't use the DB.

To enable cross-paper comparison in the demo, the runner would need to either: (a) maintain an in-memory hash index across all analyzed papers and compare as it goes, or (b) persist figure hashes to the DB during the demo run.

---

## Configuration: Method Weights

Defined in `AnalysisConfig` (config.py), these weights reflect research-based reliability:

| Method | Weight | Rationale |
|--------|--------|-----------|
| Perceptual Hash | 0.90 | Very low false-positive rate |
| Clone Detection | 0.85 | High specificity with RANSAC geometric verification |
| P-value Recheck | 0.80 | Deterministic mathematical check |
| LLM Vision | 0.70 | Good but model-dependent |
| GRIM Test | 0.60 | Reliable for integer-scale data |
| Noise Analysis | 0.50 | Moderate — depends on image type |
| ELA | 0.35 | High false-positive rate in literature |
| Benford's Law | 0.30 | Many legitimate non-conformity reasons |
| Duplicate Check | 0.25 | Highly context-dependent |

The evidence aggregation system uses convergence (2+ independent methods agreeing) as a stronger signal than any single method's weight.
