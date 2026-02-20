# Rosette Architecture

## Overview

Rosette is an academic integrity analysis system that detects image manipulation, statistical fabrication, and data recycling in scientific papers. It operates in two modes:

- **Without AI** (default, `--skip-llm`): Runs all deterministic/CV-based methods
- **With AI** (`--use-llm` + `ANTHROPIC_API_KEY`): Adds LLM vision screening and statistical analysis

There are three execution paths: the **demo pipeline** (`rosette demo`) for benchmarking against known fixtures, the **production pipeline** (`rosette analyze`/`rosette batch`) for real papers with full DB persistence, and the **campaign investigation system** (`rosette campaign`) for large-scale multi-paper investigations across co-author networks and research domains.

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
     │  │  image_forensics.py + run_analysis.py              │   │
     │  │  - error_level_analysis()  [ELA, z-score scoring]  │   │
     │  │  - clone_detection()       [ORB/SIFT + RANSAC]     │   │
     │  │  - block_clone_detection() [Fridrich DCT blocks]   │   │
     │  │  - noise_analysis()        [Laplacian variance]    │   │
     │  │  - dct_analysis()          [double JPEG detect]    │   │
     │  │  - jpeg_ghost_detection()  [mixed compression]     │   │
     │  │  - frequency_analysis()    [FFT spectral]          │   │
     │  │  - analyze_metadata()      [software/ICC checks]   │   │
     │  │  Images pre-loaded once, shared across methods.    │   │
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
     │  - Detect converging evidence (>=2 methods,                │
│  -   conf >= 0.6 AND method weight >= 0.3)                │
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
│  ║  ┌────────────────────────┐  Weight: 0.70           ║ │
│  ║  │ 1. ELA                 │  Z-score + thresholds:  ║ │
│  ║  │    error_level_        │  max_diff >=25 → low    ║ │
│  ║  │    analysis()          │  max_diff >=40 → medium ║ │
│  ║  │                        │  max_diff >=60 → high   ║ │
│  ║  └────────────────────────┘  (requires z>3σ above   ║ │
│  ║                               mean for med/high)    ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.85           ║ │
│  ║  │ 2. Clone Detection     │  Tiered thresholds:     ║ │
│  ║  │    clone_detection()   │  inliers >=20 → low     ║ │
│  ║  │    ORB/SIFT + RANSAC   │  inliers >=40 → medium  ║ │
│  ║  │    + Lowe's ratio test │  inliers >=60 → high    ║ │
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
│  ║      PHASE 2: ADVANCED IMAGE FORENSICS (per fig)   ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.30           ║ │
│  ║  │ 10. DCT Analysis       │  Periodicity score:     ║ │
│  ║  │     dct_analysis()     │  >0.3 → suspicious      ║ │
│  ║  │     Double JPEG comp.  │  >0.5 → medium          ║ │
│  ║  │     + block copy-move  │  >0.7 → high            ║ │
│  ║  └────────────────────────┘  (compression-sensitive) ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.30           ║ │
│  ║  │ 11. JPEG Ghost         │  Ghost regions:         ║ │
│  ║  │     jpeg_ghost_        │  0 regions → low        ║ │
│  ║  │     detection()        │  1-2 regions → medium   ║ │
│  ║  │     Mixed compression  │  3+ regions → high      ║ │
│  ║  └────────────────────────┘  (compression-sensitive) ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.15           ║ │
│  ║  │ 12. FFT Frequency      │  Spectral score:        ║ │
│  ║  │     frequency_         │  >2.5 → suspicious      ║ │
│  ║  │     analysis()         │  >3.0 → medium          ║ │
│  ║  │     Manipulation det.  │  >5.0 → high            ║ │
│  ║  └────────────────────────┘  (compression-sensitive) ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  (No weight — used for  ║ │
│  ║  │ 13. Metadata Forensics │   context, not scoring) ║ │
│  ║  │     analyze_metadata() │  Software mismatch,     ║ │
│  ║  │     metadata_          │  ICC profile anomalies  ║ │
│  ║  │     forensics.py       │                         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      PHASE 2: ADVANCED STATISTICAL METHODS          ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.60           ║ │
│  ║  │ 14. GRIMMER Test       │  Mean/SD/N consistency  ║ │
│  ║  │     grimmer_test()     │  Inconsistency → high   ║ │
│  ║  │     statistical.py     │                         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.45           ║ │
│  ║  │ 15. Terminal Digit     │  Chi-squared on last    ║ │
│  ║  │     terminal_digit_    │  digit distribution:    ║ │
│  ║  │     test()             │  p<0.001 → medium       ║ │
│  ║  │                        │  p<0.01  → low          ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.70           ║ │
│  ║  │ 16. Variance Ratio     │  Suspiciously uniform   ║ │
│  ║  │     variance_ratio_    │  SDs across groups:     ║ │
│  ║  │     test()             │  p<0.01 → high          ║ │
│  ║  │                        │  else   → medium        ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      PHASE 2: TEXT & SPECIALIZED ANALYSIS           ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.80           ║ │
│  ║  │ 17. Tortured Phrases   │  Unique phrases:        ║ │
│  ║  │     detect_tortured_   │  1-2 → medium           ║ │
│  ║  │     phrases()          │  3-4 → high             ║ │
│  ║  │     text_forensics.py  │  5+  → critical         ║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.65           ║ │
│  ║  │ 18. SPRITE Test        │  Mean/SD achievability  ║ │
│  ║  │     sprite_test()      │  on Likert scales:      ║ │
│  ║  │     sprite.py          │  SD not achievable→high ║ │
│  ║  │                        │  Mean not achievable→med║ │
│  ║  └────────────────────────┘                         ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  (No weight — per-image ║ │
│  ║  │ 19. Western Blot       │   specialized)          ║ │
│  ║  │     analyze_western_   │  Duplicate lanes,       ║ │
│  ║  │     blot()             │  splice boundaries,     ║ │
│  ║  │     western_blot.py    │  uniform profiles       ║ │
│  ║  └────────────────────────┘                         ║ │
│  ╚══════════════════════════════════════════════════════╝ │
│                                                          │
│  ╔══════════════════════════════════════════════════════╗ │
│  ║      LLM METHODS (opt-in only, --use-llm)          ║ │
│  ║                                                     ║ │
│  ║  ┌────────────────────────┐  Weight: 0.70           ║ │
│  ║  │ 20. LLM Screening     │  Haiku (fast)           ║ │
│  ║  │     screen_figure()    │  suspicious + conf>0.5  ║ │
│  ║  │                        │  triggers detailed:     ║ │
│  ║  │     ─────────────────  │                         ║ │
│  ║  │                        │                         ║ │
│  ║  │ 21. LLM Detailed      │  Sonnet (thorough)      ║ │
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
         │     - Count methods w/ conf>=0.6    │
│       AND weight >= 0.3            │
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
| **Per-figure analysis** | ELA, Clone Detection, Block Clone, Noise, DCT, JPEG Ghost, FFT, Metadata Forensics, phash/ahash (images pre-loaded once) | Extracted figures |
| **Statistical analysis** | GRIM, p-value recheck, Benford's Law | Extracted text → stats |
| **Advanced statistical** | GRIMMER, Terminal Digit, Variance Ratio | Extracted text → stats |
| **Text analysis** | Tortured Phrase Detection | Extracted text |
| **Specialized** | SPRITE Consistency, Western Blot Analysis | Extracted stats / figures |
| **Table analysis** | Duplicate value check | Extracted tables |
| **Cross-reference** | Intra-paper phash pairwise comparison | Per-figure phash values |
| **LLM analysis** (opt-in) | Screening → Detailed (sequential per figure) | Extracted figures |
| **Aggregation** | `aggregate_findings()` | All findings collected |

For standalone images (not from PDFs), all per-figure image methods apply (ELA, clone detection, block-based clone detection, noise analysis, DCT, JPEG ghost, FFT, metadata forensics, phash).

---

## Module Map

```
rosette/
├── cli.py                    # Click CLI: demo, analyze, batch, discover, status, config, serve, db
├── cli_campaign.py           # Campaign CLI: create, run, pause, status, list, dashboard, export
├── config.py                 # Pydantic config: LLM, analysis weights, storage, discovery, campaign
├── validation.py             # Input validation utilities (DOIs, paths, parameters)
│
├── demo/
│   ├── runner.py             # Standalone demo pipeline (no DB, benchmarks fixtures)
│   └── fixtures.py           # Fixture download: synthetic, RSIIL, retracted, clean
│
├── pipeline/
│   ├── orchestrator.py       # Production pipeline (DB-backed, resumable, async)
│   └── stages.py             # Stage definitions and ordering (auto/LLM split)
│
├── campaign/
│   ├── orchestrator.py       # Campaign execution engine (3 modes: network, domain, mill)
│   ├── triage.py             # Two-tier funnel: auto risk scoring → LLM promotion
│   ├── expander.py           # Co-author network expansion and seed discovery
│   ├── hash_scanner.py       # Cross-paper image hash matching
│   └── dashboard.py          # Campaign HTML dashboard generation
│
├── extraction/
│   ├── pdf_parser.py         # PyMuPDF: text, metadata, PDF download
│   ├── figure_extractor.py   # PyMuPDF: embedded image extraction + caption association
│   ├── stats_extractor.py    # Regex: means/N, test statistics, p-values, decimals
│   └── table_extractor.py    # pdfplumber: table extraction
│
├── analysis/
│   ├── image_forensics.py    # OpenCV: ELA, clone (ORB/SIFT), noise, DCT, JPEG ghost, FFT
│   ├── statistical.py        # scipy: GRIM, Benford, p-value recheck, duplicate check
│   ├── cross_reference.py    # imagehash: phash, ahash, distance, cross-paper lookup
│   ├── llm_vision.py         # LLM vision: screening, detailed analysis, classification
│   ├── evidence.py           # Aggregation: convergence detection, risk assessment
│   ├── author_network.py     # Co-author graph analysis (Louvain community detection)
│   ├── metadata_forensics.py # Metadata-based manipulation detection
│   ├── western_blot.py       # Western blot-specific analysis
│   ├── run_analysis.py       # Analysis orchestration: image pre-loading + method dispatch
│   ├── text_forensics.py     # Tortured phrase detection
│   └── sprite.py             # SPRITE consistency test for Likert scale data
│
├── llm/
│   ├── base.py               # LLMProvider protocol / LLMResponse type
│   ├── claude.py             # Anthropic Claude: vision, text, batch, retry logic
│   ├── prompts.py            # Prompt templates (screening, analysis, stats, proof)
│   └── prompts_western_blot.py # Western blot-specific prompt templates
│
├── discovery/
│   ├── openalex.py           # OpenAlex API
│   ├── pubmed.py             # PubMed/NCBI API
│   ├── semantic_scholar.py   # Semantic Scholar API
│   ├── crossref.py           # Crossref API
│   ├── unpaywall.py          # Unpaywall (OA PDF URLs)
│   ├── retraction_watch.py   # Retraction Watch status lookups
│   ├── pubpeer.py            # PubPeer comment checks
│   └── priority.py           # Priority scoring algorithm
│
├── api/
│   ├── app.py                # FastAPI application setup
│   ├── routes.py             # REST endpoint handlers
│   └── schemas.py            # Pydantic request/response schemas
│
├── calibration/
│   ├── benchmark.py          # Benchmark suite runner
│   └── metrics.py            # Accuracy, precision, recall metrics
│
├── db/
│   ├── models.py             # SQLAlchemy models (see DB Models below)
│   ├── session.py            # Session management
│   ├── migrations.py         # Schema migrations
│   └── alembic/              # Alembic migration scripts
│
└── reporting/
    ├── dashboard.py          # HTML dashboard generation (demo results overview)
    ├── proof.py              # Per-paper HTML/Markdown evidence reports
    ├── pretty.py             # Rich terminal output
    ├── evidence_package.py   # Evidence package export (campaign results)
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
| Tortured Phrases | 0.80 | Strong signal for paper mill output |
| P-value Recheck | 0.80 | Deterministic mathematical check |
| ELA | 0.70 | Z-score calibrated confidence scoring reduces false positives |
| Variance Ratio | 0.70 | Detects suspiciously uniform standard deviations |
| LLM Vision | 0.70 | Good but model-dependent |
| SPRITE | 0.65 | Mean/SD achievability on Likert scales |
| GRIM Test | 0.60 | Reliable for integer-scale data |
| GRIMMER Test | 0.60 | Extends GRIM to SD/N consistency |
| Noise Analysis | 0.50 | Moderate — depends on image type |
| Terminal Digit | 0.45 | Uniformity test on last digits |
| DCT Analysis | 0.30 | Compression-sensitive — low weight prevents false convergence |
| JPEG Ghost | 0.30 | Compression-sensitive — low weight prevents false convergence |
| Benford's Law | 0.30 | Many legitimate non-conformity reasons |
| Duplicate Check | 0.25 | Highly context-dependent |
| FFT Frequency | 0.15 | Compression-sensitive — low weight prevents false convergence |

Convergence requires 2+ methods with confidence >= 0.6 **and** weight >= 0.3 agreeing on the same figure. Compression-sensitive methods (DCT, JPEG ghost, FFT) have weights <= 0.30, so they contribute to scoring but cannot trigger convergence determination on their own. This prevents PDF extraction artifacts from producing false multi-method agreement.

---

## Pipeline Stages

The production pipeline runs papers through a sequence of stages, each tracked in the DB for resumability:

```
discover → prioritize → download → extract_text → extract_figures → extract_stats
→ analyze_images_auto → analyze_stats → classify_figures → analyze_images_llm
→ aggregate → report
```

The `analyze_images` stage was split into two phases:
- **`analyze_images_auto`**: Deterministic/CV methods (ELA, clone detection, block-based clone detection, noise analysis, DCT, JPEG ghost, FFT, metadata forensics, perceptual hashing) — runs without LLM
- **`analyze_images_llm`**: LLM-based screening and detailed analysis — opt-in, runs after auto tier

The legacy stage name `analyze_images` is mapped to both for backward compatibility.

---

## DB Models

SQLAlchemy models in `db/models.py`:

| Model | Purpose |
|-------|---------|
| `Paper` | Core paper record (DOI, metadata, processing status, risk level) |
| `Figure` | Extracted figure images with captions and hashes |
| `Finding` | Individual detection findings (method, severity, confidence) |
| `Report` | Generated analysis reports (HTML/Markdown) |
| `ProcessingLog` | Stage-level processing log for resumability |
| `Author` | Author records with risk scores |
| `AuthorPaperLink` | Many-to-many author↔paper associations |
| `Campaign` | Investigation campaign definition and state |
| `CampaignPaper` | Paper membership in campaigns with triage tier tracking |
| `ImageHashMatch` | Cross-paper perceptual hash matches |

### DB / Alembic Migrations

Schema migrations are managed via Alembic under `db/alembic/`. Run migrations with `rosette db upgrade` / `rosette db downgrade`.

---

## Campaign Investigation System

The campaign system enables large-scale investigations spanning hundreds of papers. It operates in three modes and uses a two-tier triage funnel to manage LLM costs.

### Three Investigation Modes

| Mode | Description | Seed Input |
|------|-------------|------------|
| **Network Expansion** | Follow co-author networks outward from suspicious papers | Seed DOIs |
| **Domain Scan** | Systematic sweep of a research field | Field name + filters |
| **Paper Mill** | Trace image reuse connectivity across papers | Seed DOIs |

### Two-Tier Triage Funnel

```
All papers → Auto tier (cheap CV/stats) → Risk scoring → Promoted papers → LLM tier (expensive)
```

Papers enter the **auto tier** first, where cheap deterministic methods run. An auto-risk score is computed from signal weights:

| Signal | Points | Description |
|--------|--------|-------------|
| `hash_match` | +30 | Cross-paper perceptual hash match |
| `clone_detection` | +25 | Copy-move forgery detected |
| `ela_suspicious` | +15 | ELA anomaly above threshold |
| `noise_inconsistency` | +15 | Noise variance ratio outlier |
| `grim_failure` | +20 | GRIM test failure |
| `pvalue_mismatch` | +20 | P-value recalculation mismatch |
| `retraction_flagged` | +25 | Known retraction/expression of concern |
| `pubpeer_comments` | +15 | PubPeer flags exist |

Papers scoring above the promotion threshold (default 30) advance to the **LLM tier** for Claude-based vision screening and detailed analysis.

### Campaign Execution Flow

1. **Seed phase**: Collect initial papers (from DOIs, author search, or field query)
2. **Expansion phase** (network/mill modes): Discover connected papers via co-author networks or image hash matching
3. **Triage phase**: Run all papers through auto tier, promote high-risk to LLM tier
4. **Analysis phase**: Full pipeline analysis on promoted papers
5. **Reporting phase**: Generate campaign dashboard and evidence packages

---

## Three-Tier LLM Model Strategy

The system uses three Claude models, each chosen for a specific cost/capability trade-off:

| Tier | Model | Purpose | Rationale |
|------|-------|---------|-----------|
| **Screening** | Haiku (`model_screen`) | Fast pass over every figure | Low cost per image, sufficient for binary suspicious/not flagging |
| **Analysis** | Sonnet (`model_analyze`) | Detailed per-anomaly analysis of flagged figures | Strong vision capability at moderate cost; only runs on figures that pass screening |
| **Proof Reports** | Opus (`model_proof`) | Generating publication-quality evidence reports | Highest reasoning capability for nuanced write-ups; called rarely (once per paper) |

This tiered approach keeps LLM costs manageable: Haiku screens many figures cheaply, Sonnet analyzes only the suspicious subset, and Opus generates final reports only for papers with confirmed findings. Models are configured in `LLMConfig` (`config.py`) and can be overridden via YAML or environment variables
