"""Download and generate test fixtures for rosette integration tests and demos.

Downloads to tests/fixtures/:
  - RSIIL benchmark manipulated images (from GitHub)
  - Synthetic copy-move forgery images (generated locally)
  - Retracted PMC papers with known image issues
  - Bik's 20K survey paper (PMC4941872)
  - Retraction Watch flagged papers (via CrossRef)
  - Clean control papers from high-impact journals

Idempotent: skips already-downloaded files. Shows progress with rich.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageFilter

from rosette.reporting.pretty import console, create_progress

logger = logging.getLogger(__name__)

# Resolve project root relative to this file: rosette/demo/fixtures.py -> project root
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = _PACKAGE_DIR / "tests" / "fixtures"

# ---- a) RSIIL Benchmark Images ----
# The full RSIIL dataset lives on Zenodo (tens of GB).  The GitHub repo only
# ships a handful of example images.  We split them by ground truth:
#   - Forgery images  → tested with expected="findings"
#   - Pristine/natural → used as additional clean controls
RSIIL_BASE = "https://raw.githubusercontent.com/phillipecardenuto/rsiil/main"

# Known forgery — figure_forgery.png is a programmatically manipulated image
# .figs/ illustration images are also forgery samples from the RSIIL repo
RSIIL_FORGERY_IMAGES = [
    ("notebooks/detection_sample/figure_forgery.png", "figure_forgery.png"),
    (".figs/compound-data.jpg", "compound_data.jpg"),
    (".figs/simple-data.jpg", "simple_data.jpg"),
    (".figs/rsiid.jpg", "rsiid.jpg"),
]

# Known clean — pristine reference and a natural (unmanipulated) photograph
RSIIL_CLEAN_IMAGES = [
    ("notebooks/detection_sample/figure_pristine.png", "figure_pristine.png"),
    ("notebooks/natural_image/natural_image.png", "natural_image.png"),
]

# Ground-truth annotation maps — not actual images, exclude from analysis
RSIIL_MASK_IMAGES = {
    "figure_forgery_map.png",
    "figure_pristine_map.png",
}

# ---- Full RSIIL dataset from Zenodo ----
RSIIL_ZENODO_URL = "https://zenodo.org/api/records/15095089/files"
RSIIL_DATA_DIR = _PACKAGE_DIR / "data" / "rsiil"

# Zenodo file names and their target extraction directories.
# We skip trainset.7z (~37 GB) — we aren't training a model, so the test
# split alone provides enough tampered samples for demo evaluation.
_ZENODO_FILES = {
    "artificial_forgery_src_data.zip": "pristine",
    "testset.7z": "test",
}

# ---- c) Retracted PMC Papers ----
# Papers retracted for image manipulation, figure duplication, or data fabrication.
# Includes papers from Bik's MCB study and other well-documented retractions.
RETRACTED_PAPERS = [
    {
        "pmcid": "PMC3838194",
        "desc": "Retracted MCB paper with known image duplication",
        "filename": "PMC3838194.pdf",
    },
    {
        "pmcid": "PMC3911491",
        "desc": "Retracted paper with Western blot manipulation",
        "filename": "PMC3911491.pdf",
    },
    {
        "pmcid": "PMC4230661",
        "desc": "Retracted paper with figure re-use",
        "filename": "PMC4230661.pdf",
    },
    {
        "pmcid": "PMC10574276",
        "desc": "Retracted: RGX365 myogenesis effects (Nutrients, 2023)",
        "filename": "PMC10574276.pdf",
    },
    {
        "pmcid": "PMC10336160",
        "desc": "Retracted: PRC2/NANOG suppression of differentiation (iScience, 2023)",
        "filename": "PMC10336160.pdf",
    },
    {
        "pmcid": "PMC10428593",
        "desc": "Retracted: LDHD biomarker in lung adenocarcinoma (BMC Cancer, 2023)",
        "filename": "PMC10428593.pdf",
    },
    {
        "pmcid": "PMC9750160",
        "desc": "Retracted: COL10A1-DDR2 in pancreatic cancer (Front Oncol, 2022)",
        "filename": "PMC9750160.pdf",
    },
    {
        "pmcid": "PMC10802042",
        "desc": "Retracted: BTG1 prediction in AML (Clin Epigenetics, 2024)",
        "filename": "PMC10802042.pdf",
    },
    {
        "pmcid": "PMC10945082",
        "desc": "Retracted: Genetic mutations in STAD (J Cell Mol Med, 2024)",
        "filename": "PMC10945082.pdf",
    },
    {
        "pmcid": "PMC10808764",
        "desc": "Retracted: Dual-phenotype HCC MRI features (Front Oncol, 2023)",
        "filename": "PMC10808764.pdf",
    },
    {
        "pmcid": "PMC6374809",
        "desc": "Retracted: Western blot band duplication in colorectal cancer (BioMed Res Int)",
        "filename": "PMC6374809.pdf",
    },
    {
        "pmcid": "PMC8459722",
        "desc": "Retracted: Systematic Western blot manipulation across figures (PLoS ONE)",
        "filename": "PMC8459722.pdf",
    },
    {
        "pmcid": "PMC5765828",
        "desc": "Retracted: Western blot image manipulation (Oncotarget)",
        "filename": "PMC5765828.pdf",
    },
    {
        "pmcid": "PMC4940002",
        "desc": "Retracted: data fabrication with manipulated gel images (Tumor Biology)",
        "filename": "PMC4940002.pdf",
    },
    {
        "pmcid": "PMC5334359",
        "desc": "Retracted: figure duplication across multiple panels (Oncotarget)",
        "filename": "PMC5334359.pdf",
    },
]

# ---- d) Bik's 20K Survey Paper ----
SURVEY_PAPER = {
    "pmcid": "PMC4941872",
    "desc": "Bik et al. 2016 - The prevalence of inappropriate image duplication",
    "filename": "PMC4941872.pdf",
}

# ---- e) Retraction Watch Flagged Papers ----
RETRACTION_WATCH_PAPERS = [
    {
        "pmcid": "PMC5428407",
        "desc": "Retracted for figure duplication/manipulation",
        "filename": "PMC5428407.pdf",
    },
    {
        "pmcid": "PMC4148020",
        "desc": "Retracted paper with data fabrication and image issues",
        "filename": "PMC4148020.pdf",
    },
    {
        "pmcid": "PMC10494514",
        "desc": "Retracted: gastric cancer KDM5C image duplication from 8+ papers",
        "filename": "PMC10494514.pdf",
    },
    {
        "pmcid": "PMC6003856",
        "desc": "Retracted: figure manipulation flagged by Bik (PLoS ONE)",
        "filename": "PMC6003856.pdf",
    },
    {
        "pmcid": "PMC10010624",
        "desc": "Retracted: miR-29b gastric cancer wound healing image duplication",
        "filename": "PMC10010624.pdf",
    },
]

# ---- f) Clean Control Papers ----
# Landmark papers from top journals with no known integrity issues.
# Diverse subjects: genome editing, structural biology, single-cell, immunotherapy,
# epigenetics, computational biology, microbiome, neuroscience, cancer genomics.
CLEAN_PAPERS = [
    {
        "pmcid": "PMC7095418",
        "desc": "SARS-CoV-2 structure (Nature, 2020)",
        "filename": "PMC7095418.pdf",
    },
    {
        "pmcid": "PMC7228219",
        "desc": "COVID-19 clinical features (Lancet, 2020)",
        "filename": "PMC7228219.pdf",
    },
    {"pmcid": "PMC3624763", "desc": "CRISPR review (Science, 2013)", "filename": "PMC3624763.pdf"},
    {
        "pmcid": "PMC6599654",
        "desc": "Deep learning in medicine review",
        "filename": "PMC6599654.pdf",
    },
    {
        "pmcid": "PMC5766781",
        "desc": "Single-cell RNA-seq (Nature Methods)",
        "filename": "PMC5766781.pdf",
    },
    {
        "pmcid": "PMC3795411",
        "desc": "Cong et al. - CRISPR/Cas genome engineering (Science, 2013)",
        "filename": "PMC3795411.pdf",
    },
    {
        "pmcid": "PMC6286148",
        "desc": "Jinek et al. - RNA-guided DNA endonuclease (Science, 2012)",
        "filename": "PMC6286148.pdf",
    },
    {
        "pmcid": "PMC4873371",
        "desc": "Komor et al. - Programmable base editing (Nature, 2016)",
        "filename": "PMC4873371.pdf",
    },
    {
        "pmcid": "PMC6907074",
        "desc": "Anzalone et al. - Prime editing (Nature, 2019)",
        "filename": "PMC6907074.pdf",
    },
    {
        "pmcid": "PMC4481139",
        "desc": "Macosko et al. - Drop-seq single-cell profiling (Cell, 2015)",
        "filename": "PMC4481139.pdf",
    },
    {
        "pmcid": "PMC7433347",
        "desc": "June & Sadelain - CAR-T therapy review (NEJM, 2018)",
        "filename": "PMC7433347.pdf",
    },
    {
        "pmcid": "PMC4015143",
        "desc": "Horvath - DNA methylation age / epigenetic clock (Genome Biology, 2013)",
        "filename": "PMC4015143.pdf",
    },
    {
        "pmcid": "PMC8371605",
        "desc": "Jumper et al. - AlphaFold protein structure prediction (Nature, 2021)",
        "filename": "PMC8371605.pdf",
    },
    {
        "pmcid": "PMC3564958",
        "desc": "Human Microbiome Project Consortium (Nature, 2012)",
        "filename": "PMC3564958.pdf",
    },
    {
        "pmcid": "PMC4078027",
        "desc": "Liao et al. - TRPV1 cryo-EM structure (Nature, 2013)",
        "filename": "PMC4078027.pdf",
    },
    {
        "pmcid": "PMC4139937",
        "desc": "Nishimasu et al. - Cas9 crystal structure (Cell, 2014)",
        "filename": "PMC4139937.pdf",
    },
    {
        "pmcid": "PMC3439153",
        "desc": "ENCODE Consortium - Encyclopedia of DNA elements (Nature, 2012)",
        "filename": "PMC3439153.pdf",
    },
    {
        "pmcid": "PMC4790845",
        "desc": "Deisseroth - Optogenetics 10 years review (Nat Neurosci, 2015)",
        "filename": "PMC4790845.pdf",
    },
    {
        "pmcid": "PMC5394987",
        "desc": "Saxton & Sabatini - mTOR signaling review (Cell, 2017)",
        "filename": "PMC5394987.pdf",
    },
    {
        "pmcid": "PMC4845755",
        "desc": "Sirohi et al. - Zika virus cryo-EM structure (Science, 2016)",
        "filename": "PMC4845755.pdf",
    },
    {
        "pmcid": "PMC3776390",
        "desc": "Alexandrov et al. - Mutational signatures in cancer (Nature, 2013)",
        "filename": "PMC3776390.pdf",
    },
    {
        "pmcid": "PMC7745181",
        "desc": "Polack et al. - BNT162b2 COVID-19 vaccine Phase 3 trial (NEJM, 2020)",
        "filename": "PMC7745181.pdf",
    },
    {
        "pmcid": "PMC7383595",
        "desc": "RECOVERY trial - Dexamethasone in COVID-19 (NEJM, 2021)",
        "filename": "PMC7383595.pdf",
    },
    {
        "pmcid": "PMC7164637",
        "desc": "Wrapp et al. - SARS-CoV-2 spike cryo-EM structure (Science, 2020)",
        "filename": "PMC7164637.pdf",
    },
    {
        "pmcid": "PMC8728224",
        "desc": "Varadi et al. - AlphaFold Protein Structure Database (NAR, 2022)",
        "filename": "PMC8728224.pdf",
    },
    {
        "pmcid": "PMC9812260",
        "desc": "Tabula Sapiens - Human cell atlas 500K cells (Science, 2022)",
        "filename": "PMC9812260.pdf",
    },
]


def _download_file(url: str, dest: Path, client: httpx.Client) -> bool:
    """Download a file if it doesn't already exist. Returns True if downloaded.

    On failure, prints a visible warning to the console (not just to the
    logger, which may be unconfigured when invoked via the CLI).
    """
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = client.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        console.print(f"  [red]FAILED[/red] {dest.name}: {exc}")
        return False


def _download_streaming(
    url: str, dest: Path, client: httpx.Client, desc: str = "Downloading"
) -> bool:
    """Stream-download a large file with progress bar.

    Skips if *dest* already exists and its size matches the remote
    Content-Length.  Removes and re-downloads partial/corrupt files.
    Returns True if a new file was downloaded.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=600.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            # Skip if file already exists and is complete
            if dest.exists() and total > 0 and dest.stat().st_size == total:
                return False
            # Remove partial downloads
            if dest.exists():
                dest.unlink()

            with create_progress() as progress:
                task = progress.add_task(desc, total=total or None)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.advance(task, len(chunk))

        # Verify download completeness
        if total > 0 and dest.stat().st_size != total:
            console.print(
                f"  [red]INCOMPLETE[/red] {dest.name}: "
                f"got {dest.stat().st_size} bytes, expected {total}"
            )
            dest.unlink()
            return False

        return True
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        console.print(f"  [red]FAILED[/red] {dest.name}: {exc}")
        # Remove partial download
        if dest.exists():
            dest.unlink()
        return False


def _find_7z() -> str:
    """Find the native 7z binary, raising a clear error if not installed."""
    for name in ("7z", "7za"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "7z is not installed. Install it with:\n"
        "  macOS:  brew install p7zip\n"
        "  Ubuntu: sudo apt install p7zip-full\n"
        "  Fedora: sudo dnf install p7zip p7zip-plugins"
    )


def _extract_7z(archive_path: Path, target_dir: Path) -> None:
    """Extract a .7z archive using the native 7z binary with progress."""
    bin_7z = _find_7z()
    target_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [bin_7z, "x", str(archive_path), f"-o{target_dir}", "-y", "-bsp1"],
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7z extraction failed (exit {result.returncode}):\n{result.stderr}"
        )


def download_rsiil_zenodo(client: httpx.Client) -> dict[str, int]:
    """Download and extract the full RSIIL dataset from Zenodo.

    Downloads three archive files, extracts them into ``data/rsiil/``, and
    deletes the archives afterward.  Idempotent: skips extraction if target
    directories already contain files.

    Returns a dict with per-split file counts, e.g.
    ``{"pristine": 2923, "train": 26496, "test": 12927}``.
    """
    import zipfile

    RSIIL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for archive_name, subdir in _ZENODO_FILES.items():
        target_dir = RSIIL_DATA_DIR / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Skip extraction if target directory already has files
        existing = list(target_dir.rglob("*"))
        existing_files = [p for p in existing if p.is_file()]
        if existing_files:
            console.print(
                f"  [dim]{subdir}/ already has {len(existing_files)} files — skipping.[/dim]"
            )
            counts[subdir] = len(existing_files)
            continue

        # Download the archive
        archive_path = RSIIL_DATA_DIR / archive_name
        download_url = f"{RSIIL_ZENODO_URL}/{archive_name}/content"
        console.print(f"  Downloading [bold]{archive_name}[/bold]...")
        downloaded = _download_streaming(download_url, archive_path, client, desc=archive_name)

        if not downloaded and not archive_path.exists():
            console.print(f"  [red]Skipping {subdir} — download failed.[/red]")
            counts[subdir] = 0
            continue

        # Extract with progress tracking
        console.print(f"  Extracting to [bold]{subdir}/[/bold]...")
        try:
            if archive_name.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    members = zf.namelist()
                    target_resolved = str(target_dir.resolve())
                    for member in members:
                        member_path = (target_dir / member).resolve()
                        if not str(member_path).startswith(target_resolved):
                            raise ValueError(
                                f"Zip Slip detected: {member!r} escapes target directory"
                            )
                    with create_progress() as progress:
                        task = progress.add_task(
                            f"Extracting {archive_name}", total=len(members)
                        )
                        for member in members:
                            zf.extract(member, target_dir)
                            progress.advance(task)
            elif archive_name.endswith(".7z"):
                _extract_7z(archive_path, target_dir)
        except Exception as exc:
            console.print(f"  [red]Extraction failed for {archive_name}: {exc}[/red]")
            counts[subdir] = 0
            continue

        # Delete archive after successful extraction
        try:
            archive_path.unlink()
            console.print(f"  [dim]Deleted {archive_name} to save disk space.[/dim]")
        except OSError:
            pass

        extracted_files = [p for p in target_dir.rglob("*") if p.is_file()]
        counts[subdir] = len(extracted_files)
        console.print(f"  [green]{subdir}: {len(extracted_files)} files extracted.[/green]")

    return counts


def _is_ground_truth(path: Path) -> bool:
    """Return True for ground-truth masks and annotation overlays."""
    name = path.name.lower()
    return name.endswith("_map.png") or name.endswith("_gt.png")


def _is_pristine_ref(path: Path) -> bool:
    """Return True for pristine reference images in the RSIIL testset."""
    name = path.name.lower()
    # Explicit pristine markers in filenames
    if "_pristine" in name or "_host_pristine" in name:
        return True
    # Files under simple/pristine/ are all pristine
    parts = path.parts
    if "simple" in parts and "pristine" in parts:
        return True
    return False


def sample_rsiil_images(sample_size: int = 30, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Sample images from the full RSIIL Zenodo dataset for demo analysis.

    Returns ``(pristine_paths, tampered_paths)`` — each a list of up to
    *sample_size* image paths.  If the Zenodo dataset has not been downloaded
    the returned lists are empty.

    Ground-truth masks (``*_map.png``, ``*_gt.png``) are excluded.  Pristine
    reference images (``*_pristine*``, files under ``simple/pristine/``) are
    placed in the pristine pool, not the tampered pool.
    """
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    pristine_dir = RSIIL_DATA_DIR / "pristine"
    test_dir = RSIIL_DATA_DIR / "test"

    def _collect(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # Pristine source images (from the zip archive).
    # Filter out ground-truth masks (*_gt.png, *_map.png) — these are binary
    # annotations, not real images, and produce extreme forensic artifacts.
    pristine_all = [p for p in _collect(pristine_dir) if not _is_ground_truth(p)]

    # Testset images need classification: tampered vs pristine vs excluded
    test_all = _collect(test_dir)
    tampered_all: list[Path] = []
    for p in test_all:
        if _is_ground_truth(p):
            continue  # skip masks and GT annotations
        if _is_pristine_ref(p):
            pristine_all.append(p)
        else:
            tampered_all.append(p)

    pristine_all.sort()

    if not pristine_all and not tampered_all:
        return [], []

    rng = np.random.RandomState(seed)

    pristine_sample: list[Path] = []
    if pristine_all:
        idx = rng.choice(len(pristine_all), size=min(sample_size, len(pristine_all)), replace=False)
        pristine_sample = [pristine_all[i] for i in sorted(idx)]

    tampered_sample: list[Path] = []
    if tampered_all:
        idx = rng.choice(len(tampered_all), size=min(sample_size, len(tampered_all)), replace=False)
        tampered_sample = [tampered_all[i] for i in sorted(idx)]

    return pristine_sample, tampered_sample


def _pmc_pdf_url(pmcid: str) -> str:
    """Construct a PDF download URL via Europe PMC.

    The old NCBI endpoint (www.ncbi.nlm.nih.gov/pmc/articles/.../pdf/) now
    returns 403 behind Cloudflare.  Europe PMC's render endpoint works for all
    PMC articles (not just Open Access) with no authentication required.
    """
    return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"


def download_rsiil(client: httpx.Client) -> int:
    """Download RSIIL benchmark images. Returns count of newly downloaded files."""
    out_dir = FIXTURES_DIR / "rsiil"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    all_images = RSIIL_FORGERY_IMAGES + RSIIL_CLEAN_IMAGES
    with create_progress() as progress:
        task = progress.add_task("RSIIL images", total=len(all_images))
        for remote_path, local_name in all_images:
            url = f"{RSIIL_BASE}/{remote_path}"
            dest = out_dir / local_name
            if _download_file(url, dest, client):
                downloaded += 1
            progress.advance(task)

    return downloaded


def generate_synthetic_forgeries() -> int:
    """Generate synthetic copy-move forgery test images. Returns count created."""
    out_dir = FIXTURES_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    created = 0

    rng = np.random.RandomState(42)

    configs = [
        ("copymove_gel_01", "gel-like background with copied band"),
        ("copymove_gel_02", "gel background with duplicated lane"),
        ("copymove_micro_01", "microscopy-like with cloned cell cluster"),
        ("copymove_micro_02", "microscopy with duplicated region"),
        ("copymove_blot_01", "Western blot with copied band"),
        ("copymove_blot_02", "Western blot with lane duplication"),
        ("copymove_chart_01", "chart background with copied bar"),
        ("spliced_panel_01", "two different noise profiles spliced"),
        ("retouched_01", "heavy blur applied to region"),
        ("retouched_02", "region with different JPEG compression"),
        ("copymove_gel_03", "gel with smaller copied band region"),
        ("copymove_flow_01", "flow cytometry-like with cloned dot cluster"),
        ("spliced_panel_02", "three-region splice with varying noise"),
        ("retouched_03", "selective sharpening applied to region"),
        ("copymove_tissue_01", "tissue-like background with copied region"),
    ]

    with create_progress() as progress:
        task = progress.add_task("Synthetic forgeries", total=len(configs))
        for name, _ in configs:
            dest = out_dir / f"{name}.png"
            if dest.exists():
                progress.advance(task)
                continue

            width, height = 512, 512
            base = rng.randint(20, 60, (height, width, 3), dtype=np.uint8)

            for y in range(0, height, 64):
                intensity = rng.randint(40, 200)
                base[y : y + 32, :, :] = intensity

            img = Image.fromarray(base)

            if "copymove" in name:
                arr = np.array(img)
                src_y, src_x = rng.randint(50, 200), rng.randint(50, 200)
                size = rng.randint(60, 120)
                dst_y = min(src_y + rng.randint(100, 200), height - size)
                dst_x = min(src_x + rng.randint(50, 150), width - size)
                arr[dst_y : dst_y + size, dst_x : dst_x + size] = arr[
                    src_y : src_y + size, src_x : src_x + size
                ]
                img = Image.fromarray(arr)

            elif "spliced" in name:
                arr = np.array(img, dtype=np.float64)
                arr[:, : width // 2] += rng.normal(0, 2, (height, width // 2, 3))
                arr[:, width // 2 :] += rng.normal(0, 50, (height, width - width // 2, 3))
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

            elif "retouched" in name:
                arr = np.array(img)
                region = Image.fromarray(arr[100:250, 100:300])
                region = region.filter(ImageFilter.GaussianBlur(radius=8))
                arr[100:250, 100:300] = np.array(region)
                img = Image.fromarray(arr)

            img.save(str(dest))
            created += 1
            progress.advance(task)

    return created


def download_pmc_papers(papers: list[dict], category: str, client: httpx.Client) -> int:
    """Download PDFs from PMC. Returns count of newly downloaded files."""
    out_dir = FIXTURES_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with create_progress() as progress:
        task = progress.add_task(f"PMC {category}", total=len(papers))
        for paper in papers:
            url = _pmc_pdf_url(paper["pmcid"])
            dest = out_dir / paper["filename"]
            if _download_file(url, dest, client):
                downloaded += 1
            progress.advance(task)

    return downloaded


def _count_files(directory: Path) -> int:
    """Count files in a directory (non-recursive)."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file())


def download_all() -> dict[str, int]:
    """Download all fixture categories. Returns per-category download counts."""
    console.rule("[bold blue]Downloading Test Fixtures[/bold blue]")
    console.print()

    results: dict[str, int] = {}

    with httpx.Client(
        headers={"User-Agent": "rosette-test-fixtures/0.1 (academic research tool)"},
        follow_redirects=True,
        timeout=60.0,
    ) as client:
        console.print("[bold]1/6[/bold] RSIIL Benchmark Images")
        results["rsiil"] = download_rsiil(client)

        console.print("[bold]2/6[/bold] Synthetic Forgery Images")
        results["synthetic"] = generate_synthetic_forgeries()

        console.print("[bold]3/6[/bold] Retracted PMC Papers (Bik MCB Study)")
        results["retracted"] = download_pmc_papers(RETRACTED_PAPERS, "retracted", client)

        console.print("[bold]4/6[/bold] Bik 20K Survey Paper")
        results["survey"] = download_pmc_papers([SURVEY_PAPER], "survey", client)

        console.print("[bold]5/6[/bold] Retraction Watch Flagged Papers")
        results["retraction_watch"] = download_pmc_papers(
            RETRACTION_WATCH_PAPERS, "retraction_watch", client
        )

        console.print("[bold]6/6[/bold] Clean Control Papers")
        results["clean"] = download_pmc_papers(CLEAN_PAPERS, "clean", client)

    console.print()
    console.rule("[bold green]Download Complete[/bold green]")

    total_new = sum(results.values())
    if total_new == 0:
        console.print("[dim]All fixtures already present. Nothing to download.[/dim]")
    else:
        console.print(f"[bold]Downloaded {total_new} new files.[/bold]")

    # Per-category summary with expected vs actual counts
    expected_counts: dict[str, int] = {
        "rsiil": len(RSIIL_FORGERY_IMAGES) + len(RSIIL_CLEAN_IMAGES),
        "synthetic": 10,
        "retracted": len(RETRACTED_PAPERS),
        "survey": 1,
        "retraction_watch": len(RETRACTION_WATCH_PAPERS),
        "clean": len(CLEAN_PAPERS),
    }
    any_missing = False
    for cat, count in results.items():
        existing = _count_files(FIXTURES_DIR / cat)
        expected = expected_counts.get(cat, 0)
        if existing < expected:
            any_missing = True
            console.print(
                f"  [yellow]{cat}: {count} new, {existing}/{expected} files "
                f"({expected - existing} missing)[/yellow]"
            )
        else:
            console.print(f"  {cat}: {count} new, {existing} total files")

    if any_missing:
        console.print()
        console.print(
            "[yellow]Some fixtures failed to download. "
            "Re-run to retry, or check network/firewall settings.[/yellow]"
        )

    console.print()
    return results
