"""EXIF/ICC metadata forensics for detecting image editing artifacts.

Editing tools (Photoshop, GIMP) leave fingerprints in EXIF data and ICC
color profiles. This module detects software fields containing known editing
tools, create/modify date mismatches, ICC profiles inconsistent with
scientific imaging equipment, and XMP metadata from editing workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Known image editing software signatures
_EDITING_SOFTWARE = {
    "adobe photoshop": "Adobe Photoshop",
    "photoshop": "Adobe Photoshop",
    "gimp": "GIMP",
    "paint.net": "Paint.NET",
    "pixelmator": "Pixelmator",
    "affinity photo": "Affinity Photo",
    "corel": "CorelDRAW/Corel Photo-Paint",
    "paintshop": "PaintShop Pro",
    "photopea": "Photopea",
    "lightroom": "Adobe Lightroom",
    "capture one": "Capture One",
    "darktable": "darktable",
    "rawtherapee": "RawTherapee",
    "snapseed": "Snapseed",
    "imagemagick": "ImageMagick",
    "irfanview": "IrfanView",
}

# Scientific imaging software (expected, not suspicious)
_SCIENTIFIC_SOFTWARE = {
    "imagej",
    "fiji",
    "cellprofiler",
    "metamorph",
    "zeiss",
    "leica",
    "nikon",
    "olympus",
    "keyence",
    "hamamatsu",
    "biorad",
    "bio-rad",
    "chemidoc",
    "licor",
    "li-cor",
    "azure biosystems",
    "syngene",
    "uvp",
}

# ICC profiles commonly associated with editing
_EDITING_ICC_PROFILES = {
    "srgb iec61966-2.1",
    "adobe rgb (1998)",
    "prophoto rgb",
    "display p3",
}

# ICC profiles expected from scientific instruments
_SCIENTIFIC_ICC_PROFILES = {
    "gray gamma 2.2",
    "gray gamma 1.8",
    "generic gray profile",
}


@dataclass
class MetadataFinding:
    """A single finding from metadata analysis."""

    finding_type: str
    description: str
    confidence: float
    severity: str


@dataclass
class MetadataForensicsResult:
    """Result of metadata forensics analysis."""

    suspicious: bool
    findings: list[MetadataFinding] = field(default_factory=list)
    software: str | None = None
    create_date: str | None = None
    modify_date: str | None = None
    icc_profile: str | None = None
    has_xmp: bool = False
    details: str = ""


def _extract_exif(image_path: str) -> dict[str, str]:
    """Extract EXIF metadata as a flat dict of tag-name -> value strings."""
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()
        if not exif_data:
            return {}
        result = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            result[tag_name] = str(value)
        return result
    except Exception as e:
        logger.debug("EXIF extraction failed for %s: %s", image_path, e)
        return {}


def _check_icc_profile(image_path: str) -> str | None:
    """Extract ICC profile description from an image."""
    try:
        img = Image.open(image_path)
        icc = img.info.get("icc_profile")
        if icc and isinstance(icc, bytes):
            # Try to extract the profile description from raw bytes
            # ICC profile description is typically at a known offset
            try:
                desc_marker = b"desc"
                idx = icc.find(desc_marker)
                if idx != -1:
                    # Skip tag type + reserved
                    start = idx + 12
                    length = int.from_bytes(icc[start - 4 : start], "big")
                    desc = icc[start : start + min(length, 100)]
                    return desc.decode("ascii", errors="ignore").strip("\x00").strip()
            except Exception:
                pass
            return f"ICC profile present ({len(icc)} bytes)"
        return None
    except Exception:
        return None


def _check_xmp(image_path: str) -> bool:
    """Check if the image contains XMP metadata (common in edited images)."""
    try:
        with open(image_path, "rb") as f:
            content = f.read(65536)  # Read first 64KB
            return b"<x:xmpmeta" in content or b"xmp" in content.lower()
    except Exception:
        return False


def analyze_metadata(image_path: str) -> MetadataForensicsResult:
    """Analyze image metadata for signs of editing.

    Checks EXIF data, ICC profiles, and XMP metadata for indicators of
    image manipulation or editing with software not typically used for
    scientific image acquisition.

    Args:
        image_path: Path to the image file.

    Returns:
        MetadataForensicsResult with findings and metadata details.
    """
    findings: list[MetadataFinding] = []

    # Extract EXIF data
    exif = _extract_exif(image_path)
    software = exif.get("Software", "")
    create_date = exif.get("DateTimeOriginal", exif.get("DateTime", ""))
    modify_date = exif.get("DateTime", "")

    # Check for editing software
    if software:
        software_lower = software.lower()

        # Check against known editing tools
        for pattern, name in _EDITING_SOFTWARE.items():
            if pattern in software_lower:
                # Check if it's also scientific software (some overlap)
                is_scientific = any(s in software_lower for s in _SCIENTIFIC_SOFTWARE)
                if not is_scientific:
                    findings.append(
                        MetadataFinding(
                            finding_type="editing_software",
                            description=(
                                f"Image was processed with {name} ({software}). "
                                f"This is an image editing tool not typically used for "
                                f"scientific image acquisition."
                            ),
                            confidence=0.6,
                            severity="medium",
                        )
                    )
                break

    # Check for date mismatches
    if create_date and modify_date and create_date != modify_date:
        try:
            fmt = "%Y:%m:%d %H:%M:%S"
            dt_create = datetime.strptime(create_date[:19], fmt)
            dt_modify = datetime.strptime(modify_date[:19], fmt)
            delta = abs((dt_modify - dt_create).total_seconds())

            if delta > 86400:  # More than 1 day apart
                findings.append(
                    MetadataFinding(
                        finding_type="date_mismatch",
                        description=(
                            f"Creation date ({create_date}) and modification date "
                            f"({modify_date}) differ by {delta / 86400:.1f} days. "
                            f"This may indicate post-acquisition editing."
                        ),
                        confidence=0.4,
                        severity="low",
                    )
                )
        except (ValueError, IndexError):
            pass

    # Check ICC profile
    icc_profile = _check_icc_profile(image_path)
    if icc_profile:
        icc_lower = icc_profile.lower()
        # Check for editing-associated profiles
        for edit_profile in _EDITING_ICC_PROFILES:
            if edit_profile in icc_lower:
                findings.append(
                    MetadataFinding(
                        finding_type="icc_profile",
                        description=(
                            f"ICC color profile '{icc_profile}' is typically associated "
                            f"with edited images, not scientific instrument output."
                        ),
                        confidence=0.35,
                        severity="low",
                    )
                )
                break

    # Check for XMP metadata
    has_xmp = _check_xmp(image_path)
    if has_xmp:
        findings.append(
            MetadataFinding(
                finding_type="xmp_metadata",
                description=(
                    "Image contains XMP metadata, commonly added by image editing "
                    "software like Adobe Photoshop or Lightroom."
                ),
                confidence=0.3,
                severity="low",
            )
        )

    # Check for stripped metadata (suspicious if image is large but has no EXIF)
    if not exif:
        try:
            file_size = Path(image_path).stat().st_size
            if file_size > 100_000:  # > 100KB
                findings.append(
                    MetadataFinding(
                        finding_type="stripped_metadata",
                        description=(
                            f"Image ({file_size / 1024:.0f} KB) contains no EXIF metadata. "
                            f"Metadata stripping can be a sign of image editing."
                        ),
                        confidence=0.2,
                        severity="low",
                    )
                )
        except OSError:
            pass

    suspicious = any(f.confidence > 0.5 for f in findings)
    details_parts = [f.description for f in findings]

    return MetadataForensicsResult(
        suspicious=suspicious,
        findings=findings,
        software=software or None,
        create_date=create_date or None,
        modify_date=modify_date or None,
        icc_profile=icc_profile,
        has_xmp=has_xmp,
        details="; ".join(details_parts) if details_parts else "No metadata anomalies detected",
    )
