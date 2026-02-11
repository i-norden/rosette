"""Extract reported statistical values from text using regex patterns."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MeanReport:
    """A reported mean and sample size."""

    mean: float
    n: int
    context: str


@dataclass
class TestStatistic:
    """A reported test statistic (t, F, chi-square, r, etc.)."""

    test_type: str
    statistic: float
    df: tuple[int, ...]
    p_value: float | None
    context: str


@dataclass
class PValue:
    """A reported p-value."""

    value: float
    comparison: str  # '<', '=', or '>'
    context: str


# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------

_CONTEXT_CHARS = 80


def _surrounding(text: str, match: re.Match, chars: int = _CONTEXT_CHARS) -> str:
    """Return up to *chars* characters of text surrounding *match*."""
    start = max(0, match.start() - chars)
    end = min(len(text), match.end() + chars)
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Mean / N patterns --------------------------------------------------------
# Matches forms like:
#   M = 3.45, N = 120
#   mean = 3.45 (n = 120)
#   M=3.45,N=120
_MEAN_N_RE = re.compile(
    r"(?:M|mean)\s*=\s*"
    r"(?P<mean>-?\d+(?:\.\d+)?)"
    r"[\s,;]*"
    r"[\(\[]?\s*(?:N|n)\s*=\s*"
    r"(?P<n>\d+)"
    r"\s*[\)\]]?",
    re.IGNORECASE,
)

# t-test -------------------------------------------------------------------
# t(df) = X.XX, p = 0.XXX  or  t(df) = X.XX, p < .001
_T_TEST_RE = re.compile(
    r"t\s*\(\s*(?P<df>\d+)\s*\)\s*=\s*"
    r"(?P<stat>-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*p\s*(?P<pcmp>[<>=])\s*(?P<pval>\.?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# F-test -------------------------------------------------------------------
# F(df1, df2) = X.XX, p = 0.XXX
_F_TEST_RE = re.compile(
    r"F\s*\(\s*(?P<df1>\d+)\s*,\s*(?P<df2>\d+)\s*\)\s*=\s*"
    r"(?P<stat>-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*p\s*(?P<pcmp>[<>=])\s*(?P<pval>\.?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# Chi-square ---------------------------------------------------------------
# chi2(df) = X.XX  or  χ²(df) = X.XX
_CHI2_RE = re.compile(
    r"(?:chi2|chi-square|chi-squared|\u03c7\u00b2|\u03c72)\s*"
    r"\(\s*(?P<df>\d+)\s*\)\s*=\s*"
    r"(?P<stat>-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*p\s*(?P<pcmp>[<>=])\s*(?P<pval>\.?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# Correlation coefficient --------------------------------------------------
# r = 0.XX, p = 0.XXX   (but NOT preceded by letters to avoid matching
# words that end in 'r').
_R_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"r\s*=\s*"
    r"(?P<stat>-?\d*\.?\d+)"
    r"(?:\s*,\s*p\s*(?P<pcmp>[<>=])\s*(?P<pval>\.?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

# Standalone p-value -------------------------------------------------------
_P_VALUE_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"p\s*(?P<cmp>[<>=])\s*"
    r"(?P<val>\.?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# General decimal number (for extract_numerical_values) --------------------
_DECIMAL_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"(-?\d+\.\d+)"
    r"(?![a-zA-Z])",
)


# ---------------------------------------------------------------------------
# Helper to normalise numeric strings
# ---------------------------------------------------------------------------


def _parse_float(s: str) -> float:
    """Parse a float, handling leading-dot notation like ``.001``."""
    s = s.strip()
    if s.startswith("."):
        s = "0" + s
    return float(s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_means_and_ns(text: str) -> list[MeanReport]:
    """Find patterns like ``M = X.XX, N = YY`` in *text*.

    Args:
        text: The source text to search.

    Returns:
        A list of :class:`MeanReport` instances.
    """
    results: list[MeanReport] = []
    for m in _MEAN_N_RE.finditer(text):
        try:
            mean = _parse_float(m.group("mean"))
            n = int(m.group("n"))
            results.append(MeanReport(mean=mean, n=n, context=_surrounding(text, m)))
        except (ValueError, TypeError):
            logger.debug("Could not parse mean/N match: %s", m.group(0))
    return results


def extract_test_statistics(text: str) -> list[TestStatistic]:
    """Find test-statistic patterns (t, F, chi-square, r) in *text*.

    Supported formats:
    - ``t(df) = X.XX, p = 0.XXX``
    - ``F(df1, df2) = X.XX, p = 0.XXX``
    - ``chi2(df) = X.XX, p = 0.XXX``
    - ``r = 0.XX, p = 0.XXX``

    Args:
        text: The source text to search.

    Returns:
        A list of :class:`TestStatistic` instances.
    """
    results: list[TestStatistic] = []

    # t-test
    for m in _T_TEST_RE.finditer(text):
        try:
            stat = _parse_float(m.group("stat"))
            df: tuple[int, ...] = (int(m.group("df")),)
            p_val: float | None = None
            if m.group("pval"):
                p_val = _parse_float(m.group("pval"))
            results.append(
                TestStatistic(
                    test_type="t",
                    statistic=stat,
                    df=df,
                    p_value=p_val,
                    context=_surrounding(text, m),
                )
            )
        except (ValueError, TypeError):
            logger.debug("Could not parse t-test match: %s", m.group(0))

    # F-test
    for m in _F_TEST_RE.finditer(text):
        try:
            stat = _parse_float(m.group("stat"))
            df = (int(m.group("df1")), int(m.group("df2")))
            p_val = None
            if m.group("pval"):
                p_val = _parse_float(m.group("pval"))
            results.append(
                TestStatistic(
                    test_type="F",
                    statistic=stat,
                    df=df,
                    p_value=p_val,
                    context=_surrounding(text, m),
                )
            )
        except (ValueError, TypeError):
            logger.debug("Could not parse F-test match: %s", m.group(0))

    # Chi-square
    for m in _CHI2_RE.finditer(text):
        try:
            stat = _parse_float(m.group("stat"))
            df = (int(m.group("df")),)
            p_val = None
            if m.group("pval"):
                p_val = _parse_float(m.group("pval"))
            results.append(
                TestStatistic(
                    test_type="chi2",
                    statistic=stat,
                    df=df,
                    p_value=p_val,
                    context=_surrounding(text, m),
                )
            )
        except (ValueError, TypeError):
            logger.debug("Could not parse chi2 match: %s", m.group(0))

    # Correlation r
    for m in _R_RE.finditer(text):
        try:
            stat = _parse_float(m.group("stat"))
            p_val = None
            if m.group("pval"):
                p_val = _parse_float(m.group("pval"))
            results.append(
                TestStatistic(
                    test_type="r",
                    statistic=stat,
                    df=(),
                    p_value=p_val,
                    context=_surrounding(text, m),
                )
            )
        except (ValueError, TypeError):
            logger.debug("Could not parse r match: %s", m.group(0))

    return results


def extract_p_values(text: str) -> list[PValue]:
    """Find all p-value expressions in *text*.

    Matches forms like ``p < .001``, ``p = 0.05``, ``p > 0.10``.

    Args:
        text: The source text to search.

    Returns:
        A list of :class:`PValue` instances.
    """
    results: list[PValue] = []
    for m in _P_VALUE_RE.finditer(text):
        try:
            value = _parse_float(m.group("val"))
            comparison = m.group("cmp")
            results.append(
                PValue(
                    value=value,
                    comparison=comparison,
                    context=_surrounding(text, m),
                )
            )
        except (ValueError, TypeError):
            logger.debug("Could not parse p-value match: %s", m.group(0))
    return results


def extract_numerical_values(text: str) -> list[float]:
    """Extract all decimal numbers from *text*.

    This is a broad extraction intended for results sections where many
    numeric values may be reported.  Only numbers containing a decimal
    point are returned (integers are skipped to reduce noise).

    Args:
        text: The source text to search.

    Returns:
        A list of floats in the order they appear in the text.
    """
    values: list[float] = []
    for m in _DECIMAL_RE.finditer(text):
        try:
            values.append(float(m.group(1)))
        except ValueError:
            pass
    return values
