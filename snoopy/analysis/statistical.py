"""Statistical integrity checks for reported research data.

Implements several well-known statistical tests used to detect inconsistencies
or fabrication in published numerical results.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GRIMResult:
    """Result of the GRIM (Granularity-Related Inconsistency of Means) test."""

    consistent: bool
    mean: float
    n: int
    product: float
    nearest_integer: int
    difference: float


@dataclass
class BenfordResult:
    """Result of the Benford's Law conformity test."""

    conforms: bool
    chi_squared: float
    p_value: float
    observed_distribution: dict[int, float]
    expected_distribution: dict[int, float]
    n_values: int


@dataclass
class PValueResult:
    """Result of p-value recalculation consistency check."""

    consistent: bool
    reported_p: float
    computed_p: float
    difference: float
    significance_changed: bool
    test_type: str


@dataclass
class DuplicateResult:
    """Result of duplicate / suspicious value pattern detection."""

    suspicious: bool
    duplicate_count: int
    total_values: int
    duplicate_ratio: float
    round_number_ratio: float
    details: str


def grim_test(
    mean: float,
    n: int,
    scale_points: int = 5,
    decimals: int = 2,
) -> GRIMResult:
    """Apply the GRIM test to a reported mean and sample size.

    The GRIM test checks whether a reported mean is mathematically consistent
    with the reported sample size when the underlying data must be integers
    (e.g. Likert scale items). The product of mean * n should be close to an
    integer value.

    Args:
        mean: The reported mean value.
        n: The reported sample size.
        scale_points: Number of scale points (unused in calculation but retained
            for documentation of the scale).
        decimals: Number of decimal places in the reported mean.

    Returns:
        GRIMResult indicating whether the mean is consistent.
    """
    product = mean * n
    nearest_integer = round(product)
    difference = abs(product - nearest_integer)

    # Tolerance based on precision of reported mean and sample size
    tolerance = (1 / (10 ** decimals)) * n / 2

    consistent = difference <= tolerance

    return GRIMResult(
        consistent=consistent,
        mean=mean,
        n=n,
        product=product,
        nearest_integer=nearest_integer,
        difference=difference,
    )


def benford_test(values: list[float]) -> BenfordResult:
    """Test whether a set of values conforms to Benford's Law.

    Benford's Law predicts the distribution of leading digits in many
    naturally occurring datasets. Fabricated data often deviates from this
    distribution.

    Args:
        values: List of numerical values to test.

    Returns:
        BenfordResult with chi-squared statistic and conformity flag.
    """
    # Filter to positive values and extract leading digits
    positive_values = [v for v in values if v > 0]
    if not positive_values:
        return BenfordResult(
            conforms=True,
            chi_squared=0.0,
            p_value=1.0,
            observed_distribution={d: 0.0 for d in range(1, 10)},
            expected_distribution={d: math.log10(1 + 1 / d) for d in range(1, 10)},
            n_values=0,
        )

    leading_digits: list[int] = []
    for v in positive_values:
        # Get leading digit by converting to string
        s = f"{v:.15g}"
        for ch in s:
            if ch.isdigit() and ch != "0":
                leading_digits.append(int(ch))
                break

    n_values = len(leading_digits)
    if n_values == 0:
        return BenfordResult(
            conforms=True,
            chi_squared=0.0,
            p_value=1.0,
            observed_distribution={d: 0.0 for d in range(1, 10)},
            expected_distribution={d: math.log10(1 + 1 / d) for d in range(1, 10)},
            n_values=0,
        )

    # Expected Benford distribution
    expected_distribution = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

    # Observed distribution
    digit_counts = Counter(leading_digits)
    observed_distribution = {d: digit_counts.get(d, 0) / n_values for d in range(1, 10)}

    # Chi-squared test
    observed_counts = np.array([digit_counts.get(d, 0) for d in range(1, 10)], dtype=np.float64)
    expected_counts = np.array([expected_distribution[d] * n_values for d in range(1, 10)], dtype=np.float64)

    chi2_stat, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)

    conforms = p_value >= 0.001

    return BenfordResult(
        conforms=conforms,
        chi_squared=float(chi2_stat),
        p_value=float(p_value),
        observed_distribution=observed_distribution,
        expected_distribution=expected_distribution,
        n_values=n_values,
    )


def pvalue_check(
    test_type: str,
    statistic: float,
    df: tuple[int, ...],
    reported_p: float,
) -> PValueResult:
    """Recompute a p-value from test statistics and compare to the reported value.

    Supports t-tests, F-tests, chi-squared tests, and Pearson r correlations.

    Args:
        test_type: One of 't', 'F', 'chi2', 'r'.
        statistic: The reported test statistic.
        df: Degrees of freedom as a tuple. For t and chi2 use (df,). For F use
            (df1, df2). For r use (n,) where n is the sample size.
        reported_p: The p-value reported in the paper.

    Returns:
        PValueResult indicating consistency between reported and computed p.
    """
    computed_p: float

    if test_type == "t":
        computed_p = float(stats.t.sf(abs(statistic), df[0]) * 2)
    elif test_type == "F":
        computed_p = float(stats.f.sf(statistic, df[0], df[1]))
    elif test_type == "chi2":
        computed_p = float(stats.chi2.sf(statistic, df[0]))
    elif test_type == "r":
        # Convert Pearson r to t-statistic: t = r * sqrt((n-2)/(1-r^2))
        n = df[0]
        r = statistic
        r_squared = r * r
        if r_squared >= 1.0:
            computed_p = 0.0
        else:
            t_stat = r * math.sqrt((n - 2) / (1 - r_squared))
            computed_p = float(stats.t.sf(abs(t_stat), n - 2) * 2)
    else:
        raise ValueError(f"Unknown test type: {test_type!r}. Use 't', 'F', 'chi2', or 'r'.")

    difference = abs(reported_p - computed_p)

    # Check if significance conclusion changes (crossing the 0.05 threshold)
    reported_significant = reported_p < 0.05
    computed_significant = computed_p < 0.05
    significance_changed = reported_significant != computed_significant

    # Flag as inconsistent if difference > 0.01 or significance conclusion changes
    consistent = difference <= 0.01 and not significance_changed

    return PValueResult(
        consistent=consistent,
        reported_p=reported_p,
        computed_p=computed_p,
        difference=difference,
        significance_changed=significance_changed,
        test_type=test_type,
    )


def duplicate_value_check(values: list[list[str]]) -> DuplicateResult:
    """Check table data for suspicious patterns of duplicate values.

    Examines a table (list of rows, each row a list of string cell values) for
    signs of data fabrication such as excessive exact duplicates, impossible
    precision, and round-number bias.

    Args:
        values: Table data as a list of rows, where each row is a list of
            string cell values.

    Returns:
        DuplicateResult with suspicion indicators and pattern details.
    """
    all_values: list[str] = []
    for row in values:
        for cell in row:
            stripped = cell.strip()
            if stripped:
                all_values.append(stripped)

    total_values = len(all_values)
    if total_values == 0:
        return DuplicateResult(
            suspicious=False,
            duplicate_count=0,
            total_values=0,
            duplicate_ratio=0.0,
            round_number_ratio=0.0,
            details="No values to check.",
        )

    # Count exact duplicates
    value_counts = Counter(all_values)
    duplicate_count = sum(count - 1 for count in value_counts.values() if count > 1)
    duplicate_ratio = duplicate_count / total_values if total_values > 0 else 0.0

    # Check for round-number bias
    numeric_values: list[float] = []
    round_count = 0
    for v in all_values:
        try:
            num = float(v)
            numeric_values.append(num)
            # A number is "round" if it has no significant fractional part
            # or ends in 0 or 5 when considering one decimal place
            if num == int(num):
                round_count += 1
            elif round(num, 1) == num:
                last_digit = int(round(abs(num) * 10)) % 10
                if last_digit == 0 or last_digit == 5:
                    round_count += 1
        except ValueError:
            continue

    round_number_ratio = round_count / len(numeric_values) if numeric_values else 0.0

    # Check for impossible precision (too many decimal places with no variation)
    max_decimals = 0
    for v in all_values:
        if "." in v:
            decimal_part = v.split(".")[-1].rstrip("0")
            max_decimals = max(max_decimals, len(decimal_part))

    # Build details
    details_parts: list[str] = []
    if duplicate_ratio > 0.3:
        details_parts.append(
            f"High duplicate ratio ({duplicate_ratio:.2%}): "
            f"{duplicate_count} duplicates out of {total_values} values."
        )
    if round_number_ratio > 0.7 and len(numeric_values) >= 5:
        details_parts.append(
            f"Round-number bias detected ({round_number_ratio:.2%} of numeric values)."
        )
    if max_decimals > 10:
        details_parts.append(
            f"Suspicious precision: values with up to {max_decimals} decimal places."
        )

    # Most duplicated values
    most_common = value_counts.most_common(3)
    repeated = [(val, cnt) for val, cnt in most_common if cnt > 1]
    if repeated:
        top_dupes = ", ".join(f"'{v}' ({c}x)" for v, c in repeated)
        details_parts.append(f"Most duplicated: {top_dupes}.")

    details = " ".join(details_parts) if details_parts else "No suspicious patterns detected."

    suspicious = duplicate_ratio > 0.3 or (round_number_ratio > 0.7 and len(numeric_values) >= 5)

    return DuplicateResult(
        suspicious=suspicious,
        duplicate_count=duplicate_count,
        total_values=total_values,
        duplicate_ratio=duplicate_ratio,
        round_number_ratio=round_number_ratio,
        details=details,
    )
