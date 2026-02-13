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
    if n < 2:
        return GRIMResult(
            consistent=True,
            mean=mean,
            n=n,
            product=mean * n,
            nearest_integer=round(mean * n),
            difference=abs(mean * n - round(mean * n)),
        )

    product = mean * n
    nearest_integer = round(product)
    difference = abs(product - nearest_integer)

    # GRIM tolerance per Brown & Heathers (2017): independent of N
    tolerance = 0.5 / (10**decimals)

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
    expected_counts = np.array(
        [expected_distribution[d] * n_values for d in range(1, 10)], dtype=np.float64
    )

    chi2_stat, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)

    conforms = bool(p_value >= 0.001)

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
    one_tailed: bool = False,
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
        n = df[0] if df else 0
        r = statistic
        # Validate: need |r| < 1 and n >= 3 (df = n-2 >= 1)
        if abs(r) >= 1.0 or n < 3:
            return PValueResult(
                consistent=True,
                reported_p=reported_p,
                computed_p=reported_p,
                difference=0.0,
                significance_changed=False,
                test_type=test_type,
            )
        r_squared = r * r
        t_stat = r * math.sqrt((n - 2) / (1 - r_squared))
        tail_p = float(stats.t.sf(abs(t_stat), n - 2))
        computed_p = tail_p if one_tailed else tail_p * 2
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


# ---------------------------------------------------------------------------
# 2.6 GRIMMER test
# ---------------------------------------------------------------------------


@dataclass
class GRIMMERResult:
    """Result of the GRIMMER (SD) consistency test."""

    consistent: bool
    mean: float
    sd: float
    n: int
    possible_sds: list[float]
    details: str


def grimmer_test(
    mean: float,
    sd: float,
    n: int,
    decimals: int = 2,
) -> GRIMMERResult:
    """Apply the GRIMMER test to a reported mean, SD, and sample size.

    The GRIMMER test extends GRIM by checking whether a reported standard
    deviation is mathematically possible given the reported mean and sample
    size when the underlying data must be integers (e.g. Likert-scale items).

    Args:
        mean: The reported mean value.
        sd: The reported standard deviation.
        n: The reported sample size.
        decimals: Number of decimal places in the reported SD.

    Returns:
        GRIMMERResult indicating whether the SD is consistent.
    """
    # Trivial case: SD is undefined for n < 2, and any single value is valid
    # for n == 2 (only one degree of freedom).  We follow the convention of
    # returning consistent for n < 3 since there is almost no constraint.
    if n < 3:
        return GRIMMERResult(
            consistent=True,
            mean=mean,
            sd=sd,
            n=n,
            possible_sds=[],
            details="N < 3; GRIMMER test not applicable.",
        )

    tolerance = 0.5 / (10**decimals)

    # The total (sum of all items) implied by the reported mean.
    total = round(mean * n)

    # For large N the full enumeration is infeasible; use an approximation.
    if n > 20:
        # Under integer data, the minimum variance occurs when values are as
        # close to the mean as possible (all values are floor or ceil of the
        # mean).  Maximum variance occurs when values are at the extremes of
        # the feasible range.
        item_mean = total / n
        floor_val = math.floor(item_mean)
        ceil_val = floor_val + 1

        # Number of items at ceil_val so that floor*k + ceil*(n-k) = total
        n_ceil = total - n * floor_val
        n_floor = n - n_ceil

        # Minimum sum-of-squares given the constraint
        min_ss = n_floor * (floor_val**2) + n_ceil * (ceil_val**2)
        min_var = (min_ss - (total**2) / n) / (n - 1)
        min_sd = math.sqrt(max(min_var, 0.0))

        # Maximum sum-of-squares: put values at 0 and the maximum integer
        # that keeps the sum equal to total.  A practical upper bound is
        # total itself (one item = total, rest = 0) but we don't know the
        # scale range, so we use this as the upper bound.
        max_ss = total**2  # one item carries the entire sum
        # But the remaining n-1 items are 0, so ss = total^2
        # However ss also equals sum(xi^2), and if one item = total, rest 0:
        max_var = (max_ss - (total**2) / n) / (n - 1)
        max_sd = math.sqrt(max(max_var, 0.0))

        consistent = (sd >= min_sd - tolerance) and (sd <= max_sd + tolerance)
        return GRIMMERResult(
            consistent=consistent,
            mean=mean,
            sd=sd,
            n=n,
            possible_sds=[],  # too many to enumerate
            details=(
                f"Approximation for large N ({n}). "
                f"Feasible SD range: [{min_sd:.{decimals}f}, {max_sd:.{decimals}f}]. "
                f"Reported SD {sd} {'is' if consistent else 'is NOT'} within range."
            ),
        )

    # --- Full enumeration for small N (n <= 20) ---
    # We need to find all possible sum-of-squares values given that the n
    # integer items sum to *total*.  This is equivalent to finding all
    # partitions of *total* into exactly *n* non-negative integer parts and
    # computing sum(xi^2) for each.
    #
    # We use dynamic programming on (remaining items, remaining sum) ->
    # set of achievable sum-of-squares values.

    # Bound each item value: 0 <= xi <= total (conservative; we don't know
    # the scale, so we allow any non-negative integer up to total).
    max_item = total  # upper bound on any single item

    # DP: build set of achievable sum-of-squares.
    # current_ss_set is a set of possible sum-of-squares after placing k items.
    # State: (remaining_sum) -> set of achievable sum-of-squares
    # To keep memory bounded, we iterate item by item.

    # Start: 0 items placed, remaining sum = total, ss so far = 0
    # dp[remaining_sum] = set of achievable ss values
    dp: dict[int, set[int]] = {total: {0}}

    for _ in range(n):
        new_dp: dict[int, set[int]] = {}
        for rem_sum, ss_set in dp.items():
            # This item can be 0..min(rem_sum, max_item)
            upper = min(rem_sum, max_item)
            for xi in range(upper + 1):
                new_rem = rem_sum - xi
                xi_sq = xi * xi
                if new_rem not in new_dp:
                    new_dp[new_rem] = set()
                for prev_ss in ss_set:
                    new_dp[new_rem].add(prev_ss + xi_sq)
        dp = new_dp

    # After placing all n items, remaining sum must be 0.
    achievable_ss = dp.get(0, set())

    # Convert each achievable sum-of-squares to a sample SD.
    possible_sds: list[float] = []
    for ss in sorted(achievable_ss):
        var = (ss - (total**2) / n) / (n - 1)
        if var < -1e-12:
            continue
        possible_sds.append(math.sqrt(max(var, 0.0)))

    # De-duplicate after rounding to the reported precision.
    seen: set[str] = set()
    unique_sds: list[float] = []
    for s in possible_sds:
        key = f"{s:.{decimals + 2}f}"
        if key not in seen:
            seen.add(key)
            unique_sds.append(s)
    possible_sds = unique_sds

    # Check if reported SD matches any achievable SD within tolerance.
    consistent = any(abs(sd - s) <= tolerance for s in possible_sds)

    return GRIMMERResult(
        consistent=consistent,
        mean=mean,
        sd=sd,
        n=n,
        possible_sds=possible_sds,
        details=(
            f"Enumerated {len(possible_sds)} possible SDs for N={n}, mean={mean}. "
            f"Reported SD {sd} {'matches' if consistent else 'does NOT match'} "
            f"any achievable value (tolerance={tolerance})."
        ),
    )


# ---------------------------------------------------------------------------
# 2.7 Terminal digit analysis
# ---------------------------------------------------------------------------


@dataclass
class TerminalDigitResult:
    """Result of a terminal-digit uniformity test."""

    suspicious: bool
    chi_squared: float
    p_value: float
    digit_counts: dict[int, int]
    n_values: int
    details: str


def _extract_terminal_digit(value: float) -> int | None:
    """Return the last significant digit of *value*.

    For values with a fractional part the last non-zero digit after the
    decimal point is returned.  For integers the units digit is returned.
    Returns ``None`` when no meaningful digit can be extracted.
    """
    s = f"{value:.15g}"

    if "." in s:
        # Strip trailing zeros from the fractional part, then take last char.
        integer_part, frac_part = s.split(".", 1)
        frac_stripped = frac_part.rstrip("0")
        if frac_stripped:
            return int(frac_stripped[-1])
        # All fractional digits were zero -> treat as integer
        s = integer_part

    # Integer path: take the last digit of the absolute value representation.
    s = s.lstrip("-")
    if s and s[-1].isdigit():
        return int(s[-1])
    return None


def terminal_digit_test(
    values: list[float],
    alpha: float = 0.01,
) -> TerminalDigitResult:
    """Test whether terminal (last significant) digits are uniformly distributed.

    Under authentic data the last significant digit of reported values should
    be approximately uniformly distributed over 0-9.  Fabricated data often
    shows clustering on certain digits (e.g. 0 and 5).

    Args:
        values: List of numerical values whose terminal digits will be tested.
        alpha: Significance threshold for the chi-squared test.

    Returns:
        TerminalDigitResult with chi-squared statistic and suspicion flag.
    """
    digits: list[int] = []
    for v in values:
        d = _extract_terminal_digit(v)
        if d is not None:
            digits.append(d)

    n_values = len(digits)
    digit_counts: dict[int, int] = {d: 0 for d in range(10)}
    for d in digits:
        digit_counts[d] += 1

    if n_values < 10:
        return TerminalDigitResult(
            suspicious=False,
            chi_squared=0.0,
            p_value=1.0,
            digit_counts=digit_counts,
            n_values=n_values,
            details="Too few values for terminal-digit analysis.",
        )

    observed = np.array([digit_counts[d] for d in range(10)], dtype=np.float64)
    expected = np.full(10, n_values / 10.0)

    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    chi2_stat = float(chi2_stat)
    p_value = float(p_value)

    suspicious = p_value < alpha

    return TerminalDigitResult(
        suspicious=suspicious,
        chi_squared=chi2_stat,
        p_value=p_value,
        digit_counts=digit_counts,
        n_values=n_values,
        details=(
            f"Terminal-digit chi-squared = {chi2_stat:.4f}, p = {p_value:.6f}. "
            f"{'Non-uniform distribution detected' if suspicious else 'Distribution consistent with uniformity'} "
            f"(alpha = {alpha})."
        ),
    )


# ---------------------------------------------------------------------------
# 2.8 Distribution fit tests
# ---------------------------------------------------------------------------


@dataclass
class DistributionFitResult:
    """Result of a distribution-fit (KS + Anderson-Darling) test."""

    suspicious: bool
    ks_statistic: float
    ks_p_value: float
    anderson_statistic: float | None
    anderson_critical: float | None
    best_fit: str
    details: str


def distribution_fit_test(
    values: list[float],
    distribution: str = "norm",
) -> DistributionFitResult:
    """Test whether *values* follow the specified distribution.

    Fits the distribution parameters via MLE, then applies both a
    Kolmogorov-Smirnov test and (where available) an Anderson-Darling test.

    Args:
        values: Sample data to test.
        distribution: Name of the scipy continuous distribution to test
            against (e.g. ``"norm"``, ``"expon"``).

    Returns:
        DistributionFitResult with test statistics and suspicion flag.
    """
    arr = np.asarray(values, dtype=np.float64)

    if len(arr) < 5:
        return DistributionFitResult(
            suspicious=False,
            ks_statistic=0.0,
            ks_p_value=1.0,
            anderson_statistic=None,
            anderson_critical=None,
            best_fit=distribution,
            details="Too few values for distribution-fit testing.",
        )

    # Fit distribution parameters (loc, scale, and any shape parameters).
    dist_obj = getattr(stats, distribution)
    params = dist_obj.fit(arr)

    # Kolmogorov-Smirnov test against the fitted distribution.
    ks_stat, ks_p = stats.kstest(arr, distribution, args=params)
    ks_stat = float(ks_stat)
    ks_p = float(ks_p)

    # Anderson-Darling test (only available for a subset of distributions).
    anderson_stat: float | None = None
    anderson_crit: float | None = None
    anderson_exceeds = False

    # scipy.stats.anderson supports: 'norm', 'expon', 'logistic', 'gumbel',
    # 'gumbel_l', 'gumbel_r'.  Map our distribution name to the anderson
    # dist parameter where possible.
    _anderson_map = {
        "norm": "norm",
        "expon": "expon",
        "logistic": "logistic",
        "gumbel_r": "gumbel_r",
        "gumbel_l": "gumbel_l",
        "gumbel": "gumbel",
    }

    anderson_dist = _anderson_map.get(distribution)
    if anderson_dist is not None:
        try:
            # Anderson-Darling expects raw data; it fits internally.
            ad_result = stats.anderson(arr, dist=anderson_dist)
            anderson_stat = float(ad_result.statistic)
            # Critical values are at [15%, 10%, 5%, 2.5%, 1%] significance.
            # We use the 5% level (index 2).
            if len(ad_result.critical_values) > 2:
                anderson_crit = float(ad_result.critical_values[2])
                anderson_exceeds = anderson_stat > anderson_crit
        except Exception:
            pass  # Anderson-Darling not available for this distribution

    # Flag suspicious if KS p-value < 0.01 AND Anderson statistic exceeds
    # critical value at the 5% level (when available).  If Anderson is not
    # available we rely solely on the KS test.
    if anderson_stat is not None and anderson_crit is not None:
        suspicious = ks_p < 0.01 and anderson_exceeds
    else:
        suspicious = ks_p < 0.01

    return DistributionFitResult(
        suspicious=suspicious,
        ks_statistic=ks_stat,
        ks_p_value=ks_p,
        anderson_statistic=anderson_stat,
        anderson_critical=anderson_crit,
        best_fit=distribution,
        details=(
            f"KS test: D = {ks_stat:.4f}, p = {ks_p:.6f}. "
            + (
                f"Anderson-Darling: A^2 = {anderson_stat:.4f}, "
                f"critical (5%) = {anderson_crit:.4f}. "
                if anderson_stat is not None and anderson_crit is not None
                else "Anderson-Darling: not available for this distribution. "
            )
            + f"Data {'does NOT fit' if suspicious else 'consistent with'} "
            f"{distribution} distribution."
        ),
    )


# ---------------------------------------------------------------------------
# 2.9 Variance ratio consistency test (Simonsohn 2013)
# ---------------------------------------------------------------------------


@dataclass
class VarianceRatioResult:
    """Result of a variance-ratio consistency test across reported SDs."""

    suspicious: bool
    n_groups: int
    observed_variance_of_sds: float
    expected_variance_of_sds: float
    ratio: float
    p_value: float
    details: str


def variance_ratio_test(
    sd_n_pairs: list[tuple[float, int]],
    alpha: float = 0.05,
) -> VarianceRatioResult:
    """Test whether a set of reported SDs show realistic sampling variability.

    When multiple groups are assumed to share a common population variance,
    the sample SDs should exhibit a predictable amount of variability.
    Fabricated data often shows SDs that are implausibly similar (low
    variability) because fabricators tend to copy or slightly tweak a single
    SD value (Simonsohn, 2013).

    Args:
        sd_n_pairs: List of ``(reported_sd, sample_size)`` tuples, one per
            group / study.
        alpha: Significance threshold (one-tailed, testing for *too little*
            variability).

    Returns:
        VarianceRatioResult with the ratio of observed to expected variance
        of the sample SDs and a suspicion flag.
    """
    if len(sd_n_pairs) < 3:
        return VarianceRatioResult(
            suspicious=False,
            n_groups=len(sd_n_pairs),
            observed_variance_of_sds=0.0,
            expected_variance_of_sds=0.0,
            ratio=1.0,
            p_value=1.0,
            details="Need at least 3 (SD, N) pairs for the variance-ratio test.",
        )

    sds = np.array([s for s, _ in sd_n_pairs], dtype=np.float64)
    ns = np.array([n for _, n in sd_n_pairs], dtype=np.float64)

    k = len(sds)  # number of groups

    # Pool the SDs to estimate the common population SD (sigma).
    # Weighted by degrees of freedom: pooled_var = sum((n_i - 1)*s_i^2) / sum(n_i - 1)
    dfs = ns - 1.0
    pooled_var = np.sum(dfs * sds**2) / np.sum(dfs)
    sigma = math.sqrt(pooled_var)

    # Observed variance of the reported SDs
    observed_var = float(np.var(sds, ddof=1))

    # Expected variance of sample SDs under the null.
    # For a normal population, Var(s) ~ sigma^2 / (2 * n_i) approximately
    # (using the asymptotic result for the sample standard deviation).
    # The expected variance of the set of SDs is the average of these.
    expected_var = float(np.mean(sigma**2 / (2.0 * ns)))

    # Ratio: values much less than 1 indicate suspiciously low variability.
    ratio = observed_var / expected_var if expected_var > 0 else float("inf")

    # Under the null, (k - 1) * observed_var / expected_var is approximately
    # chi-squared with (k - 1) degrees of freedom.  We perform a one-tailed
    # test for *too little* variability (left tail).
    chi2_stat = (k - 1) * ratio
    p_value = float(stats.chi2.cdf(chi2_stat, df=k - 1))

    suspicious = p_value < alpha

    return VarianceRatioResult(
        suspicious=suspicious,
        n_groups=k,
        observed_variance_of_sds=observed_var,
        expected_variance_of_sds=expected_var,
        ratio=ratio,
        p_value=p_value,
        details=(
            f"{k} groups, pooled sigma = {sigma:.4f}. "
            f"Observed Var(SDs) = {observed_var:.6f}, "
            f"expected Var(SDs) = {expected_var:.6f}, "
            f"ratio = {ratio:.4f}. "
            f"Chi-squared({k - 1}) = {chi2_stat:.4f}, p = {p_value:.6f}. "
            f"SDs are {'suspiciously similar (possible fabrication)' if suspicious else 'consistent with sampling variability'}."
        ),
    )
