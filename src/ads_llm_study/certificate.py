from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import log, sqrt

import numpy as np
from scipy import optimize, stats

Threshold = float | str | Fraction


def _threshold_fraction(threshold: Threshold) -> Fraction:
    value = threshold if isinstance(threshold, Fraction) else Fraction(str(threshold))
    if not 0 < value < 1:
        raise ValueError("threshold must lie in (0, 1)")
    return value


def quota(panel_size: int, threshold: Threshold) -> int:
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    # Convert through the decimal representation so an exact boundary such as
    # 100 * 0.07 is not promoted from quota 7 to quota 8 by binary floating
    # point (on CPython, that product is 7.000000000000001).
    scaled = _threshold_fraction(threshold) * panel_size
    return -(-scaled.numerator // scaled.denominator)


def clopper_pearson(count: int, trials: int, alpha: float) -> tuple[float, float]:
    if not 0 <= count <= trials or trials < 1:
        raise ValueError("count must lie in [0, trials] and trials must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    lower = 0.0 if count == 0 else float(stats.beta.ppf(alpha / 2, count, trials - count + 1))
    upper = (
        1.0
        if count == trials
        else float(stats.beta.ppf(1 - alpha / 2, count + 1, trials - count))
    )
    return lower, upper


def clopper_pearson_lower(count: int, trials: int, alpha: float) -> float:
    """One-sided exact binomial lower confidence bound."""
    if not 0 <= count <= trials or trials < 1:
        raise ValueError("count must lie in [0, trials] and trials must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return 0.0 if count == 0 else float(stats.beta.ppf(alpha, count, trials - count + 1))


def directed_binomial_errors(
    probability: float,
    panel_size: int,
    threshold: Threshold,
) -> tuple[float, float]:
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    q = quota(panel_size, threshold)
    false_rejection = float(stats.binom.cdf(q - 1, panel_size, probability))
    false_acceptance = float(stats.binom.sf(q - 1, panel_size, probability))
    return false_rejection, false_acceptance


def interval_certifies(
    lower: float,
    upper: float,
    panel_size: int,
    threshold: Threshold,
    delta: float,
) -> bool:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("interval endpoints must satisfy 0 <= lower <= upper <= 1")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    threshold_float = float(_threshold_fraction(threshold))
    if lower >= threshold_float:
        false_rejection, _ = directed_binomial_errors(lower, panel_size, threshold)
        return bool(false_rejection <= delta)
    if upper < threshold_float:
        _, false_acceptance = directed_binomial_errors(upper, panel_size, threshold)
        return bool(false_acceptance <= delta)
    return False


@dataclass(frozen=True)
class PanelCoverageBound:
    panel_size: int
    certified_columns: int
    certified_fraction: float
    exact_lower_coverage: float
    hoeffding_lower_coverage: float

    @property
    def lower_coverage(self) -> float:
        """Recommended bound; retained as an API alias for existing callers."""
        return self.exact_lower_coverage


@dataclass(frozen=True)
class OperationalCertificate:
    evaluator_intervals: tuple[tuple[float, float], ...]
    per_column_alpha: float
    evaluator_mass_correction: float
    generator_correction: float
    panel_bounds: tuple[PanelCoverageBound, ...]


def _build_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    threshold: Threshold,
    delta: float,
    per_column_alpha: float,
    evaluator_mass_correction: float,
    eta_g: float,
) -> OperationalCertificate:
    counts = tuple(int(count) for count in column_counts)
    candidates = tuple(sorted(set(int(size) for size in panel_sizes)))
    if not counts:
        raise ValueError("at least one workload column is required")
    if evaluator_sample_size < 1:
        raise ValueError("evaluator_sample_size must be positive")
    if not candidates or candidates[0] < 1:
        raise ValueError("panel_sizes must contain positive integers")
    if not 0 < per_column_alpha < 1:
        raise ValueError("per_column_alpha must lie in (0, 1)")
    if not 0 <= evaluator_mass_correction < 1:
        raise ValueError("evaluator_mass_correction must lie in [0, 1)")
    if not 0 < eta_g < 1:
        raise ValueError("eta_g must lie in (0, 1)")
    if any(count < 0 or count > evaluator_sample_size for count in counts):
        raise ValueError("every column count must lie in [0, evaluator_sample_size]")

    intervals = tuple(
        clopper_pearson(count, evaluator_sample_size, per_column_alpha) for count in counts
    )
    generator_tail_alpha = eta_g / len(candidates)
    correction = sqrt(log(len(candidates) / eta_g) / (2 * len(counts)))
    bounds = []
    for panel_size in candidates:
        certified = sum(
            interval_certifies(lower, upper, panel_size, threshold, delta)
            for lower, upper in intervals
        )
        fraction = certified / len(counts)
        exact_generator_lower = clopper_pearson_lower(
            certified,
            len(counts),
            generator_tail_alpha,
        )
        bounds.append(
            PanelCoverageBound(
                panel_size=panel_size,
                certified_columns=certified,
                certified_fraction=fraction,
                exact_lower_coverage=max(
                    0.0,
                    exact_generator_lower - evaluator_mass_correction,
                ),
                hoeffding_lower_coverage=max(
                    0.0,
                    fraction - evaluator_mass_correction - correction,
                ),
            )
        )
    return OperationalCertificate(
        evaluator_intervals=intervals,
        per_column_alpha=per_column_alpha,
        evaluator_mass_correction=evaluator_mass_correction,
        generator_correction=correction,
        panel_bounds=tuple(bounds),
    )


def familywise_operational_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    threshold: Threshold,
    delta: float,
    eta_e: float,
    eta_g: float,
) -> OperationalCertificate:
    if not 0 < eta_e < 1 or not 0 < eta_g < 1 or eta_e + eta_g >= 1:
        raise ValueError("eta_e and eta_g must be positive and sum to less than one")
    if not column_counts:
        raise ValueError("at least one workload column is required")
    return _build_certificate(
        column_counts,
        evaluator_sample_size,
        panel_sizes,
        threshold,
        delta,
        eta_e / len(column_counts),
        0.0,
        eta_g,
    )


def mass_controlled_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    threshold: Threshold,
    delta: float,
    eta_e: float,
    eta_g: float,
    xi_e: float,
) -> OperationalCertificate:
    if not 0 < eta_e < 1 or not 0 < eta_g < 1 or eta_e + eta_g >= 1:
        raise ValueError("eta_e and eta_g must be positive and sum to less than one")
    if not 0 < xi_e < 1:
        raise ValueError("xi_e must lie in (0, 1)")
    return _build_certificate(
        column_counts,
        evaluator_sample_size,
        panel_sizes,
        threshold,
        delta,
        eta_e * xi_e,
        xi_e,
        eta_g,
    )


def certification_boundaries(
    panel_size: int,
    threshold: float,
    delta: float,
) -> tuple[float, float]:
    """Return the reject-side and accept-side probability cutoffs.

    Probabilities below the first cutoff have false-acceptance probability at
    most ``delta``; probabilities above the second have false-rejection
    probability at most ``delta``. The interval between them is the pointwise
    uncertainty band for the declared finite panel.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    q = quota(panel_size, threshold)

    def false_acceptance_residual(probability: float) -> float:
        return float(stats.binom.sf(q - 1, panel_size, probability) - delta)

    def false_rejection_residual(probability: float) -> float:
        return float(stats.binom.cdf(q - 1, panel_size, probability) - delta)

    reject_boundary = (
        threshold
        if false_acceptance_residual(threshold) <= 0
        else float(
            optimize.brentq(
                false_acceptance_residual,
                0.0,
                threshold,
                xtol=1e-14,
                rtol=1e-14,
            )
        )
    )
    accept_boundary = (
        threshold
        if false_rejection_residual(threshold) <= 0
        else float(
            optimize.brentq(
                false_rejection_residual,
                threshold,
                1.0,
                xtol=1e-14,
                rtol=1e-14,
            )
        )
    )
    return reject_boundary, accept_boundary


def required_robust_margin(panel_size: int, delta: float) -> float:
    if panel_size % 2 == 0:
        raise ValueError("strict-majority security profiles require odd panels")

    def residual(margin: float) -> float:
        q = panel_size // 2 + 1
        return float(stats.binom.cdf(q - 1, panel_size, 0.5 + margin) - delta)

    return float(optimize.brentq(residual, 0.0, 0.5, xtol=1e-14, rtol=1e-14))


def fixed_share_attack_probability(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    honest_clarity: float | np.ndarray,
) -> float | np.ndarray:
    alpha = np.asarray(adversarial_fraction, dtype=float)
    gamma = np.asarray(honest_clarity, dtype=float)
    p = (1.0 - alpha) * (0.5 + gamma)
    result = stats.binom.cdf(panel_size // 2, panel_size, p)
    return float(result) if np.ndim(result) == 0 else result


def targeted_attack_probability(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    clarity: float | np.ndarray,
) -> float | np.ndarray:
    alpha = np.asarray(adversarial_fraction, dtype=float)
    gamma = np.asarray(clarity, dtype=float)
    p = np.clip(0.5 + gamma - alpha, 0.0, 1.0)
    result = stats.binom.cdf(panel_size // 2, panel_size, p)
    return float(result) if np.ndim(result) == 0 else result
