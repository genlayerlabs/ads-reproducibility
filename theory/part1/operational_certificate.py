"""Finite-sample certificates from Part 1's resolvability section.

The module turns column counts from an i.i.d. generator--evaluator matrix into
simultaneous lower bounds on ideal population resolvability.  It deliberately
uses exact binomial tails and Clopper--Pearson intervals rather than a normal
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import log, sqrt
from typing import Sequence

from scipy import stats


Threshold = float | str | Fraction


def _threshold_fraction(tau: Threshold) -> Fraction:
    """Interpret a decimal threshold exactly, including values such as 0.07."""
    value = tau if isinstance(tau, Fraction) else Fraction(str(tau))
    if not 0 < value < 1:
        raise ValueError("tau must lie strictly between zero and one")
    return value


def quota(k: int, tau: Threshold) -> int:
    """Return the inclusive quota ceil(tau * k) without float-boundary drift."""
    if k < 1:
        raise ValueError("k must be positive")
    threshold = _threshold_fraction(tau)
    numerator = threshold.numerator * k
    return (numerator + threshold.denominator - 1) // threshold.denominator


def clopper_pearson_interval(
    positives: int,
    sample_size: int,
    alpha: float,
) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval with miscoverage at most alpha."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not 0 <= positives <= sample_size:
        raise ValueError("positives must lie between zero and sample_size")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")

    if positives == 0:
        lower = 0.0
    else:
        lower = float(
            stats.beta.ppf(
                alpha / 2,
                positives,
                sample_size - positives + 1,
            )
        )

    if positives == sample_size:
        upper = 1.0
    else:
        upper = float(
            stats.beta.ppf(
                1 - alpha / 2,
                positives + 1,
                sample_size - positives,
            )
        )
    return lower, upper


def clopper_pearson_lower(
    positives: int,
    sample_size: int,
    alpha: float,
) -> float:
    """One-sided exact binomial lower confidence bound.

    The returned value has lower-tail miscoverage at most ``alpha``.  It is
    used for the generator layer, where only a lower bound on the probability
    of a certified column is needed.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not 0 <= positives <= sample_size:
        raise ValueError("positives must lie between zero and sample_size")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if positives == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, positives, sample_size - positives + 1))


def directed_binomial_errors(
    p: float,
    k: int,
    tau: Threshold,
) -> tuple[float, float]:
    """Return false-rejection and false-acceptance tails at parameter p."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie between zero and one")
    q = quota(k, tau)
    false_rejection = float(stats.binom.cdf(q - 1, k, p))
    false_acceptance = float(stats.binom.sf(q - 1, k, p))
    return false_rejection, false_acceptance


def interval_certifies(
    lower: float,
    upper: float,
    k: int,
    tau: Threshold,
    delta: float,
) -> bool:
    """Apply the sound pointwise interval certificate."""
    if not 0 <= lower <= upper <= 1:
        raise ValueError("interval endpoints must satisfy 0 <= lower <= upper <= 1")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")

    threshold = float(_threshold_fraction(tau))
    if lower >= threshold:
        false_rejection, _ = directed_binomial_errors(lower, k, tau)
        return false_rejection <= delta
    if upper < threshold:
        _, false_acceptance = directed_binomial_errors(upper, k, tau)
        return false_acceptance <= delta
    return False


@dataclass(frozen=True)
class PanelCoverageBound:
    """One row of the simultaneous operational certificate."""

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
    """Intervals, common generator correction, and bounds for all candidates."""

    evaluator_intervals: tuple[tuple[float, float], ...]
    per_column_alpha: float
    evaluator_mass_correction: float
    generator_correction: float
    panel_bounds: tuple[PanelCoverageBound, ...]


def _build_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    tau: Threshold,
    delta: float,
    per_column_alpha: float,
    evaluator_mass_correction: float,
    eta_g: float,
) -> OperationalCertificate:
    counts = tuple(int(count) for count in column_counts)
    candidates = tuple(sorted(set(int(k) for k in panel_sizes)))
    if not counts:
        raise ValueError("at least one generator column is required")
    if not candidates or candidates[0] < 1:
        raise ValueError("panel_sizes must contain positive integers")
    if not 0 < per_column_alpha < 1:
        raise ValueError("per_column_alpha must lie strictly between zero and one")
    if not 0 <= evaluator_mass_correction < 1:
        raise ValueError("evaluator_mass_correction must lie in [0, 1)")
    if not 0 < eta_g < 1:
        raise ValueError("eta_g must lie strictly between zero and one")
    if any(count < 0 or count > evaluator_sample_size for count in counts):
        raise ValueError("every column count must be between zero and M")

    generator_count = len(counts)
    intervals = tuple(
        clopper_pearson_interval(count, evaluator_sample_size, per_column_alpha)
        for count in counts
    )
    generator_tail_alpha = eta_g / len(candidates)
    correction = sqrt(log(len(candidates) / eta_g) / (2 * generator_count))

    bounds: list[PanelCoverageBound] = []
    for k in candidates:
        certified = sum(
            interval_certifies(lower, upper, k, tau, delta)
            for lower, upper in intervals
        )
        fraction = certified / generator_count
        exact_generator_lower = clopper_pearson_lower(
            certified,
            generator_count,
            generator_tail_alpha,
        )
        bounds.append(
            PanelCoverageBound(
                panel_size=k,
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


def operational_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    tau: Threshold,
    delta: float,
    eta_e: float,
    eta_g: float,
) -> OperationalCertificate:
    """Compute the familywise certificate for i.i.d. generators.

    ``column_counts[a]`` is the positive count in generator column ``a``.
    The same evaluator rows may underlie all counts; column independence is
    neither checked nor required by the theorem.
    """
    generator_count = len(column_counts)
    if generator_count < 1:
        raise ValueError("at least one generator column is required")
    if not 0 < eta_e < 1 or not 0 < eta_g < 1 or eta_e + eta_g >= 1:
        raise ValueError("eta_e and eta_g must be positive and sum to less than one")
    return _build_certificate(
        column_counts=column_counts,
        evaluator_sample_size=evaluator_sample_size,
        panel_sizes=panel_sizes,
        tau=tau,
        delta=delta,
        per_column_alpha=eta_e / generator_count,
        evaluator_mass_correction=0.0,
        eta_g=eta_g,
    )


def mass_controlled_certificate(
    column_counts: Sequence[int],
    evaluator_sample_size: int,
    panel_sizes: Sequence[int],
    tau: Threshold,
    delta: float,
    eta_e: float,
    eta_g: float,
    xi_e: float,
) -> OperationalCertificate:
    """Compute the scalable certificate that charges interval-failure mass xi_e."""
    if not 0 < eta_e < 1 or not 0 < eta_g < 1 or eta_e + eta_g >= 1:
        raise ValueError("eta_e and eta_g must be positive and sum to less than one")
    if not 0 < xi_e < 1:
        raise ValueError("xi_e must lie strictly between zero and one")
    return _build_certificate(
        column_counts=column_counts,
        evaluator_sample_size=evaluator_sample_size,
        panel_sizes=panel_sizes,
        tau=tau,
        delta=delta,
        per_column_alpha=eta_e * xi_e,
        evaluator_mass_correction=xi_e,
        eta_g=eta_g,
    )
