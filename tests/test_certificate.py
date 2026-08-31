from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from ads_llm_study.certificate import (
    certification_boundaries,
    clopper_pearson,
    clopper_pearson_lower,
    directed_binomial_errors,
    fixed_share_attack_probability,
    interval_certifies,
    mass_controlled_certificate,
    quota,
    targeted_attack_probability,
)
from ads_llm_study.paths import REPO_ROOT


def test_quota_uses_the_declared_decimal_threshold_exactly() -> None:
    assert quota(5, 0.5) == 3
    assert quota(7, 0.5) == 4
    assert quota(100, 0.07) == 7


def test_clopper_pearson_boundaries() -> None:
    lower, upper = clopper_pearson(0, 20, 0.05)
    assert lower == 0.0
    assert 0 < upper < 1
    lower, upper = clopper_pearson(20, 20, 0.05)
    assert 0 < lower < 1
    assert upper == 1.0
    assert clopper_pearson_lower(0, 20, 0.05) == 0.0
    assert 0 < clopper_pearson_lower(20, 20, 0.05) < 1


def test_interval_certificate_is_sound_at_endpoints() -> None:
    assert interval_certifies(0.80, 0.90, 21, 0.5, 0.05)
    assert interval_certifies(0.10, 0.20, 21, 0.5, 0.05)
    assert not interval_certifies(0.45, 0.55, 21, 0.5, 0.05)


def test_certification_boundaries_match_the_odd_majority_margin() -> None:
    reject, accept = certification_boundaries(1537, 0.5, 0.01)
    assert reject < 0.5 < accept
    assert 0.5 - reject == pytest.approx(accept - 0.5)
    assert interval_certifies(accept + 1e-12, min(1.0, accept + 0.01), 1537, 0.5, 0.01)
    assert interval_certifies(max(0.0, reject - 0.01), reject - 1e-12, 1537, 0.5, 0.01)


def test_two_attack_models_have_distinct_frontiers() -> None:
    alpha = 0.30
    curved_clarity = alpha / (2 * (1 - alpha))
    assert fixed_share_attack_probability(1537, alpha, curved_clarity) == pytest.approx(0.5)
    assert targeted_attack_probability(1537, alpha, alpha) == pytest.approx(0.5)
    assert fixed_share_attack_probability(1537, 0.30, 0.30) == pytest.approx(
        1.165447233636577e-6
    )


def test_reference_certificate_parity_when_paper_checkout_is_adjacent() -> None:
    reference_path = (
        REPO_ROOT.parent
        / "prob-bft-paper"
        / "papers"
        / "part1-foundations"
        / "simulations"
        / "operational_certificate.py"
    )
    if not reference_path.is_file():
        pytest.skip("adjacent paper checkout is unavailable")
    spec = importlib.util.spec_from_file_location("paper_certificate", reference_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for probability in np.linspace(0.05, 0.95, 19):
        assert directed_binomial_errors(probability, 21, 0.5) == pytest.approx(
            module.directed_binomial_errors(probability, 21, 0.5)
        )
    for interval in ((0.1, 0.2), (0.45, 0.55), (0.8, 0.9)):
        assert interval_certifies(*interval, 21, 0.5, 0.05) == module.interval_certifies(
            *interval, 21, 0.5, 0.05
        )

    arguments = {
        "column_counts": [0, 11, 23, 40],
        "evaluator_sample_size": 40,
        "panel_sizes": [5, 13, 49],
        "delta": 0.01,
        "eta_e": 0.025,
        "eta_g": 0.025,
        "xi_e": 0.05,
    }
    actual = mass_controlled_certificate(threshold=0.5, **arguments)
    expected = module.mass_controlled_certificate(tau=0.5, **arguments)
    for actual_interval, expected_interval in zip(
        actual.evaluator_intervals, expected.evaluator_intervals, strict=True
    ):
        assert actual_interval == pytest.approx(expected_interval)
    assert actual.per_column_alpha == expected.per_column_alpha
    assert actual.evaluator_mass_correction == expected.evaluator_mass_correction
    assert actual.generator_correction == pytest.approx(expected.generator_correction)
    for actual_bound, expected_bound in zip(
        actual.panel_bounds, expected.panel_bounds, strict=True
    ):
        assert actual_bound.panel_size == expected_bound.panel_size
        assert actual_bound.certified_columns == expected_bound.certified_columns
        assert actual_bound.certified_fraction == pytest.approx(
            expected_bound.certified_fraction
        )
        assert actual_bound.exact_lower_coverage == pytest.approx(
            expected_bound.exact_lower_coverage
        )
        assert actual_bound.hoeffding_lower_coverage == pytest.approx(
            expected_bound.hoeffding_lower_coverage
        )
        assert actual_bound.lower_coverage == pytest.approx(expected_bound.lower_coverage)
