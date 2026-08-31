"""Numerical invariants for the operational finite-sample certificate."""

from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np
from scipy import stats

from operational_certificate import (
    clopper_pearson_interval,
    clopper_pearson_lower,
    directed_binomial_errors,
    interval_certifies,
    mass_controlled_certificate,
    operational_certificate,
    quota,
)
from operational_monte_carlo import (
    _vectorized_interval_certifies,
    run_scenario,
)


class QuotaTests(unittest.TestCase):
    def test_decimal_threshold_has_no_float_boundary_drift(self) -> None:
        self.assertEqual(quota(100, 0.07), 7)
        self.assertEqual(quota(10, Fraction(1, 2)), 5)
        self.assertEqual(quota(3, "0.5"), 2)


class ClopperPearsonTests(unittest.TestCase):
    def test_grid_coverage_is_at_least_nominal(self) -> None:
        sample_size = 20
        alpha = 0.05
        intervals = [
            clopper_pearson_interval(c, sample_size, alpha)
            for c in range(sample_size + 1)
        ]
        for p in np.linspace(0.0, 1.0, 101):
            coverage = sum(
                stats.binom.pmf(c, sample_size, p)
                for c, (lower, upper) in enumerate(intervals)
                if lower - 1e-14 <= p <= upper + 1e-14
            )
            self.assertGreaterEqual(coverage, 1 - alpha - 1e-12)

    def test_one_sided_lower_bound_has_nominal_coverage(self) -> None:
        sample_size = 20
        alpha = 0.025
        bounds = [
            clopper_pearson_lower(c, sample_size, alpha) for c in range(sample_size + 1)
        ]
        for p in np.linspace(0.0, 1.0, 101):
            coverage = sum(
                stats.binom.pmf(c, sample_size, p)
                for c, lower in enumerate(bounds)
                if lower <= p + 1e-14
            )
            self.assertGreaterEqual(coverage, 1 - alpha - 1e-12)


class PointwiseCertificateTests(unittest.TestCase):
    def test_every_certified_grid_interval_is_sound(self) -> None:
        delta = 0.1
        tau = 0.5
        endpoints = np.linspace(0.0, 1.0, 21)
        for k in range(1, 31):
            for lower in endpoints:
                for upper in endpoints[endpoints >= lower]:
                    if not interval_certifies(lower, upper, k, tau, delta):
                        continue
                    for p in np.linspace(lower, upper, 31):
                        false_rejection, false_acceptance = directed_binomial_errors(
                            p, k, tau
                        )
                        error = false_rejection if p >= tau else false_acceptance
                        self.assertLessEqual(error, delta + 1e-12)

    def test_threshold_crossing_interval_is_not_certified(self) -> None:
        self.assertFalse(interval_certifies(0.49, 0.51, 101, 0.5, 0.05))

    def test_vectorized_monte_carlo_rule_matches_scalar_rule(self) -> None:
        intervals = [
            (0.0, 0.0),
            (0.0, 0.49),
            (0.35, 0.49),
            (0.49, 0.50),
            (0.49, 0.51),
            (0.50, 0.50),
            (0.50, 0.65),
            (0.51, 1.0),
            (1.0, 1.0),
        ]
        lower = np.asarray([bounds[0] for bounds in intervals])
        upper = np.asarray([bounds[1] for bounds in intervals])

        for k in (1, 2, 21, 101):
            for delta in (0.01, 0.05, 0.20):
                expected = np.asarray(
                    [interval_certifies(lo, hi, k, 0.5, delta) for lo, hi in intervals]
                )
                actual = _vectorized_interval_certifies(
                    lower,
                    upper,
                    k,
                    0.5,
                    delta,
                )
                np.testing.assert_array_equal(actual, expected)


class EndToEndComputationTests(unittest.TestCase):
    def test_monte_carlo_scenario_is_reproducible_from_its_seed(self) -> None:
        arguments = {
            "generator_columns": 20,
            "evaluator_rows": 30,
            "repetitions": 4,
            "correlation_mixture": 0.9,
            "tau": 0.5,
            "delta": 0.05,
            "beta": 0.4,
            "eta_e": 0.025,
            "eta_g": 0.025,
            "xi_e": 0.05,
            "seed": 1234,
        }
        first = run_scenario(**arguments)
        second = run_scenario(**arguments)
        self.assertEqual(first.lower_bounds.shape, (4, 8))
        np.testing.assert_array_equal(first.lower_bounds, second.lower_bounds)
        np.testing.assert_array_equal(
            first.hoeffding_lower_bounds,
            second.hoeffding_lower_bounds,
        )
        np.testing.assert_array_equal(
            first.selected_panel_sizes,
            second.selected_panel_sizes,
        )

    def test_exact_and_hoeffding_generator_bounds_and_clipping(self) -> None:
        result = operational_certificate(
            column_counts=[200] * 200,
            evaluator_sample_size=200,
            panel_sizes=[101, 51, 101],
            tau=0.5,
            delta=0.01,
            eta_e=0.025,
            eta_g=0.025,
        )
        self.assertEqual([row.panel_size for row in result.panel_bounds], [51, 101])
        self.assertTrue(
            all(row.certified_columns == 200 for row in result.panel_bounds)
        )
        self.assertTrue(
            all(
                abs(
                    row.exact_lower_coverage
                    - clopper_pearson_lower(200, 200, 0.025 / 2)
                )
                < 1e-14
                for row in result.panel_bounds
            )
        )
        self.assertTrue(
            all(
                abs(row.hoeffding_lower_coverage - (1 - result.generator_correction))
                < 1e-14
                for row in result.panel_bounds
            )
        )
        self.assertTrue(
            all(
                row.lower_coverage == row.exact_lower_coverage
                for row in result.panel_bounds
            )
        )

    def test_ambiguous_columns_produce_zero_lower_bound(self) -> None:
        result = operational_certificate(
            column_counts=[50] * 40,
            evaluator_sample_size=100,
            panel_sizes=[25, 51],
            tau=0.5,
            delta=0.05,
            eta_e=0.025,
            eta_g=0.025,
        )
        self.assertTrue(all(row.certified_columns == 0 for row in result.panel_bounds))
        self.assertTrue(all(row.lower_coverage == 0 for row in result.panel_bounds))
        self.assertTrue(
            all(row.hoeffding_lower_coverage == 0 for row in result.panel_bounds)
        )

    def test_mass_controlled_version_uses_declared_coverage_charge(self) -> None:
        xi_e = 0.04
        result = mass_controlled_certificate(
            column_counts=[200] * 200,
            evaluator_sample_size=200,
            panel_sizes=[51, 101],
            tau=0.5,
            delta=0.01,
            eta_e=0.025,
            eta_g=0.025,
            xi_e=xi_e,
        )
        self.assertAlmostEqual(result.per_column_alpha, 0.025 * xi_e)
        self.assertEqual(result.evaluator_mass_correction, xi_e)
        self.assertTrue(
            all(
                abs(
                    row.hoeffding_lower_coverage
                    - (1 - xi_e - result.generator_correction)
                )
                < 1e-14
                for row in result.panel_bounds
            )
        )
        self.assertTrue(
            all(
                abs(
                    row.exact_lower_coverage
                    - (clopper_pearson_lower(200, 200, 0.025 / 2) - xi_e)
                )
                < 1e-14
                for row in result.panel_bounds
            )
        )


if __name__ == "__main__":
    unittest.main()
