"""Numerical invariants for the non-adaptive corruption curves."""

from __future__ import annotations

import unittest

import numpy as np

from adversarial_security_profiles import (
    GENLAYER_PANEL_SIZES,
    committee_capture_probability,
    fixed_share_attack_probability,
    maximum_certified_corruption,
    required_robust_margin,
    required_honest_clarity,
    required_honest_clarity_fixed_share,
    semantic_attack_probability,
)


class ImplementedScheduleTests(unittest.TestCase):
    def test_schedule_matches_pinned_fee_manager(self) -> None:
        self.assertEqual(
            GENLAYER_PANEL_SIZES,
            (
                5,
                7,
                11,
                13,
                23,
                25,
                47,
                49,
                95,
                97,
                191,
                193,
                383,
                385,
                767,
                769,
                1535,
                1537,
            ),
        )


class SecurityCurveTests(unittest.TestCase):
    def test_half_adversarial_membership_captures_with_probability_half(self) -> None:
        for panel_size in GENLAYER_PANEL_SIZES:
            self.assertAlmostEqual(
                committee_capture_probability(panel_size, 0.5),
                0.5,
                places=12,
            )

    def test_capture_probability_decreases_with_panel_size_below_half(self) -> None:
        probabilities = np.asarray(
            [
                committee_capture_probability(panel_size, 0.25)
                for panel_size in GENLAYER_PANEL_SIZES
            ]
        )
        self.assertTrue(np.all(np.diff(probabilities) < 0.0))

    def test_semantic_attack_crosses_half_at_population_boundary(self) -> None:
        for panel_size in GENLAYER_PANEL_SIZES:
            self.assertAlmostEqual(
                semantic_attack_probability(panel_size, 0.12, 0.12),
                0.5,
                places=12,
            )

    def test_semantic_attack_is_monotone_in_corruption(self) -> None:
        alpha = np.linspace(0.0, 0.30, 301)
        probabilities = semantic_attack_probability(191, alpha, 0.20)
        self.assertTrue(np.all(np.diff(probabilities) >= -1e-15))

    def test_fixed_share_crosses_half_on_curved_population_boundary(self) -> None:
        for panel_size in GENLAYER_PANEL_SIZES:
            alpha = 0.30
            boundary_clarity = alpha / (2.0 * (1.0 - alpha))
            self.assertAlmostEqual(
                fixed_share_attack_probability(
                    panel_size,
                    alpha,
                    boundary_clarity,
                ),
                0.5,
                places=12,
            )

    def test_targeted_contamination_is_at_least_as_strong_as_fixed_share(self) -> None:
        alpha = np.linspace(0.0, 0.5, 101)
        gamma = np.linspace(0.0, 0.5, 101)
        targeted = semantic_attack_probability(191, alpha, gamma)
        fixed_share = fixed_share_attack_probability(191, alpha, gamma)
        self.assertTrue(np.all(targeted >= fixed_share - 1e-15))

    def test_margin_floor_hits_declared_error(self) -> None:
        delta = 0.01
        for panel_size in GENLAYER_PANEL_SIZES:
            margin = required_robust_margin(panel_size, delta)
            self.assertAlmostEqual(
                semantic_attack_probability(panel_size, 0.0, margin),
                delta,
                places=11,
            )

    def test_larger_panels_certify_smaller_margins(self) -> None:
        margins = np.asarray(
            [required_robust_margin(panel_size, 0.01) for panel_size in GENLAYER_PANEL_SIZES]
        )
        self.assertTrue(np.all(np.diff(margins) < 0.0))

    def test_certified_corruption_is_clarity_minus_margin_floor(self) -> None:
        panel_size = 383
        delta = 0.01
        margin = required_robust_margin(panel_size, delta)
        self.assertAlmostEqual(
            maximum_certified_corruption(panel_size, margin + 0.10, delta),
            0.10,
            places=12,
        )
        self.assertTrue(
            np.isnan(maximum_certified_corruption(panel_size, margin / 2.0, delta))
        )

    def test_required_clarity_is_corruption_plus_margin_floor(self) -> None:
        panel_size = 1537
        delta = 0.01
        alpha = np.asarray([0.0, 0.05, 0.20])
        expected = alpha + required_robust_margin(panel_size, delta)
        np.testing.assert_allclose(
            required_honest_clarity(panel_size, alpha, delta),
            expected,
            rtol=0.0,
            atol=1e-14,
        )

    def test_fixed_share_required_clarity_hits_declared_error(self) -> None:
        panel_size = 1537
        delta = 0.01
        for alpha in (0.0, 0.10, 0.30):
            clarity = required_honest_clarity_fixed_share(
                panel_size,
                alpha,
                delta,
            )
            self.assertAlmostEqual(
                fixed_share_attack_probability(panel_size, alpha, clarity),
                delta,
                places=11,
            )

    def test_fixed_share_example_at_thirty_percent(self) -> None:
        probability = fixed_share_attack_probability(1537, 0.30, 0.30)
        self.assertAlmostEqual(probability, 1.165447233636577e-6, places=14)

    def test_phase_diagram_probabilities_are_symmetric_about_alpha_equals_gamma(self) -> None:
        panel_size = 1537
        delta = 0.01
        margin = required_robust_margin(panel_size, delta)
        safe_probability = semantic_attack_probability(
            panel_size,
            0.20,
            0.20 + margin,
        )
        successful_probability = semantic_attack_probability(
            panel_size,
            0.20,
            0.20 - margin,
        )
        self.assertAlmostEqual(safe_probability, delta, places=11)
        self.assertAlmostEqual(successful_probability, 1.0 - delta, places=11)
        self.assertAlmostEqual(
            safe_probability + successful_probability,
            1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
