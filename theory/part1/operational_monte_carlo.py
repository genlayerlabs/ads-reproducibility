"""Monte Carlo calibration and power study for the operational certificate.

This synthetic experiment checks that the reported lower bounds rarely exceed
known population coverage and shows how power changes with A, M, and shared-row
dependence.  It is a diagnostic for the finite-sample theorem, not evidence that
the chosen clarity distribution describes real LLMs.

Example:
    python operational_monte_carlo.py --repetitions 500 --correlations 0,0.9
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path

import numpy as np
from scipy import stats

from operational_certificate import directed_binomial_errors, quota


# A deliberately heterogeneous generator population on both sides of tau=1/2.
DEFAULT_P_VALUES = np.array([0.30, 0.40, 0.46, 0.49, 0.51, 0.54, 0.60, 0.70])
DEFAULT_WEIGHTS = np.array([0.15, 0.20, 0.05, 0.10, 0.10, 0.05, 0.20, 0.15])
DEFAULT_PANEL_SIZES = np.array([21, 41, 61, 81, 101, 151, 201, 301])


@dataclass(frozen=True)
class ScenarioResult:
    """Complete output of one calibration-and-power scenario."""

    generator_columns: int
    evaluator_rows: int
    repetitions: int
    correlation_mixture: float
    tau: float
    delta: float
    beta: float
    eta_e: float
    eta_g: float
    xi_e: float
    seed: int
    panel_sizes: np.ndarray
    true_coverage: np.ndarray
    # The exact one-sided binomial inversion is the primary exterior bound.
    lower_bounds: np.ndarray
    hoeffding_lower_bounds: np.ndarray
    selected_panel_sizes: np.ndarray
    hoeffding_selected_panel_sizes: np.ndarray
    generator_correction: float

    @property
    def violation_mask(self) -> np.ndarray:
        """Whether any exact simultaneous lower bound exceeds true coverage."""
        return np.any(
            self.lower_bounds > self.true_coverage + 1e-12,
            axis=1,
        )

    @property
    def hoeffding_violation_mask(self) -> np.ndarray:
        """Whether any Hoeffding lower bound exceeds true coverage."""
        return np.any(
            self.hoeffding_lower_bounds > self.true_coverage + 1e-12,
            axis=1,
        )

    @property
    def pass_mask(self) -> np.ndarray:
        """Whether the exact bound certifies the predeclared target."""
        return self.selected_panel_sizes >= 0

    @property
    def hoeffding_pass_mask(self) -> np.ndarray:
        """Whether the Hoeffding bound certifies the predeclared target."""
        return self.hoeffding_selected_panel_sizes >= 0


def _clopper_pearson_lookup(m: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    counts = np.arange(m + 1)
    lower = np.zeros(m + 1)
    upper = np.ones(m + 1)
    positive = counts > 0
    below_full = counts < m
    lower[positive] = stats.beta.ppf(
        alpha / 2,
        counts[positive],
        m - counts[positive] + 1,
    )
    upper[below_full] = stats.beta.ppf(
        1 - alpha / 2,
        counts[below_full] + 1,
        m - counts[below_full],
    )
    return lower, upper


def _one_sided_cp_lower_lookup(n: int, alpha: float) -> np.ndarray:
    """Lookup table for a one-sided exact binomial lower bound."""
    counts = np.arange(n + 1)
    lower = np.zeros(n + 1)
    positive = counts > 0
    lower[positive] = stats.beta.ppf(
        alpha,
        counts[positive],
        n - counts[positive] + 1,
    )
    return lower


def _true_coverages(
    p_values: np.ndarray,
    weights: np.ndarray,
    panel_sizes: np.ndarray,
    tau: float,
    delta: float,
) -> np.ndarray:
    coverages = []
    for k in panel_sizes:
        errors = []
        for p in p_values:
            false_rejection, false_acceptance = directed_binomial_errors(
                float(p), int(k), tau
            )
            errors.append(false_rejection if p >= tau else false_acceptance)
        coverages.append(float(np.sum(weights * (np.asarray(errors) <= delta))))
    return np.asarray(coverages)


def _shared_row_counts(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    m: int,
    correlation_mixture: float,
) -> np.ndarray:
    """Generate binomial marginals with controllable dependence across columns.

    On a common row, every column thresholds the same uniform draw.  On an
    idiosyncratic row, columns use independent uniforms.  Rows remain i.i.d.,
    and every column count remains exactly Binomial(m, p_a).
    """
    common_rows = int(rng.binomial(m, correlation_mixture))
    if common_rows:
        common_uniforms = np.sort(rng.random(common_rows))
        common_counts = np.searchsorted(
            common_uniforms,
            probabilities,
            side="right",
        )
    else:
        common_counts = np.zeros(len(probabilities), dtype=int)
    independent_counts = rng.binomial(m - common_rows, probabilities)
    return common_counts + independent_counts


def _vectorized_interval_certifies(
    lower: np.ndarray,
    upper: np.ndarray,
    panel_size: int,
    tau: float,
    delta: float,
) -> np.ndarray:
    """Vectorized equivalent of operational_certificate.interval_certifies."""
    q = quota(panel_size, tau)
    accept_flags = (lower >= tau) & (stats.binom.cdf(q - 1, panel_size, lower) <= delta)
    reject_flags = (upper < tau) & (stats.binom.sf(q - 1, panel_size, upper) <= delta)
    return accept_flags | reject_flags


def run_scenario(
    *,
    generator_columns: int,
    evaluator_rows: int,
    repetitions: int,
    correlation_mixture: float,
    tau: float,
    delta: float,
    beta: float,
    eta_e: float,
    eta_g: float,
    xi_e: float,
    seed: int,
) -> ScenarioResult:
    if not 0 <= correlation_mixture <= 1:
        raise ValueError("correlation_mixture must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    panel_sizes = DEFAULT_PANEL_SIZES
    true_coverage = _true_coverages(
        DEFAULT_P_VALUES,
        DEFAULT_WEIGHTS,
        panel_sizes,
        tau,
        delta,
    )
    interval_alpha = eta_e * xi_e
    lower_lookup, upper_lookup = _clopper_pearson_lookup(
        evaluator_rows,
        interval_alpha,
    )
    epsilon_g = sqrt(log(len(panel_sizes) / eta_g) / (2 * generator_columns))
    exact_generator_lookup = _one_sided_cp_lower_lookup(
        generator_columns,
        eta_g / len(panel_sizes),
    )

    lower_bounds = np.empty((repetitions, len(panel_sizes)))
    hoeffding_lower_bounds = np.empty((repetitions, len(panel_sizes)))
    selected = np.full(repetitions, -1, dtype=int)
    hoeffding_selected = np.full(repetitions, -1, dtype=int)
    for repetition in range(repetitions):
        p_columns = rng.choice(
            DEFAULT_P_VALUES,
            size=generator_columns,
            p=DEFAULT_WEIGHTS,
        )
        counts = _shared_row_counts(
            rng,
            p_columns,
            evaluator_rows,
            correlation_mixture,
        )
        lower = lower_lookup[counts]
        upper = upper_lookup[counts]

        for index, k in enumerate(panel_sizes):
            certificate_flags = _vectorized_interval_certifies(
                lower,
                upper,
                int(k),
                tau,
                delta,
            )
            certified_count = int(np.sum(certificate_flags))
            certified_fraction = certified_count / generator_columns
            lower_bounds[repetition, index] = max(
                0.0,
                exact_generator_lookup[certified_count] - xi_e,
            )
            hoeffding_lower_bounds[repetition, index] = max(
                0.0,
                certified_fraction - xi_e - epsilon_g,
            )

        passing = np.flatnonzero(lower_bounds[repetition] >= 1 - beta)
        if passing.size:
            selected[repetition] = int(panel_sizes[passing[0]])
        hoeffding_passing = np.flatnonzero(
            hoeffding_lower_bounds[repetition] >= 1 - beta
        )
        if hoeffding_passing.size:
            hoeffding_selected[repetition] = int(panel_sizes[hoeffding_passing[0]])

    return ScenarioResult(
        generator_columns=generator_columns,
        evaluator_rows=evaluator_rows,
        repetitions=repetitions,
        correlation_mixture=correlation_mixture,
        tau=tau,
        delta=delta,
        beta=beta,
        eta_e=eta_e,
        eta_g=eta_g,
        xi_e=xi_e,
        seed=seed,
        panel_sizes=panel_sizes.copy(),
        true_coverage=true_coverage,
        lower_bounds=lower_bounds,
        hoeffding_lower_bounds=hoeffding_lower_bounds,
        selected_panel_sizes=selected,
        hoeffding_selected_panel_sizes=hoeffding_selected,
        generator_correction=epsilon_g,
    )


def print_scenario(result: ScenarioResult) -> None:
    """Print one scenario in a compact, machine-readable form."""
    print(
        f"rho_mix={result.correlation_mixture:.2f}  "
        f"A={result.generator_columns}  M={result.evaluator_rows}  "
        f"repeats={result.repetitions}"
    )
    print(
        f"nominal simultaneous failure <= {result.eta_e + result.eta_g:.3f}; "
        f"exact/Hoeffding violations="
        f"{np.mean(result.violation_mask):.4f}/"
        f"{np.mean(result.hoeffding_violation_mask):.4f}; "
        f"exact/Hoeffding pass rate={np.mean(result.pass_mask):.4f}/"
        f"{np.mean(result.hoeffding_pass_mask):.4f}"
    )
    print("K,true_R,mean_exact,p05_exact,p95_exact,mean_hoeffding")
    for index, k in enumerate(result.panel_sizes):
        print(
            f"{k},{result.true_coverage[index]:.3f},"
            f"{np.mean(result.lower_bounds[:, index]):.3f},"
            f"{np.quantile(result.lower_bounds[:, index], 0.05):.3f},"
            f"{np.quantile(result.lower_bounds[:, index], 0.95):.3f},"
            f"{np.mean(result.hoeffding_lower_bounds[:, index]):.3f}"
        )
    chosen = result.selected_panel_sizes[result.pass_mask]
    if chosen.size:
        values, counts = np.unique(chosen, return_counts=True)
        frequencies = ", ".join(
            f"K={value}: {count / result.repetitions:.3f}"
            for value, count in zip(values, counts)
        )
        print(f"selected frequencies: {frequencies}")
    else:
        print("selected frequencies: none")


def _write_study_csv(
    labelled_results: list[tuple[str, ScenarioResult]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "operational_monte_carlo_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "design",
                "A",
                "M",
                "rho_mix",
                "repetitions",
                "seed",
                "tau",
                "delta",
                "beta",
                "eta_E",
                "eta_G",
                "xi_E",
                "exact_violations",
                "exact_violation_rate",
                "exact_certificate_passes",
                "exact_certificate_pass_rate",
                "hoeffding_violations",
                "hoeffding_violation_rate",
                "hoeffding_certificate_passes",
                "hoeffding_certificate_pass_rate",
                "epsilon_G",
                "mean_exact_lower_at_K_max",
                "p05_exact_lower_at_K_max",
                "p95_exact_lower_at_K_max",
                "mean_hoeffding_lower_at_K_max",
                "p05_hoeffding_lower_at_K_max",
                "p95_hoeffding_lower_at_K_max",
            ]
        )
        for label, result in labelled_results:
            last = result.lower_bounds[:, -1]
            hoeffding_last = result.hoeffding_lower_bounds[:, -1]
            writer.writerow(
                [
                    label,
                    result.generator_columns,
                    result.evaluator_rows,
                    f"{result.correlation_mixture:.2f}",
                    result.repetitions,
                    result.seed,
                    f"{result.tau:.8f}",
                    f"{result.delta:.8f}",
                    f"{result.beta:.8f}",
                    f"{result.eta_e:.8f}",
                    f"{result.eta_g:.8f}",
                    f"{result.xi_e:.8f}",
                    int(np.sum(result.violation_mask)),
                    f"{np.mean(result.violation_mask):.8f}",
                    int(np.sum(result.pass_mask)),
                    f"{np.mean(result.pass_mask):.8f}",
                    int(np.sum(result.hoeffding_violation_mask)),
                    f"{np.mean(result.hoeffding_violation_mask):.8f}",
                    int(np.sum(result.hoeffding_pass_mask)),
                    f"{np.mean(result.hoeffding_pass_mask):.8f}",
                    f"{result.generator_correction:.8f}",
                    f"{np.mean(last):.8f}",
                    f"{np.quantile(last, 0.05):.8f}",
                    f"{np.quantile(last, 0.95):.8f}",
                    f"{np.mean(hoeffding_last):.8f}",
                    f"{np.quantile(hoeffding_last, 0.05):.8f}",
                    f"{np.quantile(hoeffding_last, 0.95):.8f}",
                ]
            )

    curves_path = output_dir / "operational_monte_carlo_curves.csv"
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "design",
                "A",
                "M",
                "rho_mix",
                "seed",
                "K",
                "true_coverage",
                "mean_exact_lower",
                "p05_exact_lower",
                "p95_exact_lower",
                "mean_hoeffding_lower",
                "p05_hoeffding_lower",
                "p95_hoeffding_lower",
            ]
        )
        for label, result in labelled_results:
            for index, k in enumerate(result.panel_sizes):
                bounds = result.lower_bounds[:, index]
                hoeffding_bounds = result.hoeffding_lower_bounds[:, index]
                writer.writerow(
                    [
                        label,
                        result.generator_columns,
                        result.evaluator_rows,
                        f"{result.correlation_mixture:.2f}",
                        result.seed,
                        int(k),
                        f"{result.true_coverage[index]:.8f}",
                        f"{np.mean(bounds):.8f}",
                        f"{np.quantile(bounds, 0.05):.8f}",
                        f"{np.quantile(bounds, 0.95):.8f}",
                        f"{np.mean(hoeffding_bounds):.8f}",
                        f"{np.quantile(hoeffding_bounds, 0.05):.8f}",
                        f"{np.quantile(hoeffding_bounds, 0.95):.8f}",
                    ]
                )


def _plot_paper_study(
    labelled_results: list[tuple[str, ScenarioResult]],
    figure_path: Path,
) -> None:
    # Matplotlib is imported lazily so the numerical test suite needs only
    # NumPy and SciPy.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)

    for ax, labelled_result in zip(axes.flat, labelled_results):
        label, result = labelled_result
        means = np.mean(result.lower_bounds, axis=0)
        p05 = np.quantile(result.lower_bounds, 0.05, axis=0)
        p95 = np.quantile(result.lower_bounds, 0.95, axis=0)
        ax.fill_between(
            result.panel_sizes,
            p05,
            p95,
            color="#2176AE",
            alpha=0.20,
            label="Monte Carlo 5--95% range",
        )
        ax.plot(
            result.panel_sizes,
            means,
            color="#2176AE",
            marker="o",
            markersize=4,
            label="Mean exact lower bound",
        )
        ax.plot(
            result.panel_sizes,
            np.mean(result.hoeffding_lower_bounds, axis=0),
            color="#E07A1F",
            linestyle="-.",
            marker="^",
            markersize=3.5,
            label="Mean Hoeffding lower bound",
        )
        ax.plot(
            result.panel_sizes,
            result.true_coverage,
            color="#202020",
            linestyle="--",
            marker="s",
            markersize=3.5,
            label="True population coverage",
        )
        ax.axhline(
            1 - result.beta,
            color="#D1495B",
            linestyle=":",
            label=r"Target $1-\beta$",
        )
        if result.correlation_mixture == 0:
            dependence = "independent columns"
        else:
            dependence = (
                "shared-row mixture " + rf"$\rho={result.correlation_mixture:.1f}$"
            )
        ax.set_title(
            f"{label.capitalize()} design: "
            + rf"$A={result.generator_columns}$, $M={result.evaluator_rows}$"
            + f"\n{dependence}"
        )
        ax.text(
            0.03,
            0.96,
            (
                f"violations: {int(np.sum(result.violation_mask))}/"
                f"{result.repetitions}\n"
                f"target certified (exact/H): "
                f"{100 * np.mean(result.pass_mask):.1f}%/"
                f"{100 * np.mean(result.hoeffding_pass_mask):.1f}%"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#AAAAAA",
                "alpha": 0.9,
            },
        )
        ax.set_ylim(-0.02, 0.78)
        ax.set_xticks(result.panel_sizes)
        ax.tick_params(axis="x", rotation=35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Candidate panel size $K$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Population coverage")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_paper_study(
    *,
    repetitions: int,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
) -> list[tuple[str, ScenarioResult]]:
    """Run and persist the four predeclared scenarios reported in the paper."""
    labelled_results: list[tuple[str, ScenarioResult]] = []
    designs = [
        ("baseline", 500, 1500),
        ("powered", 2000, 3000),
    ]
    correlations = [0.0, 0.9]
    for design_index, design in enumerate(designs):
        label, generator_columns, evaluator_rows = design
        for correlation_index, correlation in enumerate(correlations):
            result = run_scenario(
                generator_columns=generator_columns,
                evaluator_rows=evaluator_rows,
                repetitions=repetitions,
                correlation_mixture=correlation,
                tau=0.5,
                delta=0.05,
                beta=0.40,
                eta_e=0.025,
                eta_g=0.025,
                xi_e=0.05,
                seed=seed + 100 * design_index + correlation_index,
            )
            labelled_results.append((label, result))
            print_scenario(result)
            print()

    _write_study_csv(labelled_results, results_dir)
    _plot_paper_study(
        labelled_results,
        figures_dir / "fig4_operational_monte_carlo.png",
    )
    return labelled_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-columns", type=int, default=500)
    parser.add_argument("--evaluator-rows", type=int, default=1500)
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--correlations", default="0,0.9")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.40)
    parser.add_argument("--eta-e", type=float, default=0.025)
    parser.add_argument("--eta-g", type=float, default=0.025)
    parser.add_argument("--xi-e", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=5510)
    parser.add_argument(
        "--paper-study",
        action="store_true",
        help="run the four fixed scenarios and write the paper artifacts",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
    )
    args = parser.parse_args()

    if args.paper_study:
        run_paper_study(
            repetitions=args.repetitions,
            seed=args.seed,
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
        )
        return

    correlations = [float(value) for value in args.correlations.split(",")]
    for index, correlation in enumerate(correlations):
        if index:
            print()
        result = run_scenario(
            generator_columns=args.generator_columns,
            evaluator_rows=args.evaluator_rows,
            repetitions=args.repetitions,
            correlation_mixture=correlation,
            tau=args.tau,
            delta=args.delta,
            beta=args.beta,
            eta_e=args.eta_e,
            eta_g=args.eta_g,
            xi_e=args.xi_e,
            seed=args.seed + index,
        )
        print_scenario(result)


if __name__ == "__main__":
    main()
