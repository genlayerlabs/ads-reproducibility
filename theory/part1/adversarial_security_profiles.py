"""Non-adaptive corruption curves for the nominal GenLayer panel ladder.

The panel schedule is pinned to ``genlayer-consensus`` at revision
``8795606edaa84ac8b46bc6687d9a4d5ad9d25f57`` (2025-12-24),
``contracts/FeeManager.sol``.  The calculations are large-population i.i.d.
benchmarks.  A deployed-network claim must instead use the active validator
set, its stake weights, and the protocol's without-replacement selection law.

Usage:
    python adversarial_security_profiles.py
    python adversarial_security_profiles.py --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy import optimize, stats


SOURCE_REVISION = "8795606edaa84ac8b46bc6687d9a4d5ad9d25f57"
GENLAYER_PANEL_SIZES = (
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
)
NORMAL_PANEL_SIZES = GENLAYER_PANEL_SIZES[0::2]
APPEAL_PANEL_SIZES = GENLAYER_PANEL_SIZES[1::2]

SIMULATION_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SIMULATION_DIR / "figures"
RESULTS_DIR = SIMULATION_DIR / "results"

plt.rcParams.update(
    {
        "figure.dpi": 170,
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.7,
        "axes.grid": True,
        "grid.alpha": 0.24,
    }
)


def strict_majority_quota(panel_size: int) -> int:
    """Smallest strict-majority count for an odd panel."""
    if panel_size <= 0 or panel_size % 2 == 0:
        raise ValueError("GenLayer panel sizes must be positive and odd")
    return panel_size // 2 + 1


def committee_capture_probability(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
) -> float | np.ndarray:
    """Probability that i.i.d. adversarial membership gets a strict majority."""
    alpha = np.asarray(adversarial_fraction, dtype=float)
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("adversarial_fraction must lie in [0, 1]")
    probability = stats.binom.sf(
        strict_majority_quota(panel_size) - 1,
        panel_size,
        alpha,
    )
    return float(probability) if probability.ndim == 0 else probability


def semantic_attack_probability(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    clarity: float | np.ndarray,
) -> float | np.ndarray:
    """Targeted-contamination failure for honest mean ``1/2 + clarity``.

    An admissible Hamming contamination of size ``alpha`` may target entries
    that support the honest population decision, reducing their mean by the
    full ``alpha``.  This is stronger than sampling an ``alpha`` share of
    fixed Byzantine identities from a mixed network; see
    :func:`fixed_share_attack_probability`.
    """
    gamma = np.asarray(clarity, dtype=float)
    if np.any((gamma < 0.0) | (gamma > 0.5)):
        raise ValueError("clarity must lie in [0, 1/2]")
    alpha = np.asarray(adversarial_fraction, dtype=float)
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("adversarial_fraction must lie in [0, 1]")
    honest_side_probability = np.clip(0.5 + gamma - alpha, 0.0, 1.0)
    probability = stats.binom.cdf(
        strict_majority_quota(panel_size) - 1,
        panel_size,
        honest_side_probability,
    )
    return float(probability) if probability.ndim == 0 else probability


def fixed_share_attack_probability(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    honest_clarity: float | np.ndarray,
) -> float | np.ndarray:
    """Panel failure for a fixed Byzantine share in the i.i.d. benchmark.

    Each sampled identity is Byzantine with probability ``alpha`` and then
    votes against the honest decision.  Otherwise it is honest and supports
    that decision with probability ``1/2 + honest_clarity``.  The unconditional
    honest-side vote probability is therefore
    ``(1 - alpha) * (1/2 + honest_clarity)``.
    """
    gamma = np.asarray(honest_clarity, dtype=float)
    if np.any((gamma < 0.0) | (gamma > 0.5)):
        raise ValueError("honest_clarity must lie in [0, 1/2]")
    alpha = np.asarray(adversarial_fraction, dtype=float)
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("adversarial_fraction must lie in [0, 1]")
    honest_side_probability = (1.0 - alpha) * (0.5 + gamma)
    probability = stats.binom.cdf(
        strict_majority_quota(panel_size) - 1,
        panel_size,
        honest_side_probability,
    )
    return float(probability) if probability.ndim == 0 else probability


def required_robust_margin(panel_size: int, error_tolerance: float) -> float:
    """Smallest post-corruption clarity giving panel error at most ``delta``."""
    if not 0.0 < error_tolerance < 0.5:
        raise ValueError("error_tolerance must lie in (0, 1/2)")

    def residual(margin: float) -> float:
        return (
            semantic_attack_probability(panel_size, 0.0, margin)
            - error_tolerance
        )

    return float(optimize.brentq(residual, 0.0, 0.5, xtol=1e-14, rtol=1e-14))


def maximum_certified_corruption(
    panel_size: int,
    clarity: float | np.ndarray,
    error_tolerance: float,
) -> float | np.ndarray:
    """Largest corruption fraction certified at a declared pointwise error."""
    gamma = np.asarray(clarity, dtype=float)
    if np.any((gamma < 0.0) | (gamma > 0.5)):
        raise ValueError("clarity must lie in [0, 1/2]")
    margin_floor = required_robust_margin(panel_size, error_tolerance)
    tolerable = gamma - margin_floor
    tolerable = np.where(tolerable >= -1e-13, np.maximum(tolerable, 0.0), np.nan)
    return float(tolerable) if tolerable.ndim == 0 else tolerable


def required_honest_clarity(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    error_tolerance: float,
) -> float | np.ndarray:
    """Clarity needed under targeted contamination and finite-panel error."""
    alpha = np.asarray(adversarial_fraction, dtype=float)
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("adversarial_fraction must lie in [0, 1]")
    required = alpha + required_robust_margin(panel_size, error_tolerance)
    return float(required) if required.ndim == 0 else required


def required_honest_clarity_fixed_share(
    panel_size: int,
    adversarial_fraction: float | np.ndarray,
    error_tolerance: float,
) -> float | np.ndarray:
    """Honest-subpopulation clarity needed under a fixed Byzantine share."""
    alpha = np.asarray(adversarial_fraction, dtype=float)
    if np.any((alpha < 0.0) | (alpha >= 1.0)):
        raise ValueError("adversarial_fraction must lie in [0, 1)")
    margin_floor = required_robust_margin(panel_size, error_tolerance)
    required = (0.5 + margin_floor) / (1.0 - alpha) - 0.5
    return float(required) if required.ndim == 0 else required


def write_margin_summary(path: Path) -> None:
    """Write exact robust-margin floors for useful service-level targets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tolerances = (1e-2, 1e-3, 1e-6)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "round_index",
                "panel_size",
                "round_type",
                "source_revision",
                "benchmark",
                "required_margin_delta_1e-2",
                "required_margin_delta_1e-3",
                "required_margin_delta_1e-6",
            ]
        )
        for round_index, panel_size in enumerate(GENLAYER_PANEL_SIZES):
            margins = [required_robust_margin(panel_size, delta) for delta in tolerances]
            writer.writerow(
                [
                    round_index,
                    panel_size,
                    "normal" if round_index % 2 == 0 else "appeal",
                    SOURCE_REVISION,
                    "large_population_iid_binomial",
                    *(f"{margin:.12f}" for margin in margins),
                ]
            )


def _plot_probability_family(
    ax: plt.Axes,
    x: np.ndarray,
    probability_function,
    colors: list,
) -> None:
    floor = 1e-14
    for level, (normal_size, appeal_size) in enumerate(
        zip(NORMAL_PANEL_SIZES, APPEAL_PANEL_SIZES)
    ):
        color = colors[level]
        normal_probability = probability_function(normal_size, x)
        appeal_probability = probability_function(appeal_size, x)
        ax.semilogy(x, np.clip(normal_probability, floor, 1.0), color=color)
        ax.semilogy(
            x,
            np.clip(appeal_probability, floor, 1.0),
            color=color,
            linestyle="--",
        )
    ax.set_ylim(floor, 1.1)


def generate_figure(path: Path, error_tolerance: float = 0.01) -> None:
    """Generate the four-panel non-adaptive corruption sensitivity figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [plt.cm.viridis(value) for value in np.linspace(0.05, 0.95, 9)]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

    capture_alpha = np.linspace(0.0, 0.5, 1001)
    _plot_probability_family(
        axes[0, 0],
        capture_alpha,
        committee_capture_probability,
        colors,
    )
    axes[0, 0].axvline(0.5, color="black", linestyle=":", linewidth=1.1)
    axes[0, 0].set_xlim(0.0, 0.5)
    axes[0, 0].set_xlabel(r"Adversarial network fraction $\alpha$")
    axes[0, 0].set_ylabel("Attack success probability")
    axes[0, 0].set_title("(a) Strict-majority committee capture")

    low_clarity = 0.05
    low_alpha = np.linspace(0.0, 0.15, 751)
    _plot_probability_family(
        axes[0, 1],
        low_alpha,
        lambda panel_size, alpha: semantic_attack_probability(
            panel_size, alpha, low_clarity
        ),
        colors,
    )
    axes[0, 1].axvline(
        low_clarity,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label=r"population boundary $\alpha=\gamma$",
    )
    axes[0, 1].set_xlim(0.0, 0.15)
    axes[0, 1].set_xlabel(r"Adversarial corruption fraction $\alpha$")
    axes[0, 1].set_ylabel("Outcome-manipulation probability")
    axes[0, 1].set_title(r"(b) Low-clarity task: $\gamma=0.05$")

    high_clarity = 0.20
    high_alpha = np.linspace(0.0, 0.30, 901)
    _plot_probability_family(
        axes[1, 0],
        high_alpha,
        lambda panel_size, alpha: semantic_attack_probability(
            panel_size, alpha, high_clarity
        ),
        colors,
    )
    axes[1, 0].axvline(
        high_clarity,
        color="black",
        linestyle=":",
        linewidth=1.1,
    )
    axes[1, 0].set_xlim(0.0, 0.30)
    axes[1, 0].set_xlabel(r"Adversarial corruption fraction $\alpha$")
    axes[1, 0].set_ylabel("Outcome-manipulation probability")
    axes[1, 0].set_title(r"(c) Clearer task: $\gamma=0.20$")

    clarity_grid = np.linspace(0.0, 0.5, 1001)
    for level, (normal_size, appeal_size) in enumerate(
        zip(NORMAL_PANEL_SIZES, APPEAL_PANEL_SIZES)
    ):
        color = colors[level]
        axes[1, 1].plot(
            clarity_grid,
            maximum_certified_corruption(
                normal_size,
                clarity_grid,
                error_tolerance,
            ),
            color=color,
        )
        axes[1, 1].plot(
            clarity_grid,
            maximum_certified_corruption(
                appeal_size,
                clarity_grid,
                error_tolerance,
            ),
            color=color,
            linestyle="--",
        )
    axes[1, 1].plot(
        clarity_grid,
        clarity_grid,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label=r"population ceiling $\alpha=\gamma$",
    )
    axes[1, 1].set_xlim(0.0, 0.5)
    axes[1, 1].set_ylim(0.0, 0.5)
    axes[1, 1].set_xlabel(r"Honest task clarity $\gamma$")
    axes[1, 1].set_ylabel(r"Maximum certified corruption $\alpha^*$")
    axes[1, 1].set_title(
        rf"(d) Pointwise attack risk at most $\delta={error_tolerance:g}$"
    )

    style_handles = [
        Line2D([0], [0], color="black", label="normal round $K$"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="appeal round $K$",
        ),
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            color=colors[level],
            linewidth=2.5,
            label=f"{normal_size}/{appeal_size}",
        )
        for level, (normal_size, appeal_size) in enumerate(
            zip(NORMAL_PANEL_SIZES, APPEAL_PANEL_SIZES)
        )
    ]
    fig.legend(
        handles=style_handles + size_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        title="Nominal FeeManager panel pairs (normal/appeal)",
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Non-adaptive corruption profiles under strict majority", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.17, hspace=0.34, wspace=0.24)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_panel_clarity_figure(
    path: Path,
    error_tolerance: float = 0.01,
) -> None:
    """Plot fixed-share clarity requirements against real panel size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_sizes = np.asarray(GENLAYER_PANEL_SIZES)
    adversarial_fractions = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    colors = [
        plt.cm.plasma(value)
        for value in np.linspace(0.05, 0.90, len(adversarial_fractions))
    ]

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    for alpha, color in zip(adversarial_fractions, colors):
        required = np.asarray(
            [
                required_honest_clarity_fixed_share(
                    panel_size,
                    alpha,
                    error_tolerance,
                )
                for panel_size in panel_sizes
            ]
        )
        ax.plot(
            panel_sizes,
            required,
            color=color,
            marker="o",
            markersize=3.5,
            label=rf"$\alpha={100 * alpha:.0f}\%$",
        )

    upper_limit = 0.84
    ax.axhspan(0.5, upper_limit, color="#D1495B", alpha=0.09)
    ax.axhline(0.5, color="#9E2A2B", linestyle="--", linewidth=1.2)
    ax.text(
        6.0,
        0.795,
        r"infeasible region: binary clarity cannot exceed $1/2$",
        color="#8B1E1E",
        fontsize=9,
    )

    pair_ticks = np.sqrt(
        np.asarray(NORMAL_PANEL_SIZES, dtype=float)
        * np.asarray(APPEAL_PANEL_SIZES, dtype=float)
    )
    pair_labels = [
        f"{normal}/{appeal}"
        for normal, appeal in zip(NORMAL_PANEL_SIZES, APPEAL_PANEL_SIZES)
    ]
    ax.set_xscale("log")
    ax.set_xticks(pair_ticks, pair_labels, rotation=30, ha="right")
    ax.set_xlim(4.5, 1800)
    ax.set_ylim(0.0, upper_limit)
    ax.set_xlabel("Implemented panel size $K$ (normal/appeal pair)")
    ax.set_ylabel(r"Minimum honest-subpopulation clarity $\gamma_{H,\min}$")
    ax.set_title(
        rf"Honest clarity required under a fixed Byzantine network share "
        rf"at $\delta={error_tolerance:g}$"
    )
    ax.legend(
        title="Byzantine network share",
        ncol=2,
        frameon=True,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _log10_odds(probability: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    transformed = np.log10(clipped / (1.0 - clipped))
    return float(transformed) if transformed.ndim == 0 else transformed


def generate_attack_phase_diagram(
    path: Path,
    panel_size: int = 1537,
    error_tolerance: float = 0.01,
) -> None:
    """Compare fixed-share and targeted-contamination attack phase diagrams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.linspace(0.0, 0.5, 601)
    gamma = np.linspace(0.0, 0.5, 601)
    alpha_grid, gamma_grid = np.meshgrid(alpha, gamma)
    contour_probabilities = (1e-6, 0.01, 0.5, 0.99, 1.0 - 1e-6)
    contour_levels = [_log10_odds(probability) for probability in contour_probabilities]
    contour_labels = {
        contour_levels[0]: r"$10^{-6}$",
        contour_levels[1]: "1%",
        contour_levels[2]: "50%",
        contour_levels[3]: "99%",
        contour_levels[4]: r"$1-10^{-6}$",
    }
    robust_margin = required_robust_margin(panel_size, error_tolerance)
    central_alpha = np.linspace(0.0, 0.5, 601)

    model_specs = (
        (
            fixed_share_attack_probability,
            "(a) Fixed Byzantine network share",
            r"50%: $\gamma_H=\alpha/[2(1-\alpha)]$",
        ),
        (
            semantic_attack_probability,
            "(b) Targeted census contamination",
            r"50%: $\gamma=\alpha$",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1), constrained_layout=True)
    image = None
    for model_index, (probability_function, title, central_label) in enumerate(
        model_specs
    ):
        ax = axes[model_index]
        attack_probability = probability_function(
            panel_size,
            alpha_grid,
            gamma_grid,
        )
        log_odds = _log10_odds(attack_probability)
        image = ax.pcolormesh(
            alpha,
            gamma,
            log_odds,
            shading="auto",
            cmap="RdYlGn_r",
            vmin=-6.0,
            vmax=6.0,
        )
        contours = ax.contour(
            alpha,
            gamma,
            log_odds,
            levels=contour_levels,
            colors="white",
            linewidths=(0.8, 1.2, 1.8, 1.2, 0.8),
            alpha=0.95,
        )
        ax.clabel(contours, fmt=contour_labels, inline=True, fontsize=7)

        if model_index == 0:
            central_gamma = central_alpha / (2.0 * (1.0 - central_alpha))
            safe_gamma = (
                central_alpha / 2.0 + robust_margin
            ) / (1.0 - central_alpha)
            successful_gamma = (
                central_alpha / 2.0 - robust_margin
            ) / (1.0 - central_alpha)
        else:
            central_gamma = central_alpha
            safe_gamma = central_alpha + robust_margin
            successful_gamma = central_alpha - robust_margin

        central_mask = (central_gamma >= 0.0) & (central_gamma <= 0.5)
        safe_mask = (safe_gamma >= 0.0) & (safe_gamma <= 0.5)
        successful_mask = (successful_gamma >= 0.0) & (successful_gamma <= 0.5)
        ax.plot(
            central_alpha[central_mask],
            central_gamma[central_mask],
            color="black",
            linewidth=1.7,
            label=central_label,
        )
        ax.plot(
            central_alpha[safe_mask],
            safe_gamma[safe_mask],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=rf"attack $\leq{100 * error_tolerance:g}\%$",
        )
        ax.plot(
            central_alpha[successful_mask],
            successful_gamma[successful_mask],
            color="black",
            linestyle=":",
            linewidth=1.6,
            label=rf"attack $\geq{100 * (1 - error_tolerance):g}\%$",
        )
        ax.text(
            0.055,
            0.43,
            "attack unlikely",
            color="#135D2D",
            weight="bold",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
        ax.text(
            0.365,
            0.055,
            "attack likely",
            color="#8B1E1E",
            weight="bold",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Adversarial fraction $\alpha$")
        if model_index == 0:
            ax.set_ylabel(r"Honest-subpopulation clarity $\gamma_H$")
        else:
            ax.set_ylabel(r"Pre-contamination census clarity $\gamma$")
        ax.set_title(title)
        ax.legend(loc="upper left", framealpha=0.92, fontsize=7.5)

    probability_ticks = (1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1 - 1e-6)
    colorbar = fig.colorbar(image, ax=axes, pad=0.025, shrink=0.88)
    colorbar.set_ticks([_log10_odds(probability) for probability in probability_ticks])
    colorbar.set_ticklabels(
        [r"$10^{-6}$", "0.1%", "1%", "10%", "50%", "90%", "99%", "99.9%", r"$1-10^{-6}$"]
    )
    colorbar.set_label("Attack success probability")

    fig.suptitle(
        rf"Outcome-manipulation phase diagrams for panel $K={panel_size}$"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--delta", type=float, default=0.01)
    args = parser.parse_args()

    figure_path = FIGURES_DIR / "fig6_adversarial_security_profiles.png"
    panel_clarity_path = FIGURES_DIR / "fig7_panel_clarity_adversary.png"
    phase_diagram_path = FIGURES_DIR / "fig8_attack_phase_diagram_k1537.png"
    summary_path = RESULTS_DIR / "adversarial_security_margin_floors.csv"
    generate_figure(figure_path, args.delta)
    generate_panel_clarity_figure(panel_clarity_path, args.delta)
    generate_attack_phase_diagram(phase_diagram_path, 1537, args.delta)
    write_margin_summary(summary_path)
    print(f"Generated {figure_path}")
    print(f"Generated {panel_clarity_path}")
    print(f"Generated {phase_diagram_path}")
    print(f"Generated {summary_path}")

    if args.show:
        image = plt.imread(figure_path)
        plt.figure(figsize=(12.0, 8.8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
