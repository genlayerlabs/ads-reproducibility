"""Numerical illustrations for Part 1, version 5.5.

The formal finite-panel certificates are exact hypergeometric probabilities.
The path plots illustrate, but do not prove, the functional limit theorems.

Usage:
    python convergence_validation.py
    python convergence_validation.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from operational_certificate import quota


FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

BLUE = "#2176AE"
RED = "#D1495B"
ORANGE = "#F28E2B"
GREEN = "#2A9D8F"
PURPLE = "#7551A1"
GRAY = "#66717E"
COLORS = [BLUE, RED, ORANGE, GREEN, PURPLE, GRAY]


def hypergeom_error(m: int, c: int, k: int, tau: float) -> float:
    """Exact panel--census disagreement under uniform sampling without replacement."""
    q = quota(k, tau)
    census_accepts = c >= quota(m, tau)
    if census_accepts:
        return float(stats.hypergeom.cdf(q - 1, m, c, k))
    return float(stats.hypergeom.sf(q - 1, m, c, k))


def binomial_error(p: float, k: int, tau: float, target_accepts: bool) -> float:
    """Small-sampling-fraction comparison, not the finite-census truth."""
    q = quota(k, tau)
    if target_accepts:
        return float(stats.binom.cdf(q - 1, k, p))
    return float(stats.binom.sf(q - 1, k, p))


def figure1_finite_panel_error(show: bool = False) -> None:
    m = 1500
    tau = 0.5
    ks = np.arange(5, 301)
    proportions = [0.55, 0.60, 0.70]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, p, color in zip(axes, proportions, COLORS):
        c = int(round(m * p))
        p_m = c / m
        exact = np.array([hypergeom_error(m, c, int(k), tau) for k in ks])
        approx = np.array([binomial_error(p_m, int(k), tau, True) for k in ks])
        ax.semilogy(ks, np.maximum(exact, 1e-16), color=color, label="Exact hypergeometric")
        ax.semilogy(
            ks,
            np.maximum(approx, 1e-16),
            color=GRAY,
            linestyle="--",
            label="Binomial comparison",
        )
        ax.set_title(rf"$M={m}$, $\mu_M={p_m:.2f}$, $\gamma_M={p_m-tau:.2f}$")
        ax.set_xlabel("Panel size $K$")
        ax.set_ylim(1e-12, 1.1)
        ax.legend()
    axes[0].set_ylabel(r"$P(D_{M,K}\ne D_M\mid\mathbf{y}_M)$")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_finite_panel_error.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _interpolated_centered_path(values: np.ndarray, mean: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    t = np.arange(n + 1) / n
    scale = np.sqrt(n * mean * (1 - mean))
    path = np.concatenate(([0.0], np.cumsum(values - mean))) / scale
    return t, path


def figure2_nested_paths(show: bool = False) -> None:
    rng = np.random.default_rng(5502)
    n = 1000
    p = 0.55
    c = int(round(n * p))
    census = np.concatenate((np.ones(c, dtype=float), np.zeros(n - c, dtype=float)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for j in range(7):
        iid = rng.binomial(1, p, size=n).astype(float)
        t, path = _interpolated_centered_path(iid, p)
        axes[0].plot(t, path, color=COLORS[j % len(COLORS)], alpha=0.75)

        permuted = rng.permutation(census)
        t, bridge = _interpolated_centered_path(permuted, c / n)
        axes[1].plot(t, bridge, color=COLORS[j % len(COLORS)], alpha=0.75)

    axes[0].set_title("i.i.d. evaluator episodes: motion limit")
    axes[1].set_title("fixed census, random order: bridge limit")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Revealed fraction $t$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Standardized centered partial sum")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_nested_paths.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _synthetic_census_counts(m: int, n_resolutions: int, rng: np.random.Generator) -> np.ndarray:
    """Produce a transparent synthetic margin spectrum on both sides of 1/2."""
    side = rng.choice([-1.0, 1.0], size=n_resolutions)
    margins = 0.005 + 0.33 * rng.beta(1.4, 3.0, size=n_resolutions)
    proportions = np.clip(0.5 + side * margins, 0.01, 0.99)
    return np.rint(m * proportions).astype(int)


def figure3_resolvability(show: bool = False) -> None:
    rng = np.random.default_rng(5503)
    m = 1500
    tau = 0.5
    delta = 0.01
    counts = _synthetic_census_counts(m, 260, rng)
    candidate_ks = np.arange(5, 301)

    errors = np.empty((len(counts), len(candidate_ks)))
    for a, c in enumerate(counts):
        errors[a] = [hypergeom_error(m, int(c), int(k), tau) for k in candidate_ks]

    k_min = np.full(len(counts), np.nan)
    for a in range(len(counts)):
        eligible = np.flatnonzero(errors[a] <= delta)
        if eligible.size:
            k_min[a] = candidate_ks[eligible[0]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    proportions = counts / m
    axes[0].scatter(
        proportions,
        k_min,
        c=np.abs(proportions - tau),
        cmap="viridis",
        s=25,
        alpha=0.8,
    )
    axes[0].axvline(tau, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"Resolution-specific census acceptance $\mu_M(T_a)$")
    axes[0].set_ylabel(r"Minimum tested $K$ ($5\leq K\leq300$)")
    axes[0].set_title("Pointwise minimum panel size")

    for d, color in zip([0.10, 0.05, 0.01, 0.001], COLORS):
        coverage = np.mean(errors <= d, axis=0)
        axes[1].plot(candidate_ks, coverage, color=color, label=rf"$\delta={d:g}$")
    axes[1].set_xlabel("Panel size $K$")
    axes[1].set_ylabel(r"Empirical coverage $R_{K,\delta}$")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Aggregate only after pointwise certification")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_resolvability.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def figure5_corruption(show: bool = False) -> None:
    m = 1500
    tau = 0.5
    c = 900  # honest mu_M = 0.60
    ks = np.arange(5, 301)
    corruption_fractions = [0.0, 0.02, 0.05, 0.08, 0.10]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for frac, color in zip(corruption_fractions, COLORS):
        b = int(round(frac * m))
        worst_c = max(0, c - b)
        errors = [hypergeom_error(m, worst_c, int(k), tau) for k in ks]
        effective_margin = c / m - tau - b / m
        if np.isclose(effective_margin, 0.0, atol=1e-12):
            effective_margin = 0.0
        ax.semilogy(
            ks,
            np.maximum(errors, 1e-16),
            color=color,
            label=rf"$b/M={frac:.2f}$, robust margin ${effective_margin:.2f}$",
        )
    ax.set_xlabel("Panel size $K$")
    ax.set_ylabel("Worst-case disagreement with honest acceptance")
    ax.set_title(r"Non-adaptive census corruption ($M=1500$, $\mu_M=0.60$)")
    ax.set_ylim(1e-12, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_corruption.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    figure1_finite_panel_error(args.show)
    figure2_nested_paths(args.show)
    figure3_resolvability(args.show)
    figure5_corruption(args.show)
    print(f"Generated four Part 1 figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
