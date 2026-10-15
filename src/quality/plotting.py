"""CPU-only publication/export figures drawn entirely from real score outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, folder: Path, name: str, settings: dict) -> str:
    """Save a clean PNG and close it to release memory."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/name, dpi=settings["figure_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def ecdf(ax, values, label: str, color: str) -> None:
    """Plot exact empirical CDF; plotting can rasterize dense segments."""
    x = np.sort(values)
    ax.plot(x, np.arange(1, len(x)+1)/len(x), label=label, color=color, linewidth=1.5)


def make_quality_figures(folder, weights, datasets, a1_domains, domain_rows,
                         remainders, equal, spearman, settings, correlation, missing_rows):
    """Generate required figures plus missingness, correlation and mean plots."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    files = []
    ordered = sorted(weights, key=lambda x: float(x["critic_weight"]))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh([r["metric"] for r in ordered], [float(r["critic_weight"]) for r in ordered], color="#4777a8")
    ax.set(xlabel="A1 Pearson-CRITIC weight", title="Locked objective weights (22 indicators)")
    ax.axvline(1/22, color="#666666", linestyle="--", label="Equal weight = 1/22")
    ax.legend(); fig.tight_layout()
    files.append(save_figure(fig, folder, "01_critic_weights.png", settings))
    domains = [r["domain"] for r in domain_rows]
    estimate = np.array([r["Q_domain"] for r in domain_rows])
    ci = np.array([[r["ci95_low"] for r in domain_rows], [r["ci95_high"] for r in domain_rows]])
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(domains))
    ax.vlines(positions, ci[0], ci[1], color="#4777a8", linewidth=2)
    ax.scatter(positions, estimate, color="#203f60", s=35, zorder=3)
    ax.set(xticks=positions, xticklabels=domains, ylabel="10% trimmed mean Q",
           title="A1 domain quality with 95% bootstrap CI (B=2000)")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    files.append(save_figure(fig, folder, "02_A1_domain_quality_ci.png", settings))
    groups = [datasets["A1"]["Q"][a1_domains == d] for d in domains]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, tick_labels=domains, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#d7e3ef"}, medianprops={"color": "#243e59"})
    ax.set(ylabel="Sample Q", ylim=(0, 1), title="A1 domain Q distributions (outliers hidden)")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    files.append(save_figure(fig, folder, "03_A1_domain_Q_boxplots.png", settings))
    for dataset, domain, number in [("A2", "arxiv", 4), ("A3", "github", 5)]:
        a = datasets["A1"]["Q"][a1_domains == domain]
        b = datasets[dataset]["Q"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for values, label, color in [(a, f"A1-{domain} (n={len(a):,})", "#4777a8"),
                                     (b, f"{dataset}-full (n={len(b):,})", "#c07842")]:
            axes[0].hist(values, bins=np.linspace(0, 1, 61), density=True, histtype="step", linewidth=1.5, label=label, color=color)
            ecdf(axes[1], values, label, color)
        axes[0].set(xlabel="Q", ylabel="Density", title=f"{domain}: sample vs full extension")
        axes[1].set(xlabel="Q", ylabel="Cumulative probability", title="Empirical CDF")
        for ax in axes: ax.legend(fontsize=8)
        fig.tight_layout()
        files.append(save_figure(fig, folder, f"0{number}_{domain}_sample_full.png", settings))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dataset, domain in zip(axes, ["A2", "A3"], ["arxiv", "github"]):
        a = datasets["A1"]["Q"][a1_domains == domain]
        b = remainders[dataset]
        ecdf(ax, a, f"A1-{domain} (n={len(a):,})", "#4777a8")
        ecdf(ax, b, f"Nonoverlap extension (n={len(b):,})", "#c07842")
        ax.set(xlabel="Q", ylabel="Cumulative probability", title=f"{domain}: disjoint record sets")
        ax.legend(fontsize=8)
    fig.tight_layout()
    files.append(save_figure(fig, folder, "06_nonoverlap_comparison.png", settings))
    rng = np.random.default_rng(settings["random_state"])
    q = datasets["A1"]["Q"]
    chosen = np.sort(rng.choice(len(q), min(len(q), settings["scatter_max_points"]), replace=False))
    for other, label, name in [(equal, "Equal-weight Q", "07_critic_vs_equal.png"),
                               (spearman, "Spearman-CRITIC Q (sensitivity)", "08_pearson_vs_spearman_Q.png")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(q[chosen], other[chosen], s=4, alpha=.2, color="#4777a8", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
        ax.set(xlabel="Pearson-CRITIC Q (main)", ylabel=label, xlim=(0, 1), ylim=(0, 1),
               title=f"A1 sensitivity scores ({len(chosen):,} points displayed)")
        fig.tight_layout(); files.append(save_figure(fig, folder, name, settings))
    names = [r["metric"] for r in weights]
    fig, ax = plt.subplots(figsize=(12, 9))
    missing = {r["metric"]: int(r["missing_before"]) for r in missing_rows if r["dataset"] == "A1"}
    ax.barh(names[::-1], [100*missing[n]/len(q) for n in names[::-1]], color="#4777a8")
    ax.set(xlabel="Missing scalars before imputation (%)", title="A1 missingness across 22 indicators")
    fig.tight_layout(); files.append(save_figure(fig, folder, "09_A1_indicator_missingness.png", settings))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(xticks=range(22), yticks=range(22), xticklabels=names, yticklabels=names,
           title="A1 Pearson correlation of normalized indicators")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(im, ax=ax, shrink=.7, label="Pearson r"); fig.tight_layout()
    files.append(save_figure(fig, folder, "10_indicator_correlation.png", settings))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions-.18, [r["mean_Q"] for r in domain_rows], width=.36, color="#9cb4ca", label="Mean Q")
    ax.bar(positions+.18, estimate, width=.36, color="#4777a8", label="10% trimmed mean Q")
    ax.set(xticks=positions, xticklabels=domains, ylabel="Q", title="A1 mean and trimmed mean by domain")
    ax.tick_params(axis="x", rotation=20); ax.legend(); fig.tight_layout()
    files.append(save_figure(fig, folder, "11_A1_mean_trimmed_mean.png", settings))
    return files
