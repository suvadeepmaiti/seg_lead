"""Reproduce Figure 4, Panels C and D, across EEG electrode counts.

Summary
-------
This script repeats the noisy tri-montage orientation analysis with 4, 8, 16,
and 32 EEG electrodes. Panel C summarizes the maximum absolute bias for the
120-degree contact-pair and consecutive 5-degree comparisons. Panel D
summarizes the maximum variability measure (1.96 times the population standard
deviation) across Monte Carlo realizations.

Data used
---------

* ``../human/data/raw/noise_recording.bdf``: background EEG noise.
* ``data/processed/topographies/*.npy``: cached phantom topographies for
  configurations 23, 34, and 42 at 0-120 degrees.
* ``data/reference/bem_reference.csv``: BEM orientation model.

Outputs
-------

* ``results/panels_cd_data.mat``: electrode counts, maximum bias values, and
  maximum variability values.
* ``results/figure4_panel_c.png`` and ``.pdf``: bias summary.
* ``results/figure4_panel_d.png`` and ``.pdf``: variability summary.

Run ``python phantom/code/panel_cd.py --help`` for reproducibility and display
options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from analysis_common import (
    ELECTRODE_SETS,
    REPOSITORY_ROOT,
    ProjectPaths,
    consecutive_step_errors,
    contact_pair_errors,
    load_bem_models,
    load_noise_topographies,
    run_monte_carlo_analysis,
    variability_half_width,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
ELECTRODE_ORDER = (4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--realisations", type=int, default=32)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows")
    return parser.parse_args()


def make_bar_figure(
    consecutive_values: list[float],
    pair_values: list[float],
    ylabel: str,
    upper_limit: float,
) -> plt.Figure:
    """Create a grouped bar chart across electrode counts."""
    positions = np.arange(len(ELECTRODE_ORDER))
    width = 0.32
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.bar(
        positions - width / 2,
        consecutive_values,
        width,
        color=BLUE,
        edgecolor="black",
        linewidth=0.8,
        label="Consecutive step",
    )
    axis.bar(
        positions + width / 2,
        pair_values,
        width,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.8,
        label="Contact-pair difference",
    )
    axis.set_xticks(positions, [str(count) for count in ELECTRODE_ORDER])
    axis.set(xlabel="Number of EEG electrodes", ylabel=ylabel, ylim=(0, upper_limit))
    axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="upper right", framealpha=0.9)
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.repo_root)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    noise, noise_rng = load_noise_topographies(paths.noise_file, args.trials, args.noise_seed)
    models = load_bem_models(paths.bem_file, ELECTRODE_SETS)
    estimates, _ = run_monte_carlo_analysis(
        paths.cache_dir,
        models,
        noise,
        noise_rng,
        ELECTRODE_SETS,
        n_realisations=args.realisations,
        trial_count=args.trials,
        phantom_seed=args.seed,
    )
    pair_errors = {count: contact_pair_errors(values) for count, values in estimates.items()}
    step_errors = {count: consecutive_step_errors(values) for count, values in estimates.items()}

    max_pair_bias = [
        float(np.max(np.abs(pair_errors[count].mean(axis=0)))) for count in ELECTRODE_ORDER
    ]
    max_step_bias = [
        float(np.max(np.abs(step_errors[count].mean(axis=0)))) for count in ELECTRODE_ORDER
    ]
    max_pair_width = [
        float(np.max(variability_half_width(pair_errors[count]))) for count in ELECTRODE_ORDER
    ]
    max_step_width = [
        float(np.max(variability_half_width(step_errors[count]))) for count in ELECTRODE_ORDER
    ]

    print("\nPanels C-D summary (recorded background noise)")
    print("Electrodes  Pair bias  Pair 1.96 SD  Step bias  Step 1.96 SD")
    for index, count in enumerate(ELECTRODE_ORDER):
        print(
            f"{count:10d}  {max_pair_bias[index]:9.2f}  {max_pair_width[index]:12.2f}  "
            f"{max_step_bias[index]:9.2f}  {max_step_width[index]:12.2f}"
        )

    data_path = paths.results_dir / "panels_cd_data.mat"
    savemat(
        data_path,
        {
            "n_electrodes": np.asarray(ELECTRODE_ORDER),
            "max_pair_bias_deg": np.asarray(max_pair_bias),
            "max_step_bias_deg": np.asarray(max_step_bias),
            "max_pair_1_96_sd_deg": np.asarray(max_pair_width),
            "max_step_1_96_sd_deg": np.asarray(max_step_width),
        },
    )
    print(f"Saved {data_path}")

    figures = {
        "figure4_panel_c": make_bar_figure(
            max_step_bias,
            max_pair_bias,
            ylabel="Maximum absolute bias (deg)",
            upper_limit=10,
        ),
        "figure4_panel_d": make_bar_figure(
            max_step_width,
            max_pair_width,
            ylabel="Maximum 1.96 SD (deg)",
            upper_limit=6,
        ),
    }
    for stem, figure in figures.items():
        for extension in ("png", "pdf"):
            figure_path = paths.results_dir / f"{stem}.{extension}"
            figure.savefig(figure_path, bbox_inches="tight", dpi=300)
            print(f"Saved {figure_path}")

    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
