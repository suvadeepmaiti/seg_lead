"""Reproduce Figure 4, Panel B: consecutive-step orientation error.

Summary
-------
This script runs the tri-montage orientation analysis with 32 EEG electrodes
and recorded background noise. It compares estimates at adjacent 5-degree
turntable positions and reports the error relative to the expected 5-degree
change.

Data used
---------

* ``../human/data/raw/noise_recording.bdf``: background EEG noise.
* ``data/processed/topographies/*.npy``: cached phantom topographies for
  configurations 23, 34, and 42 at 0-120 degrees.
* ``data/reference/bem_reference.csv``: BEM orientation model.

Outputs
-------

* ``results/panel_b_data.mat``: orientations and consecutive-step errors.
* ``results/figure4_panel_b.png``: raster figure.
* ``results/figure4_panel_b.pdf``: vector figure.

Run ``python phantom/code/panel_b.py --help`` for reproducibility and display
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
    ROTATIONS_DEG,
    ProjectPaths,
    consecutive_step_errors,
    load_bem_models,
    load_noise_topographies,
    run_monte_carlo_analysis,
    variability_half_width,
)


BLUE = "#0072B2"
ELECTRODES_32 = {32: ELECTRODE_SETS[32]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--realisations", type=int, default=32)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window")
    return parser.parse_args()


def make_figure(errors: np.ndarray) -> plt.Figure:
    """Create the Panel B consecutive-step figure."""
    orientations = ROTATIONS_DEG[:-1]
    mean_error = errors.mean(axis=0)
    width = variability_half_width(errors)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.fill_between(
        orientations,
        mean_error - width,
        mean_error + width,
        color=BLUE,
        alpha=0.25,
        label="Mean +/- 1.96 SD",
    )
    axis.plot(orientations, mean_error, color=BLUE, linewidth=2, label="Mean error")
    axis.set(xlim=(0, 355), ylim=(-12, 12), xlabel="Orientation (deg)", ylabel="Error (deg)")
    axis.set_xticks([0, 100, 200, 300])
    axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.legend(loc="upper right", framealpha=0.9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.repo_root)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    noise, noise_rng = load_noise_topographies(paths.noise_file, args.trials, args.noise_seed)
    models = load_bem_models(paths.bem_file, ELECTRODES_32)
    estimates, _ = run_monte_carlo_analysis(
        paths.cache_dir,
        models,
        noise,
        noise_rng,
        ELECTRODES_32,
        n_realisations=args.realisations,
        trial_count=args.trials,
        phantom_seed=args.seed,
    )
    errors = consecutive_step_errors(estimates[32])
    mean_error = errors.mean(axis=0)
    width = variability_half_width(errors)
    print(
        f"\nPanel B | max |mean| = {np.max(np.abs(mean_error)):.2f} deg; "
        f"max 1.96 SD = {np.max(width):.2f} deg; "
        f"mean |error| = {np.mean(np.abs(mean_error)):.2f} deg"
    )

    data_path = paths.results_dir / "panel_b_data.mat"
    savemat(
        data_path,
        {
            "orientation_deg": ROTATIONS_DEG[:-1].reshape(-1, 1),
            "consecutive_step_error_deg": errors.T,
        },
    )
    print(f"Saved {data_path}")

    figure = make_figure(errors)
    for extension in ("png", "pdf"):
        figure_path = paths.results_dir / f"figure4_panel_b.{extension}"
        figure.savefig(figure_path, bbox_inches="tight", dpi=300)
        print(f"Saved {figure_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
