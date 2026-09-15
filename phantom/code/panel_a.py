"""Reproduce Figure 4, Panel A, with recorded background EEG noise.

Summary
-------
This script estimates phantom orientation using either one contact pair or the
combined tri-montage. It adds independently sampled background EEG noise to
averaged phantom topographies and calculates the error between orientations
separated by 120 degrees.

Data used
---------

* ``../human/data/raw/noise_recording.bdf``: background EEG noise.
* ``data/processed/topographies/*.npy``: cached phantom topographies for
  configurations 23, 34, and 42 at 0-120 degrees.
* ``data/reference/bem_reference.csv``: BEM orientation model.
* ``data/reference/single_pair_no_noise.mat``: optional no-noise single-pair
  comparison curve. The simulated single-pair result is used if absent.

Outputs
-------

* ``results/panel_a_data.mat``: orientation values and single-/three-pair
  angular errors.
* ``results/figure4_panel_a.png``: raster figure.
* ``results/figure4_panel_a.pdf``: vector figure.

Run ``python phantom/code/panel_a.py --help`` for reproducibility and display
options.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from analysis_common import (
    ELECTRODE_SETS,
    REPOSITORY_ROOT,
    ROTATIONS_DEG,
    ProjectPaths,
    contact_pair_errors,
    load_bem_models,
    load_noise_topographies,
    run_monte_carlo_analysis,
    variability_half_width,
    wrap_to_180,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--realisations", type=int, default=32)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window")
    return parser.parse_args()


def load_single_pair_reference(
    path: Path,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the no-noise comparison curve or use the simulated fallback."""
    if not path.exists():
        warnings.warn(f"Comparison file not found; using noisy single-pair estimates: {path}")
        return fallback.mean(axis=0), variability_half_width(fallback)

    data = loadmat(path)
    if "est_single_err" not in data:
        raise KeyError(f"Required variable 'est_single_err' is missing from {path}")
    errors = wrap_to_180(np.asarray(data["est_single_err"])[: len(ROTATIONS_DEG), :])
    return errors.mean(axis=1), variability_half_width(errors.T)


def make_figure(
    tri_errors: np.ndarray,
    single_mean: np.ndarray,
    single_width: np.ndarray,
) -> plt.Figure:
    """Create the Panel A comparison figure."""
    tri_mean = tri_errors.mean(axis=0)
    tri_width = variability_half_width(tri_errors)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.fill_between(
        ROTATIONS_DEG,
        single_mean - single_width,
        single_mean + single_width,
        color=BLUE,
        alpha=0.25,
    )
    axis.fill_between(
        ROTATIONS_DEG,
        tri_mean - tri_width,
        tri_mean + tri_width,
        color=ORANGE,
        alpha=0.35,
    )
    axis.plot(ROTATIONS_DEG, single_mean, color=BLUE, linewidth=2, label="One contact pair")
    axis.plot(
        ROTATIONS_DEG,
        tri_mean,
        color=ORANGE,
        linewidth=2,
        linestyle="--",
        label="Three contact pairs",
    )
    axis.set(xlim=(0, 360), ylim=(-20, 20), xlabel="Orientation (deg)", ylabel="Error (deg)")
    axis.set_xticks([0, 100, 200, 300])
    axis.set_yticks([-20, -10, 0, 10, 20])
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
    models = load_bem_models(paths.bem_file, ELECTRODE_SETS)
    tri_estimates, single_estimates = run_monte_carlo_analysis(
        paths.cache_dir,
        models,
        noise,
        noise_rng,
        ELECTRODE_SETS,
        n_realisations=args.realisations,
        trial_count=args.trials,
        phantom_seed=args.seed,
        include_single_pair=True,
    )
    assert single_estimates is not None

    tri_errors = {count: contact_pair_errors(values) for count, values in tri_estimates.items()}
    single_errors = {
        count: contact_pair_errors(values) for count, values in single_estimates.items()
    }

    print("\nPanel A summary (recorded background noise)")
    for count in (32, 16, 8, 4):
        mean_error = tri_errors[count].mean(axis=0)
        width = variability_half_width(tri_errors[count])
        single_mean = single_errors[count].mean(axis=0)
        print(
            f"{count:2d} electrodes | tri max |mean| = {np.max(np.abs(mean_error)):.2f} deg; "
            f"max 1.96 SD = {np.max(width):.2f} deg; "
            f"single max |mean| = {np.max(np.abs(single_mean)):.2f} deg"
        )

    data_path = paths.results_dir / "panel_a_data.mat"
    savemat(
        data_path,
        {
            "orientation_deg": ROTATIONS_DEG.reshape(-1, 1),
            "tri_pair_error_deg": tri_errors[32].T,
            "single_pair_error_deg": single_errors[32].T,
        },
    )
    print(f"Saved {data_path}")

    single_mean, single_width = load_single_pair_reference(
        paths.comparison_file,
        single_errors[32],
    )
    figure = make_figure(tri_errors[32], single_mean, single_width)
    for extension in ("png", "pdf"):
        figure_path = paths.results_dir / f"figure4_panel_a.{extension}"
        figure.savefig(figure_path, bbox_inches="tight", dpi=300)
        print(f"Saved {figure_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
