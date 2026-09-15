"""Build processed phantom topographies from raw turntable recordings.

Summary
-------
For each required contact-pair configuration and turntable rotation, this
script reads a raw BDF recording, applies a 100 Hz high-pass filter, detects
stimulation peaks, extracts peak-amplitude scalp topographies, and stores a
reproducible subset for later Monte Carlo analysis.

Data used
---------
By default, the script reads 75 files from
``data/raw/turntable_measurements/``:

* Rotation folders ``000`` through ``120`` in 5-degree increments.
* ``23.bdf``, ``34.bdf``, and ``42.bdf`` within every rotation folder.

Outputs
-------
The script writes 75 NumPy arrays to
``data/processed/topographies/``. Files follow the pattern
``{configuration}_{rotation}.npy``; for example, ``23_000.npy``. Each array
contains up to 3,000 topographies with shape ``(32, n_topographies)``.

Run ``python code/build_cache.py --help`` for path, seed, overwrite, and cache
size options. Existing output files are skipped unless ``--overwrite`` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from analysis_common import (
    PHANTOM_SAMPLING_RATE_HZ,
    REPOSITORY_ROOT,
    TRIAL_LENGTH_MS,
    high_pass_filter,
)


STIMULATION_FREQUENCY_HZ = 130.0
N_TOPOGRAPHIES = 3000
ROTATIONS_DEG = range(0, 121, 5)
CONFIGURATIONS = ("23", "34", "42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw" / "turntable_measurements",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed" / "topographies",
    )
    parser.add_argument("--topographies", type=int, default=N_TOPOGRAPHIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_topographies(
    bdf_file: Path,
    n_topographies: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Extract stimulation-locked peak topographies from one BDF recording."""
    import mne

    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
    data = raw.get_data()[:32, :]
    if data.shape[0] < 32:
        raise ValueError(f"Expected at least 32 channels in {bdf_file}; found {data.shape[0]}")
    filtered = high_pass_filter(data, PHANTOM_SAMPLING_RATE_HZ)

    reference_channel = int(np.argmax(filtered.var(axis=1)))
    reference_signal = filtered[reference_channel]
    minimum_distance = int(PHANTOM_SAMPLING_RATE_HZ / STIMULATION_FREQUENCY_HZ * 0.7)
    threshold = np.percentile(np.abs(reference_signal), 75)
    peaks, _ = find_peaks(
        np.abs(reference_signal),
        distance=minimum_distance,
        height=threshold,
    )

    window_samples = round(TRIAL_LENGTH_MS / 1000 * PHANTOM_SAMPLING_RATE_HZ)
    correction_window = 2 * window_samples - 0.075
    phase_correction = (
        window_samples
        * np.mod(np.arange(len(peaks), dtype=float), correction_window)
        / correction_window
    )
    starts = np.round(peaks.astype(float) - phase_correction).astype(int)
    starts = starts[(starts >= 0) & (starts + window_samples < filtered.shape[1])]

    topographies = np.zeros((32, len(starts)), dtype=np.float64)
    for index, start in enumerate(starts):
        epoch = filtered[:, start : start + window_samples]
        peak = int(np.argmax(np.abs(epoch).sum(axis=0)))
        topographies[:, index] = filtered[:, start + peak]

    if topographies.shape[1] > n_topographies:
        indices = np.sort(rng.choice(topographies.shape[1], n_topographies, replace=False))
        topographies = topographies[:, indices]
    elif topographies.shape[1] < n_topographies:
        print(
            f"Warning: {bdf_file} yielded {topographies.shape[1]} topographies; "
            f"{n_topographies} requested"
        )
    return topographies


def build_cache(
    raw_dir: Path,
    output_dir: Path,
    n_topographies: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Process every required configuration and rotation."""
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    total = len(ROTATIONS_DEG) * len(CONFIGURATIONS)

    for index, (rotation, configuration) in enumerate(
        (
            (rotation, configuration)
            for rotation in ROTATIONS_DEG
            for configuration in CONFIGURATIONS
        ),
        start=1,
    ):
        source = raw_dir / f"{rotation:03d}" / f"{configuration}.bdf"
        destination = output_dir / f"{configuration}_{rotation:03d}.npy"
        if destination.exists() and not overwrite:
            print(f"[{index:02d}/{total}] Skipping existing {destination.name}")
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Required recording not found: {source}")

        print(f"[{index:02d}/{total}] Processing {source.relative_to(raw_dir)}")
        topographies = extract_topographies(source, n_topographies, rng)
        np.save(destination, topographies)
        print(f"             Saved {destination.name} with shape {topographies.shape}")


def main() -> None:
    args = parse_args()
    build_cache(
        args.raw_dir,
        args.output_dir,
        args.topographies,
        args.seed,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
