"""Shared numerical and input/output utilities for the phantom analysis.

Summary
-------
This module contains the common methods used by the three Figure 4 analysis
scripts. It reads the background-noise BDF, loads processed phantom
topographies and the BEM reference model, performs spatial correlations, runs
the Monte Carlo orientation analysis, and calculates angular errors.

Data used
---------
The calling scripts provide paths to:

* ``../human/data/raw/noise_recording.bdf``: recorded background EEG noise.
* ``data/processed/topographies/*.npy``: stimulation-locked phantom
  topographies for configurations 23, 34, and 42.
* ``data/reference/bem_reference.csv``: BEM topographies for 360 orientations.

Outputs
-------
This module does not write files directly. Its functions return NumPy arrays
containing noise topographies, orientation estimates, spatial correlations,
and error measurements. The panel scripts save those arrays and figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import butter, filtfilt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PHANTOM_SAMPLING_RATE_HZ = 4096.0
NOISE_SAMPLING_RATE_HZ = 512.0
HIGH_PASS_CUTOFF_HZ = 100.0
TRIAL_LENGTH_MS = 38.46
N_NOISE_TOPOGRAPHIES = 3000
ROTATION_STEP_DEG = 5
ROTATIONS_DEG = np.arange(0, 360, ROTATION_STEP_DEG)
MODEL_ANGLES_DEG = np.arange(360)

CALIBRATION_INTERCEPT = {"23": 190, "34": 91, "42": 322}
CALIBRATION_SLOPE = {"23": 1.1746, "34": 0.8329, "42": 0.9678}

ELECTRODE_SETS: dict[int, tuple[int, ...]] = {
    32: tuple(range(32)),
    16: (29, 0, 28, 30, 1, 27, 2, 31, 23, 6, 19, 12, 10, 18, 15, 11),
    8: (28, 30, 1, 31, 23, 6, 18, 11),
    4: (28, 1, 19, 10),
}


@dataclass(frozen=True)
class ProjectPaths:
    """Paths used by the analysis, resolved from a repository root."""

    root: Path
    bem_file: Path
    cache_dir: Path
    noise_file: Path
    comparison_file: Path
    results_dir: Path

    @classmethod
    def from_root(cls, root: Path | str = REPOSITORY_ROOT) -> "ProjectPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            bem_file=root / "data" / "reference" / "bem_reference.csv",
            cache_dir=root / "data" / "processed" / "topographies",
            noise_file=root.parent / "human" / "data" / "raw" / "noise_recording.bdf",
            comparison_file=root / "data" / "reference" / "single_pair_no_noise.mat",
            results_dir=root / "results",
        )


def high_pass_filter(data: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Apply the fifth-order high-pass filter used in the analysis."""
    b, a = butter(
        5,
        HIGH_PASS_CUTOFF_HZ / (sampling_rate_hz / 2),
        btype="high",
    )
    return filtfilt(b, a, data, axis=1)


def wrap_to_180(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to the half-open interval [-180, 180)."""
    return ((angle_deg + 180) % 360) - 180


def align_estimates(estimates: np.ndarray) -> np.ndarray:
    """Remove each realization's constant angular offset."""
    bias = estimates.mean(axis=1, keepdims=True) - ROTATIONS_DEG.mean()
    return estimates - bias


def source_for_rotation(configuration: str, rotation_deg: int) -> tuple[str, int]:
    """Map a full rotation to the measured 0-120 degree symmetry sector."""
    rotation_deg %= 360
    if rotation_deg <= 120:
        return configuration, rotation_deg
    if rotation_deg <= 240:
        return {"23": "42", "34": "23", "42": "34"}[configuration], rotation_deg - 120
    return {"23": "34", "34": "42", "42": "23"}[configuration], rotation_deg - 240


def read_bdf_channels(path: Path | str, n_channels: int = 32) -> np.ndarray:
    """Read the first channels of a BioSemi BDF file and return volts.

    This focused reader avoids adding a second BDF dependency to the analysis
    scripts. Cache generation uses MNE, which provides broader format support.
    """
    path = Path(path)
    with path.open("rb") as file:
        header = file.read(256).decode("ascii", errors="replace")
        n_records = int(header[236:244].strip())
        n_signals = int(header[252:256].strip())
        if n_records < 0:
            raise ValueError(f"BDF has an unknown record count: {path}")

        file.seek(256)
        labels = [file.read(16).decode("ascii", "replace").strip() for _ in range(n_signals)]
        file.read(80 * n_signals)
        physical_units = [file.read(8).decode("ascii", "replace").strip() for _ in range(n_signals)]
        physical_min = [float(file.read(8).decode("ascii", "replace").strip()) for _ in range(n_signals)]
        physical_max = [float(file.read(8).decode("ascii", "replace").strip()) for _ in range(n_signals)]
        digital_min = [float(file.read(8).decode("ascii", "replace").strip()) for _ in range(n_signals)]
        digital_max = [float(file.read(8).decode("ascii", "replace").strip()) for _ in range(n_signals)]
        file.read(80 * n_signals)
        samples_per_record = [
            int(file.read(8).decode("ascii", "replace").strip())
            for _ in range(n_signals)
        ]
        file.read(32 * n_signals)

    n_selected = min(n_channels, n_signals)
    if n_selected < n_channels:
        raise ValueError(f"Expected {n_channels} channels in {path}; found {n_signals}")

    scales = [
        (physical_max[i] - physical_min[i])
        / (digital_max[i] - digital_min[i] + 1e-15)
        for i in range(n_signals)
    ]
    offsets = [physical_min[i] - scales[i] * digital_min[i] for i in range(n_signals)]
    channel_data: list[list[float]] = [[] for _ in range(n_selected)]

    with path.open("rb") as file:
        file.seek(256 * (n_signals + 1))
        for _ in range(n_records):
            for signal_index in range(n_signals):
                raw_bytes = file.read(samples_per_record[signal_index] * 3)
                if signal_index >= n_selected:
                    continue
                samples = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
                values = (
                    samples[:, 0].astype(np.int32)
                    | (samples[:, 1].astype(np.int32) << 8)
                    | (samples[:, 2].astype(np.int32) << 16)
                )
                values[values >= 0x800000] -= 0x1000000
                calibrated = values * scales[signal_index] + offsets[signal_index]
                channel_data[signal_index].extend(calibrated.tolist())

    data = np.asarray(channel_data, dtype=np.float64)
    unit = physical_units[0].lower().replace(" ", "")
    if "uv" in unit or "microvolt" in unit:
        data *= 1e-6
    elif "mv" in unit or "millivolt" in unit:
        data *= 1e-3

    print(
        f"Loaded {path.name}: {data.shape[0]} channels x {data.shape[1]} samples "
        f"({labels[0]}-{labels[n_selected - 1]})"
    )
    return data


def load_noise_topographies(
    noise_file: Path | str,
    trial_count: int,
    seed: int = 0,
    n_topographies: int = N_NOISE_TOPOGRAPHIES,
) -> tuple[np.ndarray, np.random.Generator]:
    """Extract reproducible peak topographies from the noise recording."""
    data = read_bdf_channels(noise_file)
    filtered = high_pass_filter(data, NOISE_SAMPLING_RATE_HZ)
    window_samples = round(TRIAL_LENGTH_MS / 1000 * NOISE_SAMPLING_RATE_HZ)
    max_start = filtered.shape[1] - window_samples
    if max_start <= 0:
        raise ValueError("Noise recording is shorter than one analysis window")

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=n_topographies * 3)
    topographies = np.zeros((32, n_topographies), dtype=np.float64)

    for index, start in enumerate(starts[:n_topographies]):
        epoch = filtered[:, start : start + window_samples]
        peak = int(np.argmax(np.abs(epoch).sum(axis=0)))
        topographies[:, index] = filtered[:, start + peak]

    rms_uv = np.sqrt(np.mean(topographies**2)) * 1e6
    averaged_rms_uv = rms_uv / np.sqrt(trial_count)
    print(f"Noise RMS: {rms_uv:.3f} microvolts per epoch")
    print(f"Expected RMS after {trial_count} trials: {averaged_rms_uv:.3f} microvolts")
    return topographies, rng


def load_bem_models(
    bem_file: Path | str,
    electrode_sets: Mapping[int, Sequence[int]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load and normalize BEM topographies for spatial correlation."""
    bem = np.genfromtxt(bem_file, delimiter=",")
    if bem.ndim != 2 or bem.shape[0] < 32 or bem.shape[1] < 360:
        raise ValueError(f"Expected a BEM matrix of at least 32 x 360; got {bem.shape}")

    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for count, indices in electrode_sets.items():
        selected = bem[np.ix_(indices, MODEL_ANGLES_DEG)]
        centered = selected - selected.mean(axis=0)
        norm = np.sqrt(np.sum(centered**2, axis=0))
        models[count] = centered, norm
    return models


def spatial_correlations(
    topography: np.ndarray,
    models: Mapping[int, tuple[np.ndarray, np.ndarray]],
    electrode_sets: Mapping[int, Sequence[int]],
) -> dict[int, np.ndarray]:
    """Correlate one measured topography with every BEM orientation."""
    correlations: dict[int, np.ndarray] = {}
    for count, indices in electrode_sets.items():
        centered = topography[list(indices)]
        centered = centered - centered.mean()
        measured_norm = np.sqrt(np.sum(centered**2))
        model, model_norm = models[count]
        correlations[count] = (centered @ model) / (measured_norm * model_norm + 1e-15)
    return correlations


def _calibrated_indices(configuration: str, source_rotation: int, rotation: int) -> np.ndarray:
    return (
        np.round(
            CALIBRATION_INTERCEPT[configuration]
            + (source_rotation + MODEL_ANGLES_DEG - rotation)
            * CALIBRATION_SLOPE[configuration]
        ).astype(int)
        % 360
    )


def combined_correlation(
    correlations: Mapping[str, np.ndarray],
    sources: Mapping[str, tuple[str, int]],
    rotation: int,
) -> np.ndarray:
    """Combine calibrated correlations from all three contact pairs."""
    combined = np.zeros(360, dtype=np.float64)
    for pair in ("23", "34", "42"):
        source_configuration, source_rotation = sources[pair]
        indices = _calibrated_indices(source_configuration, source_rotation, rotation)
        combined += correlations[pair][indices]
    return combined


def single_pair_correlation(
    correlation: np.ndarray,
    source: tuple[str, int],
    rotation: int,
) -> np.ndarray:
    """Calibrate a single contact-pair correlation vector."""
    configuration, source_rotation = source
    return correlation[_calibrated_indices(configuration, source_rotation, rotation)]


def run_monte_carlo_analysis(
    cache_dir: Path | str,
    models: Mapping[int, tuple[np.ndarray, np.ndarray]],
    noise_topographies: np.ndarray,
    noise_rng: np.random.Generator,
    electrode_sets: Mapping[int, Sequence[int]],
    n_realisations: int = 32,
    trial_count: int = 1000,
    phantom_seed: int = 42,
    include_single_pair: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None]:
    """Estimate orientation for every rotation and Monte Carlo realization."""
    cache_dir = Path(cache_dir)
    phantom_rng = np.random.RandomState(phantom_seed)
    noise_count = noise_topographies.shape[1]
    replace_noise = noise_count < trial_count
    tri_estimates = {
        count: np.zeros((n_realisations, len(ROTATIONS_DEG)))
        for count in electrode_sets
    }
    single_estimates = (
        {
            count: np.zeros((n_realisations, len(ROTATIONS_DEG)))
            for count in electrode_sets
        }
        if include_single_pair
        else None
    )

    for rotation_index, rotation_value in enumerate(ROTATIONS_DEG):
        rotation = int(rotation_value)
        sources = {
            pair: source_for_rotation(pair, rotation)
            for pair in ("23", "34", "42")
        }
        cached = {
            pair: np.load(
                cache_dir / f"{configuration}_{source_rotation:03d}.npy"
            )
            for pair, (configuration, source_rotation) in sources.items()
        }
        available = cached["23"].shape[1]
        if any(values.shape != (32, available) for values in cached.values()):
            raise ValueError(f"Inconsistent cache shapes at rotation {rotation} degrees")
        if available < trial_count:
            raise ValueError(
                f"Requested {trial_count} trials, but only {available} are available "
                f"at rotation {rotation} degrees"
            )

        for realization in range(n_realisations):
            phantom_indices = phantom_rng.choice(available, trial_count, replace=False)
            averaged = {}
            for pair in ("23", "34", "42"):
                noise_indices = noise_rng.choice(
                    noise_count,
                    trial_count,
                    replace=replace_noise,
                )
                averaged[pair] = (
                    cached[pair][:, phantom_indices].mean(axis=1)
                    + noise_topographies[:, noise_indices].mean(axis=1)
                )

            correlations = {
                pair: spatial_correlations(averaged[pair], models, electrode_sets)
                for pair in ("23", "34", "42")
            }
            for count in electrode_sets:
                by_pair = {pair: correlations[pair][count] for pair in correlations}
                tri_estimates[count][realization, rotation_index] = np.argmax(
                    combined_correlation(by_pair, sources, rotation)
                )
                if single_estimates is not None:
                    single_estimates[count][realization, rotation_index] = np.argmax(
                        single_pair_correlation(
                            correlations["23"][count],
                            sources["23"],
                            rotation,
                        )
                    )

        if (rotation_index + 1) % 12 == 0:
            print(f"Completed {rotation_index + 1}/{len(ROTATIONS_DEG)} rotations")

    return tri_estimates, single_estimates


def contact_pair_errors(estimates: np.ndarray, separation_deg: int = 120) -> np.ndarray:
    """Calculate errors between orientations separated by a fixed angle."""
    aligned = align_estimates(estimates)
    rotation_to_index = {int(value): index for index, value in enumerate(ROTATIONS_DEG)}
    errors = np.zeros_like(aligned)
    for index, rotation_value in enumerate(ROTATIONS_DEG):
        comparison = rotation_to_index[(int(rotation_value) + separation_deg) % 360]
        errors[:, index] = wrap_to_180(
            aligned[:, comparison] - aligned[:, index] - separation_deg
        )
    return errors


def consecutive_step_errors(estimates: np.ndarray) -> np.ndarray:
    """Calculate error in the estimated change between adjacent rotations."""
    aligned = align_estimates(estimates)
    return wrap_to_180(np.diff(aligned, axis=1) - ROTATION_STEP_DEG)


def variability_half_width(values: np.ndarray) -> np.ndarray:
    """Return the historical variability measure used here: 1.96 x population SD."""
    return 1.96 * values.std(axis=0)
