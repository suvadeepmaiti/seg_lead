# Phantom analysis

This folder contains the phantom portion of the manuscript analysis.

```text
phantom/
|-- code/
|   |-- analysis_common.py
|   |-- build_cache.py
|   |-- panel_a.py
|   |-- panel_b.py
|   `-- panel_cd.py
|-- data/
|   |-- raw/turntable_measurements/
|   |-- processed/topographies/
|   `-- reference/
|       |-- bem_reference.csv
|       `-- single_pair_no_noise.mat
|-- results/
`-- requirements.txt
```

The panel scripts combine the phantom measurements with the de-identified
background recording at `../human/data/raw/noise_recording.bdf`. Outputs are
written to `phantom/results/`.

Run from the repository root:

```bash
python phantom/code/panel_a.py --no-show
python phantom/code/panel_b.py --no-show
python phantom/code/panel_cd.py --no-show
```

To rebuild the processed phantom cache:

```bash
python phantom/code/build_cache.py --overwrite
```

The random seeds are fixed by default. Each script supports `--help` for all
available options. Variability is reported as 1.96 times the population
standard deviation across Monte Carlo realizations.
