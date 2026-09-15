# Electroencephalography for Radiation-Free Orientation Estimation of Directional Deep Brain Stimulation Leads

Code and de-identified data supporting the manuscript:

**Suvadeep Maiti<sup>a,c,*</sup>, Alan Bince Jacob<sup>a,*</sup>, Marco
Mancuso<sup>b</sup>, Marie T. Krueger<sup>b</sup>, Harith Akram<sup>b</sup>,
Kirill Aristovich<sup>a</sup>, and Vladimir Litvak<sup>c,**</sup>**

<sup>a</sup> Department of Medical Physics and Biomedical Engineering,
University College London, London, UK  
<sup>b</sup> Unit of Functional Neurosurgery, UCL Queen Square Institute of
Neurology, London, UK  
<sup>c</sup> Department of Imaging Neuroscience, UCL Queen Square Institute of
Neurology, London, UK

<sup>*</sup> Equal contribution  
<sup>**</sup> Corresponding author

The repository separates the phantom and human analyses while keeping both
parts of the study in one reproducible codebase.

## Repository layout

```text
.
|-- phantom/    # Phantom code, data, references, and results
|-- human/      # De-identified human data, code, and results
|-- .gitattributes
|-- .gitignore
`-- README.md
```

See [phantom/README.md](phantom/README.md) and
[human/README.md](human/README.md) for the contents of each work area.

## Git LFS

All BDF recordings are tracked with Git LFS. Install it before cloning or
pushing this repository:

```bash
git lfs install
```

Python 3.10 or newer is recommended.

## Quick start on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r phantom/requirements.txt
python phantom/code/panel_a.py --no-show
python phantom/code/panel_b.py --no-show
python phantom/code/panel_cd.py --no-show
```

## Quick start on Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r phantom\requirements.txt
python phantom\code\panel_a.py --no-show
python phantom\code\panel_b.py --no-show
python phantom\code\panel_cd.py --no-show
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again.

In VS Code, select `.venv/bin/python` on macOS or
`.venv\Scripts\python.exe` on Windows as the Python interpreter.

## Data governance

The human work area contains de-identified EEG data. Do not add participant
keys, consent forms, direct identifiers, or re-identification metadata. Confirm
that repository access and publication are consistent with the applicable
ethics approval and institutional data-sharing policy.
