# Holo-Hilbert Spectral Analysis (HHSA) in Python

Python Authors: Ting & Codex

This repository implements a compact, inspectable Holo-Hilbert Spectral Analysis pipeline for one-dimensional neural and biomedical time series. The code is designed to be easy to compare with MATLAB while still exposing a clean Python package interface.

The pipeline follows the holospectrum workflow:

1. Decompose the signal into IMFs with EMD, CEEMDAN, or ICEEMDAN.
2. Estimate instantaneous phase, frequency, and amplitude for each carrier IMF.
3. Decompose each carrier IMF's instantaneous amplitude with a second-layer sift.
4. Estimate amplitude-modulation frequency statistics from the second layer.
5. Build Hilbert-Huang and Holo-Hilbert spectra.

## Install

Install in editable mode from the repository root:

```bash
python3 -m pip install -e ".[dev,plot]"
```

For the full neuro-data workflow, install the optional neuro dependencies too:

```bash
python3 -m pip install -e ".[dev,plot,neuro]"
```

Run the test suite:

```bash
python3 -m pytest -q
```

## Quick Start

Use the class-based interface for notebooks and homework-style scripts:

```python
import numpy as np
from hhsa_tools import HHSAAnalyzer

sample_rate = 500.0
t = np.arange(0, 5, 1 / sample_rate)
signal = (1 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 18 * t)

analyzer = HHSAAnalyzer(
    sample_rate=sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=4,
    max_am_imfs=3,
    ensemble_size=16,
    noise_width=0.1,
)

result = analyzer.fit(signal)
summary = analyzer.summarize(result)

print(result.imfs.shape)
print(result.reconstruction_error)
print(result.hht.shape)           # carrier frequency x time
print(result.holospectrum.shape)  # carrier frequency x AM frequency
```

Use the functional interface when you want direct control:

```python
from hhsa import run_hhsa

result = run_hhsa(
    signal,
    sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=4,
    max_am_imfs=3,
    carrier_hist=(1, 100, 128, "log"),
    am_hist=(0.01, 32, 64, "log"),
)
```

## Outputs

`run_hhsa` returns an `HHSAResult` with the main arrays used in the Holo-Hilbert pipeline:

- `imfs`, `residue`: first-layer carrier IMFs and final residue.
- `amplitude`, `phase`, `frequency`: instantaneous amplitude, phase, and carrier frequency for first-layer IMFs.
- `am_imfs`, `am_residues`: second-layer IMFs from each first-layer amplitude envelope.
- `am_amplitude`, `am_phase`, `am_frequency`: instantaneous amplitude, phase, and AM frequency for second-layer IMFs.
- `carrier_bins`, `am_bins`: frequency bin centers used for spectra.
- `marginal`: 1D Hilbert-Huang spectrum over carrier frequency.
- `hht`: 2D Hilbert-Huang spectrum over carrier frequency x time.
- `holospectrum`: time-averaged Holo-Hilbert spectrum over carrier frequency x AM frequency.

The reconstruction helpers are:

```python
reconstructed = result.reconstruction
error = result.reconstruction_error
```

## ICEEMDAN

ICEEMDAN is available through the functional decomposition API.

```python
from hhsa import iceemdan

imfs, residue = iceemdan(signal, ensemble_size=100, noise_width=0.2, random_state=13)
```

## HHSA Workflow

1. Prepare a one-dimensional signal.
   Remove NaNs, detrend if needed, and keep the sampling rate in Hz.

2. Choose the first-layer decomposition.
   Use `decomposition="iceemdan"` for the main HHSA direction, `decomposition="ceemdan"` as a CEEMDAN baseline, and `decomposition="emd"` for fast debugging.

3. Choose the instantaneous-frequency estimator.
   Use `frequency_method="quad"` for Hilbert/quadrature phase, `"gzc"` for Generalized Zero-Crossing, or `"hybrid"` to combine both estimates.

4. Run the second layer.
   `run_hhsa` automatically decomposes every first-layer instantaneous-amplitude trace and stores the second-layer outputs in `result.am_imfs`, `result.am_amplitude`, and `result.am_frequency`.

5. Inspect spectra and quality.
   Check `result.reconstruction_error`, IMF energies with `mode_energy(result.imfs)`, the 2D HHT in `result.hht`, and the holospectrum in `result.holospectrum`.

## Verification

Run the included open-data example:

```bash
python3 examples/verify_open_ecg.py
```

The script first tries `scipy.datasets.electrocardiogram`, an open ECG signal derived from the MIT-BIH Arrhythmia Database. If SciPy cannot access the dataset cache, it falls back to a synthetic amplitude-modulated signal so the code path can still be checked offline.

For MATLAB comparison, export the same one-dimensional signal and Python outputs as `.mat` files, run MATLAB ICEEMDAN/HHT on that exact vector, and compare IMF count, reconstruction error, dominant instantaneous frequency, mode energy, and holospectrum peaks.

## OpenNeuro Target

Project pitch dataset:

- Dataset: `ds004078`, version `1.2.1`
- Source: <https://openneuro.org/datasets/ds004078/versions/1.2.1>
- Modality: MEG and fMRI
- Task: Chinese naturalistic story listening and comprehension

Recommended first MEG verification:

1. Download one subject and one short MEG run with OpenNeuro.
2. Load the MEG data with MNE-Python.
3. Pick one cleaned channel or source component.
4. Crop 5-20 seconds for fast iteration.
5. Run `run_hhsa(..., decomposition="iceemdan", frequency_method="hybrid")`.
6. Compare Python and MATLAB outputs on exactly the same cropped vector.

## Package Structure

```text
.
├── README.md
├── LICENSE
├── LICENSE.txt
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── hhsa/
│   ├── decomposition.py
│   ├── frequency.py
│   ├── pipeline.py
│   └── statistics.py
├── hhsa_tools/
│   ├── __init__.py
│   └── core.py
├── examples/
└── tests/
```

`hhsa` contains the research functions. `hhsa_tools` exposes the class-based interface for notebooks and coursework.

## Function Reference

### `hhsa_tools`

- `HHSAAnalyzer(sample_rate, ...)`: notebook-friendly class wrapper around the full HHSA pipeline. Store common analysis settings once, then call `fit(signal)` for each signal.
- `HHSAAnalyzer.sample_rate`: sampling rate in Hz, used by all frequency estimates.
- `HHSAAnalyzer.decomposition`: decomposition method for both HHSA layers. Use `"iceemdan"`, `"ceemdan"`, or `"emd"`.
- `HHSAAnalyzer.frequency_method`: instantaneous-frequency method. Use `"quad"`, `"gzc"`, or `"hybrid"`.
- `HHSAAnalyzer.max_imfs`: maximum number of first-layer carrier IMFs.
- `HHSAAnalyzer.max_am_imfs`: maximum number of second-layer amplitude-modulation IMFs per carrier.
- `HHSAAnalyzer.ensemble_size`: number of noise realizations for CEEMDAN/ICEEMDAN.
- `HHSAAnalyzer.noise_width`: relative noise scale for ensemble decomposition.
- `HHSAAnalyzer.random_state`: seed used for reproducible ensemble noise.
- `HHSAAnalyzer.fit(signal)`: runs `run_hhsa` with the analyzer's configured decomposition, frequency, and ensemble parameters. It returns the full `HHSAResult`.
- `HHSAAnalyzer.summarize(result)`: returns common summary outputs from an `HHSAResult`, including mode energy, frequency bins, marginal spectrum, HHT, holospectrum, and reconstruction error. The optional `bins` argument is retained for compatibility; current summaries use the bins stored on `result`.

### `hhsa.pipeline`

- `run_hhsa(signal, sample_rate, ...)`: main two-layer HHSA workflow. It decomposes the signal, computes instantaneous carrier statistics, decomposes amplitude envelopes, and builds HHT plus holospectrum arrays.
- `HHSAResult`: dataclass returned by `run_hhsa`. It stores first-layer IMFs, second-layer AM IMFs, instantaneous statistics, spectral bins, marginal spectrum, HHT, and holospectrum.
- `HHSAResult.reconstruction`: property that reconstructs the original signal from first-layer IMFs plus residue.
- `HHSAResult.reconstruction_error`: property that reports relative reconstruction error.

### `hhsa.decomposition`

- `emd(signal, ...)`: compact vanilla Empirical Mode Decomposition. Returns `(imfs, residue)`.
- `ceemdan(signal, ...)`: compact Complete Ensemble EMD with Adaptive Noise. Returns `(imfs, residue)` and is useful as a baseline.
- `iceemdan(signal, ...)`: Improved CEEMDAN-style decomposition following the MATLAB ICEEMDAN structure. Returns `(imfs, residue)` for direct use in HHSA.

### `hhsa.frequency`

- `quadrature_frequency(imf, sample_rate)`: computes instantaneous amplitude, phase, and frequency using the Hilbert transform.
- `generalized_zero_crossing(imf, sample_rate)`: estimates instantaneous frequency from zero crossings and extrema.
- `frequency_transform(imfs, sample_rate, method=...)`: applies a frequency estimator to every IMF. Supported methods are `"quad"`, `"gzc"`, and `"hybrid"`.

### `hhsa.statistics`

- `mode_energy(modes)`: returns sum-of-squares energy for each IMF.
- `normalized_entropy(values)`: computes Shannon entropy normalized to `[0, 1]`.
- `orthogonality_index(modes, signal)`: estimates how much IMF energy leaks across modes; lower values indicate cleaner separation.
- `marginal_spectrum(frequency, amplitude, ...)`: computes a simple Hilbert marginal spectrum from instantaneous frequency and amplitude arrays.
- `spectrum_bin_edges(hist)`: converts an EMD-style histogram tuple such as `(1, 100, 128, "log")` into bin centers and edges.
- `hilbert_huang_spectrum(frequency, amplitude, hist)`: builds the 1D marginal spectrum and 2D HHT over carrier frequency x time.
- `holospectrum(carrier_frequency, am_frequency, am_amplitude, carrier_hist, am_hist)`: builds the time-averaged Holo-Hilbert spectrum over carrier frequency x AM frequency.

## Code Map

- `hhsa/decomposition.py`: EMD, CEEMDAN, and ICEEMDAN decomposition functions.
- `hhsa/frequency.py`: quadrature/Hilbert and Generalized Zero-Crossing frequency estimators.
- `hhsa/pipeline.py`: two-layer HHSA pipeline and `HHSAResult`.
- `hhsa/statistics.py`: mode statistics, Hilbert-Huang spectrum, and holospectrum helpers.
- `hhsa_tools/core.py`: class-based `HHSAAnalyzer`.
- `examples/verify_open_ecg.py`: runnable verification example.
- `tests/`: focused tests for decomposition, frequency estimation, and HHSA outputs.
