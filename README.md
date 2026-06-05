# Holo-Hilbert Spectral Analysis (HHSA) in Python

This project builds a teachable HHSA pipeline for neural-signal processing:

1. Decompose the signal with EMD, CEEMDAN, or an ICEEMDAN-style method.
2. Estimate instantaneous frequency with Generalized Zero-Crossing (GZC), quadrature/Hilbert phase, or a hybrid of both.
3. Run the second EMD layer on the first-layer amplitude envelopes.
4. Summarize modes with optional statistics.

The current code is intentionally compact so it is easy to inspect and compare against MATLAB.

## Install

```bash
python3 -m pip install -e ".[dev,plot]"
```

If you do not want editable installation yet, run scripts from the repository root.

## Quick Verification

Run the local tests:

```bash
python3 -m pytest -q
```

Run the open-data verification example:

```bash
python3 examples/verify_open_ecg.py
```

The script first tries `scipy.datasets.electrocardiogram`, an open ECG signal derived from the MIT-BIH Arrhythmia Database. If SciPy cannot access the dataset cache, it uses a synthetic amplitude-modulated signal so the code path can still be checked offline.

## Minimal Use

```python
import numpy as np
from hhsa import run_hhsa, marginal_spectrum

sample_rate = 500.0
t = np.arange(0, 5, 1 / sample_rate)
signal = (1 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 18 * t)

result = run_hhsa(
    signal,
    sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=4,
    max_am_imfs=3,
    ensemble_size=16,
    noise_width=0.1,
)

freq_bins, marginal = marginal_spectrum(result.frequency, result.amplitude)
print(result.imfs.shape)
print(result.reconstruction_error)
```

## Step-By-Step HHSA Workflow

1. Prepare the signal.
   Use a one-dimensional channel or component, remove NaNs, detrend if needed, and keep the sampling rate in Hz.

2. Run the first decomposition layer.
   Use `decomposition="iceemdan"` for your BrainHack direction. Use `decomposition="ceemdan"` when you want the closer CEEMDAN baseline, and `decomposition="emd"` for quick debugging.

3. Estimate HHT frequency.
   Use `frequency_method="quad"` for Hilbert/quadrature phase, `"gzc"` for Generalized Zero-Crossing, or `"hybrid"` to merge both estimates.

4. Run the second decomposition layer.
   `run_hhsa` automatically decomposes every first-layer amplitude envelope. These second-layer modes are stored in `result.am_imfs`; their modulation frequencies are in `result.am_frequency`.

5. Inspect quality.
   Check `result.reconstruction_error`, IMF energies with `mode_energy(result.imfs)`, and the Hilbert marginal spectrum with `marginal_spectrum`.

6. Verify on open data.
   Start with the ECG example because it is small. Then download the Brain Language Processing dataset from OpenNeuro: `ds004078`, version `1.2.1`.

7. Compare with MATLAB.
   Export the same one-dimensional signal and Python outputs as `.mat` files, run MATLAB ICEEMDAN/HHT on the same signal, and compare IMF count, reconstruction error, dominant instantaneous frequency, and mode energy.

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
6. Compare Python results with MATLAB on exactly the same cropped vector.

## Code Map

- `hhsa/decomposition.py`: EMD, CEEMDAN, and ICEEMDAN-style decomposition.
- `hhsa/frequency.py`: GZC and quadrature frequency estimation.
- `hhsa/pipeline.py`: two-layer HHSA pipeline.
- `hhsa/statistics.py`: marginal spectrum, mode energy, entropy, and orthogonality index.
- `examples/verify_open_ecg.py`: runnable verification example.
- `tests/test_hhsa.py`: focused tests for frequency estimation and the HHSA pipeline.
