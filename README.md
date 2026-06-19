# Holo-Hilbert Spectral Analysis (HHSA) in Python

Python Authors: Ting & Codex

This repository implements a compact, inspectable Holo-Hilbert Spectral Analysis pipeline for one-dimensional neural and biomedical time series. The code is designed to be easy to compare with MATLAB while still exposing a clean Python package interface.

The pipeline accepts one-dimensional signals, multi-channel arrays, WAV audio,
and MNE-Python Raw/Epochs/Evoked-style EEG/MEG objects. EMD is imported from
EMD-Python or PyEMD; CEEMDAN is imported from PyEMD.

The pipeline follows the holospectrum workflow:

1. Decompose the signal into IMFs with EMD, CEEMDAN, or ICEEMDAN.
2. Estimate instantaneous phase, frequency, and amplitude for each carrier IMF.
3. Decompose each carrier IMF's instantaneous amplitude with a second-layer sift.
4. Estimate amplitude-modulation frequency statistics from the second layer.
5. Build Hilbert-Huang and Holo-Hilbert spectra.

## Install

Choose one of two install paths.

### Option 1: pip install

Install from PyPI after the package is published:

```bash
python3 -m pip install hhsa_tools
```

This installs the core scientific stack plus EMD-Python (`emd`). Add extras
when you need optional integrations:

```bash
python3 -m pip install "hhsa_tools[ceemdan,neuro,plot,notebook]"
```

`ceemdan` installs PyEMD (`EMD-signal`, imported as `PyEMD`), and `neuro`
installs MNE-Python (`mne`).

### Option 2: download the full repository package

Download or clone the full repository, then install it from the repository root:

```bash
git clone https://github.com/DHCLAIRE/HHSA-Py.git
cd HHSA-Py
python3 -m pip install .
```

For development or notebook work from the downloaded repository, install extras:

```bash
python3 -m pip install -e ".[dev,all]"
```

After installing from the repository, run the test suite:

```bash
python3 -m pytest -q
```

## EMD-Python Integration

This package is structured to merge cleanly with EMD-Python workflows:

- Decomposition uses `emd.sift.sift`, `emd.sift.ensemble_sift`,
  `emd.sift.complete_ensemble_sift`, `emd.sift.mask_sift`, and
  `emd.sift.iterated_mask_sift` when EMD-Python is available.
- Instantaneous phase, frequency, and amplitude can use
  `emd.spectra.frequency_transform` for EMD-Python methods.
- Hilbert-Huang and Holo-Hilbert spectra call `emd.spectra.hilberthuang` and
  `emd.spectra.holospectrum` when compatible, with local fallbacks for
  portability.
- HHSA stores IMFs as `modes x samples`; EMD-Python returns many arrays as
  `samples x modes`. The package conversion happens at the API boundary.

## Quick Start

Use the class-based interface for notebooks and homework-style scripts:

```python
import numpy as np
from hhsa_tools import HHSAPipeline

sample_rate = 500.0
t = np.arange(0, 5, 1 / sample_rate)
signal = (1 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 18 * t)

pipeline = HHSAPipeline(
    sample_rate=sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=10,
    max_am_imfs=3,
    ensemble_size=16,
    noise_width=0.1,
)

result = pipeline.fit(signal)
summary = pipeline.summarize(result)

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
    max_imfs=10,
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

## Decomposition Options

The HHSA pipeline uses EMD-Python's public `emd.sift` API when
`emd_backend="emd-python"` or when `emd_backend="auto"` can import it.
Returned IMFs are converted from EMD-Python's `samples x modes` convention to
HHSA's `modes x samples` convention.

- `decomposition="sift"` or `"emd"`: EMD-Python standard sift.
- `decomposition="ensemble_sift"`: EMD-Python ensemble sift.
- `decomposition="complete_ensemble_sift"`: EMD-Python complete ensemble sift.
- `decomposition="mask_sift"`: EMD-Python masking sift. Use `mask_freqs`, `mask_amp`, and `mask_amp_mode` to control masks.
- `decomposition="iterated_mask_sift"`: EMD-Python iterated masking sift.
- `decomposition="ceemdan"`: PyEMD CEEMDAN.
- `decomposition="iceemdan"`: project ICEEMDAN wrapper using imported EMD calls internally.

```python
result = run_hhsa(
    signal,
    sample_rate,
    decomposition="mask_sift",
    mask_freqs=20 / sample_rate,
    mask_amp=1.0,
    mask_amp_mode="ratio_sig",
)
```

## Sifting Acceleration

Sifting is the expensive part of EMD: each IMF repeatedly finds extrema,
builds upper/lower envelopes, subtracts their mean, and checks a stopping rule.
For noise-assisted EMD variants, the best acceleration target is the independent
ensemble trials around those sifts rather than the sequential inner sifting loop
itself. EMD was introduced by Huang et al. (1998), EEMD by Wu and Huang (2009),
and ICEEMDAN by Colominas et al. (2014).

CPU acceleration is available without extra packages for CEEMDAN and ICEEMDAN:

```python
result = run_hhsa(
    signal,
    sample_rate,
    decomposition="iceemdan",
    ensemble_size=100,
    sift_acceleration="cpu",
    n_jobs=-1,
)
```

GPU acceleration for the project ICEEMDAN path is explicit and optional.
Install the CuPy build that matches your CUDA runtime, then request the GPU
path. The implementation accelerates ensemble noise generation and reductions
with CuPy while keeping EMD-Python or PyEMD responsible for the numerically
sensitive sifting calls. PyEMD CEEMDAN exposes CPU parallel trials, not GPU
sifting.

```python
result = run_hhsa(
    signal,
    sample_rate,
    decomposition="iceemdan",
    ensemble_size=100,
    sift_acceleration="gpu",
    n_jobs=-1,
)
```

Use `sift_acceleration="none"` for the original serial behavior,
`"cpu"` for parallel CPU ensemble workers, `"gpu"` for CuPy-assisted ensemble
work, or `"auto"` to use CuPy when available and CPU workers otherwise.
For typical short EEG/MEG epochs, CPU ensemble parallelism is often the fastest
minimum-command option; GPU becomes more useful as channel count, ensemble size,
or signal length grows.

References: Huang et al., "The empirical mode decomposition and the Hilbert
spectrum for nonlinear and non-stationary time series analysis,"
Proc. R. Soc. A, 1998, https://doi.org/10.1098/rspa.1998.0193;
Wu and Huang, "Ensemble empirical mode decomposition: A noise-assisted data
analysis method," Advances in Adaptive Data Analysis, 2009,
https://doi.org/10.1142/S1793536909000047; Colominas et al., "Improved complete
ensemble EMD: A suitable tool for biomedical signal processing," Biomedical
Signal Processing and Control, 2014, https://doi.org/10.1016/j.bspc.2014.06.009.

## Tutorials

### Tutorial 1: Run HHSA With `HHSAPipeline`

Use this path when you want one reusable object for repeated analyses.

```python
import numpy as np
from hhsa_tools import HHSAPipeline

sample_rate = 200.0
t = np.arange(0, 3, 1 / sample_rate)
signal = (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * 20 * t)

pipeline = HHSAPipeline(
    sample_rate=sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=10,
    max_am_imfs=4,
    ensemble_size=32,
    noise_width=0.1,
    random_state=13,
)

result = pipeline.fit(signal)
summary = pipeline.summarize(result)

print("Carrier IMFs:", result.imfs.shape)
print("Second-layer IMF groups:", len(result.am_imfs))
print("Reconstruction error:", summary["reconstruction_error"])
```

### Tutorial 2: Run the Functional HHSA API

Use this path when you want to set every option directly in one call.

```python
from hhsa import run_hhsa

result = run_hhsa(
    signal,
    sample_rate,
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=10,
    max_am_imfs=4,
    ensemble_size=32,
    noise_width=0.1,
    carrier_hist=(1, 100, 128, "log"),
    am_hist=(0.01, 32, 64, "log"),
    random_state=13,
)

print(result.carrier_bins.shape)
print(result.am_bins.shape)
print(result.hht.shape)
print(result.holospectrum.shape)
```

### Tutorial 3: Use ICEEMDAN Alone

Use this path when you only need decomposition and not the full Holo-Hilbert spectrum.

```python
from hhsa import iceemdan

imfs, residue = iceemdan(
    signal,
    max_imfs=10,
    ensemble_size=100,
    noise_width=0.2,
    random_state=13,
)

reconstruction = imfs.sum(axis=0) + residue
```

### Tutorial 4: Plot HHT and Holospectrum

Install plot extras first:

```bash
python3 -m pip install -e ".[plot]"
```

Then plot the two main spectrum outputs:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.pcolormesh(result.carrier_bins, np.arange(result.hht.shape[1]), result.hht.T, shading="auto")
plt.xlabel("Carrier frequency (Hz)")
plt.ylabel("Sample")
plt.title("Hilbert-Huang Transform")

plt.figure()
plt.pcolormesh(result.am_bins, result.carrier_bins, result.holospectrum, shading="auto")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("AM frequency (Hz)")
plt.ylabel("Carrier frequency (Hz)")
plt.title("Holo-Hilbert Spectrum")
plt.show()
```

### Tutorial 5: Plot Sifting Options and AM/FM Tracks

Use the built-in visualization helpers to inspect decomposition outputs and
instantaneous amplitude/frequency tracks.

```python
from hhsa import plot_am_fm, plot_decomposition, plot_sifting_options

fig, axes = plot_sifting_options(
    signal,
    sample_rate,
    methods=(
        "sift",
        "ensemble_sift",
        "complete_ensemble_sift",
        "mask_sift",
        "iterated_mask_sift",
        "ceemdan",
        "iceemdan",
    ),
    max_imfs=5,
    mask_freqs=20 / sample_rate,
)

fig, axes = plot_decomposition(
    result.signal,
    result.imfs,
    result.residue,
    sample_rate=result.sample_rate,
)

fig, axes = plot_am_fm(result, max_modes=5)
```

### Tutorial 6: Run EEG or MEG Data From MNE-Python

Use this path for MNE `Raw`, `Epochs`, or `Evoked` objects. Sampling rate is
read from `raw.info["sfreq"]`, and data are analyzed one channel at a time.

```python
from hhsa_tools import HHSAPipeline

pipeline = HHSAPipeline(
    decomposition="iceemdan",
    frequency_method="hybrid",
    max_imfs=10,
    max_am_imfs=4,
    ensemble_size=32,
)

results = pipeline.fit(raw, picks="eeg")  # or picks="meg"
summaries = pipeline.summarize(results)
print(len(results))
```

### Tutorial 7: Run Multi-Channel Audio

WAV paths are read directly. Stereo or multi-channel audio returns one
`HHSAResult` per channel.

```python
from hhsa_tools import HHSAPipeline

pipeline = HHSAPipeline(decomposition="iceemdan", max_imfs=10)
results = pipeline.fit("example_audio.wav")

for channel_index, result in enumerate(results):
    print(channel_index, result.holospectrum.shape)
```

For NumPy audio arrays shaped as samples x channels, set `channel_axis="last"`:

```python
pipeline = HHSAPipeline(sample_rate=44100, decomposition="iceemdan", max_imfs=10)
results = pipeline.fit(stereo_array, channel_axis="last")
```

### Tutorial 8: Run Group Statistics

After running HHSA for two groups, compare features such as `mode_energy`,
`marginal`, `hht`, `holospectrum`, or `am_frequency`.

```python
pipeline = HHSAPipeline(sample_rate=sample_rate, decomposition="sift", max_imfs=10)

group_a_results = pipeline.fit(group_a_array, channel_axis="first")
group_b_results = pipeline.fit(group_b_array, channel_axis="first")

t_result = pipeline.t_test(group_a_results, group_b_results, feature="mode_energy")
perm_result = pipeline.permutation_test(
    group_a_results,
    group_b_results,
    feature="mode_energy",
    n_permutations=1000,
    random_state=13,
)

print(t_result.statistic, t_result.pvalue)
print(perm_result.statistic, perm_result.pvalue)
```

## HHSA Workflow

1. Prepare a one-dimensional signal.
   For arrays, remove NaNs, detrend if needed, and keep the sampling rate in Hz. For MNE objects and WAV paths, HHSA can infer the sampling rate.

2. Choose the first-layer decomposition.
   Use `decomposition="sift"` or `"emd"` for EMD-Python standard sift, `"ensemble_sift"` for EMD-Python ensemble sift, `"complete_ensemble_sift"` for EMD-Python complete ensemble sift, `"mask_sift"` for EMD-Python masking sift, `"iterated_mask_sift"` for EMD-Python iterated masking sift, `"ceemdan"` for PyEMD CEEMDAN, and `"iceemdan"` for the ICEEMDAN wrapper.

3. Choose the instantaneous-frequency estimator.
   Use `frequency_method="quad"` for Hilbert/quadrature phase, `"gzc"` for Generalized Zero-Crossing, `"hybrid"` to combine both estimates, or EMD-Python-compatible `"hilbert"`, `"direct_quad"`, and `"nht"` methods when EMD-Python is installed.

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
├── LICENSE.txt
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── hhsa/
│   ├── decomposition.py
│   ├── frequency.py
│   ├── pipeline.py
│   ├── statistics.py
│   └── visualization.py
├── hhsa_tools/
│   ├── __init__.py
│   └── core.py
├── examples/
└── tests/
```

`hhsa` contains the core implementation; `hhsa_tools` contains the notebook-friendly pipeline wrapper.

## Function Reference

### `hhsa_tools`

- `HHSAPipeline(sample_rate, ...)`: notebook-friendly class wrapper around the full HHSA pipeline. Store common analysis settings once, then call `fit(signal)` for each signal.
- `HHSAPipeline.sample_rate`: sampling rate in Hz, used by all frequency estimates.
- `HHSAPipeline.decomposition`: decomposition method for both HHSA layers. Use `"sift"`, `"emd"`, `"ensemble_sift"`, `"complete_ensemble_sift"`, `"mask_sift"`, `"iterated_mask_sift"`, `"ceemdan"`, or `"iceemdan"`.
- `HHSAPipeline.frequency_method`: instantaneous-frequency method. Use `"quad"`, `"gzc"`, `"hybrid"`, `"hilbert"`, `"direct_quad"`, or `"nht"`.
- `HHSAPipeline.max_imfs`: maximum number of first-layer carrier IMFs. The default is `10`.
- `HHSAPipeline.max_am_imfs`: maximum number of second-layer amplitude-modulation IMFs per carrier.
- `HHSAPipeline.ensemble_size`: number of noise realizations for CEEMDAN/ICEEMDAN.
- `HHSAPipeline.noise_width`: relative noise scale for ensemble decomposition.
- `HHSAPipeline.random_state`: seed used for reproducible ensemble noise.
- `HHSAPipeline.emd_backend`: EMD backend selector. Use `"auto"`, `"emd-python"`, or `"pyemd"`.
- `HHSAPipeline.mask_freqs`, `mask_amp`, `mask_amp_mode`: mask-sift controls passed to EMD-Python when `decomposition="mask_sift"`.
- `HHSAPipeline.fit(signal)`: runs HHSA with the pipeline's stored decomposition, frequency, and ensemble parameters. It accepts 1-D arrays, multi-channel arrays, WAV paths, and MNE Raw/Epochs/Evoked-like objects.
- `HHSAPipeline.summarize(result)`: returns common summary outputs from an `HHSAResult`, including mode energy, frequency bins, marginal spectrum, HHT, holospectrum, and reconstruction error. For multi-channel input, it returns one summary per channel.
- `HHSAPipeline.t_test(group_a, group_b, feature=...)`: runs a feature-wise independent t-test on HHSA result groups.
- `HHSAPipeline.permutation_test(group_a, group_b, feature=...)`: runs a two-sided feature-wise permutation test on HHSA result groups.

### `hhsa.pipeline`

- `run_hhsa(signal, sample_rate, ...)`: main two-layer HHSA workflow. It decomposes the signal, computes instantaneous carrier statistics, decomposes amplitude envelopes, and builds HHT plus holospectrum arrays. The default first-layer limit is `max_imfs=10`.
- `run_hhsa_dataset(data, sample_rate=None, ...)`: runs HHSA independently for every channel in a 2-D array, WAV path, or MNE object.
- `as_channel_matrix(data, ...)`: converts EEG, MEG, audio, or array input to channels x samples for consistent processing.
- `HHSAResult`: dataclass returned by `run_hhsa`. It stores first-layer IMFs, second-layer AM IMFs, instantaneous statistics, spectral bins, marginal spectrum, HHT, and holospectrum.
- `HHSAResult.reconstruction`: property that reconstructs the original signal from first-layer IMFs plus residue.
- `HHSAResult.reconstruction_error`: property that reports relative reconstruction error.

### `hhsa.decomposition`

- `emd(signal, ...)`: Empirical Mode Decomposition wrapper. With `backend="auto"`, it tries EMD-Python, then PyEMD. Returns `(imfs, residue)`.
- `ensemble_sift(signal, ...)`: EMD-Python ensemble sift wrapper. Returns `(imfs, residue)`.
- `complete_ensemble_sift(signal, ...)`: EMD-Python complete ensemble sift wrapper. Returns `(imfs, residue)`.
- `mask_sift(signal, ...)`: EMD-Python masking sift wrapper. Returns `(imfs, residue)`.
- `iterated_mask_sift(signal, ...)`: EMD-Python iterated masking sift wrapper. Returns `(imfs, residue)`.
- `decompose_signal(signal, method, ...)`: public dispatcher used by the pipeline and visualization helpers.
- `ceemdan(signal, ...)`: PyEMD Complete Ensemble EMD with Adaptive Noise wrapper. Returns `(imfs, residue)` and supports `sift_acceleration="cpu"` plus `n_jobs` for PyEMD parallel trials.
- `iceemdan(signal, ...)`: Improved CEEMDAN-style decomposition following the MATLAB ICEEMDAN structure. Returns `(imfs, residue)` and supports `sift_acceleration="cpu"`, `"gpu"`, or `"auto"` with `n_jobs`.

### `hhsa.frequency`

- `quadrature_frequency(imf, sample_rate)`: computes instantaneous amplitude, phase, and frequency using the Hilbert transform.
- `generalized_zero_crossing(imf, sample_rate)`: estimates instantaneous frequency from zero crossings and extrema.
- `frequency_transform(imfs, sample_rate, method=...)`: applies a frequency estimator to every IMF. Supported methods are `"quad"`, `"gzc"`, `"hybrid"`, `"hilbert"`, `"direct_quad"`, and `"nht"`. EMD-Python methods are delegated to `emd.spectra.frequency_transform` when available.

### `hhsa.statistics`

- `mode_energy(modes)`: returns sum-of-squares energy for each IMF.
- `hhsa_feature(result, feature=...)`: extracts one statistical feature vector from an `HHSAResult`.
- `hhsa_feature_matrix(results, feature=...)`: stacks HHSA result features into a padded matrix.
- `hhsa_t_test(group_a, group_b, feature=...)`: runs a feature-wise independent t-test on two HHSA result groups.
- `hhsa_permutation_test(group_a, group_b, feature=...)`: runs a two-sided feature-wise permutation test on two HHSA result groups.
- `normalized_entropy(values)`: computes Shannon entropy normalized to `[0, 1]`.
- `orthogonality_index(modes, signal)`: estimates how much IMF energy leaks across modes; lower values indicate cleaner separation.
- `marginal_spectrum(frequency, amplitude, ...)`: computes a simple Hilbert marginal spectrum from instantaneous frequency and amplitude arrays.
- `spectrum_bin_edges(hist)`: converts an EMD-style histogram tuple such as `(1, 100, 128, "log")` into bin centers and edges.
- `hilbert_huang_spectrum(frequency, amplitude, hist)`: builds the 1D marginal spectrum and 2D HHT over carrier frequency x time.
- `holospectrum(carrier_frequency, am_frequency, am_amplitude, carrier_hist, am_hist)`: builds the time-averaged Holo-Hilbert spectrum over carrier frequency x AM frequency.

### `hhsa.visualization`

- `plot_decomposition(signal, imfs, residue, ...)`: plots a signal, its IMFs, and final residue.
- `plot_sifting_options(signal, sample_rate, ...)`: compares IMF stacks from sifting, ensemble sifting, mask sifting, CEEMDAN, and ICEEMDAN options.
- `plot_am_fm(result, ...)`: plots instantaneous amplitude and instantaneous frequency tracks from an `HHSAResult`.
