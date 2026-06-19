"""EMD, CEEMDAN, and ICEEMDAN-style decomposition.

EMD and CEEMDAN are delegated to external libraries: EMD-Python (``emd``)
and PyEMD (``PyEMD``).
ICEEMDAN remains project code, but its internal EMD calls use those libraries.
"""

from __future__ import annotations

from importlib import import_module
from typing import Literal

import numpy as np

EMDBackend = Literal["auto", "emd-python", "pyemd"]
EMDPythonSift = Literal["sift", "ensemble_sift", "mask_sift"]


def _max_imfs_for_external(max_imfs: int | None) -> int:
    """Convert optional IMF limits to the convention used by EMD libraries."""

    return -1 if max_imfs is None else int(max_imfs)


def _as_1d(signal: np.ndarray) -> np.ndarray:
    """Convert input to a float 1-D array and validate its minimum length."""

    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 4:
        raise ValueError("signal must contain at least four samples")
    return x


def _emd_with_emd_python(
    signal: np.ndarray,
    *,
    max_imfs: int | None,
    max_siftings: int,
    stop_sd: float,
    sift_method: EMDPythonSift,
    ensemble_size: int,
    noise_width: float,
    mask_freqs: np.ndarray | float | None,
    mask_amp: float,
    mask_amp_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an EMD-Python sift variant and convert output orientation."""

    x = _as_1d(signal)
    sift_module = import_module("emd.sift")
    sift_function = getattr(sift_module, sift_method)
    imf_opts = {"sd_thresh": stop_sd, "max_iters": max_siftings}
    if sift_method == "sift":
        kwargs = {"max_imfs": max_imfs, "imf_opts": imf_opts}
    elif sift_method == "ensemble_sift":
        kwargs = {
            "max_imfs": max_imfs,
            "nensembles": ensemble_size,
            "ensemble_noise": noise_width,
            "imf_opts": imf_opts,
        }
    elif sift_method == "mask_sift":
        kwargs = {
            "max_imfs": max_imfs,
            "mask_freqs": mask_freqs,
            "mask_amp": mask_amp,
            "mask_amp_mode": mask_amp_mode,
            "imf_opts": imf_opts,
        }
    else:
        raise ValueError("sift_method must be 'sift', 'ensemble_sift', or 'mask_sift'")
    try:
        modes = sift_function(x, **kwargs)
    except TypeError:
        kwargs.pop("imf_opts", None)
        modes = sift_function(x, **kwargs)
    modes = np.asarray(modes, dtype=float)
    if modes.ndim == 1:
        modes = modes[:, np.newaxis]
    if modes.shape[0] == x.size:
        modes = modes.T
    if max_imfs is not None:
        modes = modes[:max_imfs]
    residue = x - modes.sum(axis=0) if modes.size else x.copy()
    return modes, residue


def _emd_with_pyemd(
    signal: np.ndarray,
    *,
    max_imfs: int | None,
    max_siftings: int,
    stop_sd: float,
    envelope_mean_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PyEMD's EMD implementation and return package-native orientation."""

    x = _as_1d(signal)
    pyemd = import_module("PyEMD")
    decomposer = pyemd.EMD()
    decomposer.MAX_ITERATION = max_siftings
    decomposer.FIXE_H = max_siftings
    decomposer.std_thr = stop_sd
    decomposer.range_thr = envelope_mean_tol
    modes = np.asarray(decomposer.emd(x, max_imf=_max_imfs_for_external(max_imfs)), dtype=float)
    if modes.ndim == 1:
        modes = modes[np.newaxis, :]
    if max_imfs is not None:
        modes = modes[:max_imfs]
    residue = x - modes.sum(axis=0) if modes.size else x.copy()
    return modes, residue


def emd(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
    envelope_mean_tol: float = 0.1,
    backend: EMDBackend = "auto",
    sift_method: EMDPythonSift = "sift",
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal into IMFs and a residue.

    ``backend="auto"`` tries EMD-Python first, then PyEMD.
    """

    backends = ["emd-python", "pyemd"] if backend == "auto" else [backend]
    last_error: Exception | None = None
    for selected in backends:
        try:
            if selected == "emd-python":
                return _emd_with_emd_python(
                    signal,
                    max_imfs=max_imfs,
                    max_siftings=max_siftings,
                    stop_sd=stop_sd,
                    sift_method=sift_method,
                    ensemble_size=ensemble_size,
                    noise_width=noise_width,
                    mask_freqs=mask_freqs,
                    mask_amp=mask_amp,
                    mask_amp_mode=mask_amp_mode,
                )
            if selected == "pyemd":
                if sift_method != "sift":
                    raise ValueError("PyEMD backend only supports sift_method='sift'")
                return _emd_with_pyemd(
                    signal,
                    max_imfs=max_imfs,
                    max_siftings=max_siftings,
                    stop_sd=stop_sd,
                    envelope_mean_tol=envelope_mean_tol,
                )
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
            last_error = exc
            if backend != "auto":
                raise
    if last_error is not None:
        raise last_error
    raise ValueError("backend must be 'auto', 'emd-python', or 'pyemd'")


def ensemble_sift(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with EMD-Python's ensemble sift."""

    return emd(
        signal,
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        backend="emd-python",
        sift_method="ensemble_sift",
        ensemble_size=ensemble_size,
        noise_width=noise_width,
    )


def mask_sift(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with EMD-Python's mask sift."""

    return emd(
        signal,
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        backend="emd-python",
        sift_method="mask_sift",
        mask_freqs=mask_freqs,
        mask_amp=mask_amp,
        mask_amp_mode=mask_amp_mode,
    )


def _normalize_noise(noise: np.ndarray) -> np.ndarray:
    """Scale a noise vector to unit standard deviation when possible."""

    std = np.std(noise)
    if std == 0:
        return noise
    return noise / std


def _can_extract_external_imf(
    signal: np.ndarray,
    *,
    max_siftings: int,
    stop_sd: float,
    envelope_mean_tol: float,
    emd_backend: EMDBackend,
) -> bool:
    """Use the selected EMD library to decide whether another IMF exists."""

    if np.allclose(signal, 0):
        return False
    modes, _ = emd(
        signal,
        max_imfs=1,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        envelope_mean_tol=envelope_mean_tol,
        backend=emd_backend,
    )
    return bool(modes.size and not np.allclose(modes[0], 0))


def ceemdan(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | np.random.Generator | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
    envelope_mean_tol: float = 0.1,
    emd_backend: EMDBackend = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with PyEMD's CEEMDAN implementation."""

    x = _as_1d(signal)
    pyemd = import_module("PyEMD")
    ext_emd = pyemd.EMD()
    ext_emd.MAX_ITERATION = max_siftings
    ext_emd.std_thr = stop_sd
    ext_emd.range_thr = envelope_mean_tol
    decomposer = pyemd.CEEMDAN(
        trials=ensemble_size,
        epsilon=noise_width,
        ext_EMD=ext_emd,
        parallel=False,
        seed=random_state,
    )
    components = np.asarray(decomposer.ceemdan(x, max_imf=_max_imfs_for_external(max_imfs)), dtype=float)
    if components.ndim == 1:
        components = components[np.newaxis, :]
    if components.shape[0] <= 1:
        return np.empty((0, x.size)), x.copy()
    imfs = components[:-1]
    residue = components[-1]
    if max_imfs is not None:
        imfs = imfs[:max_imfs]
        residue = x - imfs.sum(axis=0)
    return imfs, residue


def iceemdan(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | np.random.Generator | None = None,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
    envelope_mean_tol: float = 0.1,
    snr_flag: int = 1,
    emd_backend: EMDBackend = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Improved CEEMDAN-style decomposition.

    This follows the Colominas-style MATLAB code structure:
    pre-decompose each white-noise realization, estimate the first local mean
    from noisy copies of the normalized signal, then obtain every next mode as
    the difference between consecutive averaged local means.
    """

    x = _as_1d(signal)
    if snr_flag not in {1, 2}:
        raise ValueError("snr_flag must be 1 or 2")
    x_std = np.std(x)
    if x_std == 0:
        return np.empty((0, x.size)), x.copy()
    x_norm = x / x_std
    rng = np.random.default_rng(random_state)
    noise_modes: list[np.ndarray] = []
    noise_max_imfs = None if max_imfs is None else max_imfs + 1
    for _ in range(ensemble_size):
        noise = rng.normal(size=x.size)
        modes, residue = emd(
            noise,
            max_imfs=noise_max_imfs,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            envelope_mean_tol=envelope_mean_tol,
            backend=emd_backend,
        )
        noise_modes.append(np.vstack((modes, residue[np.newaxis, :])))

    first_means = []
    for modes in noise_modes:
        noise = _normalize_noise(modes[0])
        noisy_signal = x_norm + noise_width * noise
        first, noisy_residue = emd(
            noisy_signal,
            max_imfs=1,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            envelope_mean_tol=envelope_mean_tol,
            backend=emd_backend,
        )
        first_means.append(noisy_residue if first.size else noisy_signal)

    current_mean = np.mean(first_means, axis=0)
    imfs: list[np.ndarray] = [x_norm - current_mean]
    mode_index = 1

    while _can_extract_external_imf(
        current_mean,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        envelope_mean_tol=envelope_mean_tol,
        emd_backend=emd_backend,
    ):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break

        next_means = []
        for modes in noise_modes:
            if modes.shape[0] > mode_index:
                noise = modes[mode_index]
                if snr_flag == 2:
                    noise = _normalize_noise(noise)
                noisy_mean = current_mean + np.std(current_mean) * noise_width * noise
            else:
                noisy_mean = current_mean

            _, noisy_residue = emd(
                noisy_mean,
                max_imfs=1,
                max_siftings=max_siftings,
                stop_sd=stop_sd,
                envelope_mean_tol=envelope_mean_tol,
                backend=emd_backend,
            )
            next_means.append(noisy_residue)

        next_mean = np.mean(next_means, axis=0)
        imf = current_mean - next_mean
        if np.allclose(imf, 0):
            break
        imfs.append(imf)
        current_mean = next_mean
        mode_index += 1

    modes = np.vstack(imfs) * x_std
    residue = current_mean * x_std
    return modes, residue
