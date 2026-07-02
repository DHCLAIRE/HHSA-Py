"""EMD, CEEMDAN, and ICEEMDAN-style decomposition.

EMD and CEEMDAN are delegated to external libraries: EMD-Python (``emd``)
and PyEMD (``PyEMD``).
ICEEMDAN remains project code, but its internal EMD calls use those libraries.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import Literal

import numpy as np

EMDBackend = Literal["auto", "emd-python", "pyemd"]
EMDPythonSift = Literal["sift", "ensemble_sift", "complete_ensemble_sift", "mask_sift", "iterated_mask_sift"]
SiftAcceleration = Literal["none", "cpu", "gpu", "auto"]
DecompositionMethod = Literal[
    "emd",
    "sift",
    "ensemble_sift",
    "complete_ensemble_sift",
    "mask_sift",
    "iterated_mask_sift",
    "ceemdan",
    "iceemdan",
]

_SIFT_ACCELERATION_OPTIONS = {"none", "cpu", "gpu", "auto"}


def _validate_sift_acceleration(sift_acceleration: SiftAcceleration) -> SiftAcceleration:
    """Validate the acceleration selector used around computationally heavy sifting."""

    if sift_acceleration not in _SIFT_ACCELERATION_OPTIONS:
        raise ValueError("sift_acceleration must be 'none', 'cpu', 'gpu', or 'auto'")
    return sift_acceleration


def _worker_count(n_jobs: int | None) -> int:
    """Resolve sklearn-style worker counts without adding a runtime dependency."""

    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        return cpu_count
    if n_jobs == 0:
        raise ValueError("n_jobs must be None, -1, or a non-zero integer")
    if n_jobs < 0:
        return max(1, cpu_count + 1 + n_jobs)
    return max(1, int(n_jobs))


def _should_parallelize(sift_acceleration: SiftAcceleration, n_jobs: int | None) -> bool:
    """Return whether independent EMD calls should be farmed across CPU workers."""

    return sift_acceleration in {"cpu", "gpu", "auto"} and _worker_count(n_jobs) > 1


def _parallel_map(
    function,
    values: list[np.ndarray],
    *,
    sift_acceleration: SiftAcceleration,
    n_jobs: int | None,
) -> list[np.ndarray]:
    """Map independent sifts in order, using threads when acceleration is enabled."""

    if not _should_parallelize(sift_acceleration, n_jobs) or len(values) <= 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=_worker_count(n_jobs)) as executor:
        return list(executor.map(function, values))


def _cupy_module(sift_acceleration: SiftAcceleration):
    """Import CuPy only for explicit/automatic GPU acceleration."""

    if sift_acceleration not in {"gpu", "auto"}:
        return None
    try:
        cupy = import_module("cupy")
    except (ImportError, ModuleNotFoundError):
        if sift_acceleration == "gpu":
            raise ImportError(
                "sift_acceleration='gpu' requires CuPy. Install the CuPy package that matches your CUDA runtime."
            )
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception as exc:
        if sift_acceleration == "gpu":
            raise RuntimeError("sift_acceleration='gpu' requires a working CUDA device visible to CuPy") from exc
        return None
    return cupy


def _batched_noise(
    rng: np.random.Generator,
    ensemble_size: int,
    n_samples: int,
    xp,
) -> list[np.ndarray]:
    """Generate ensemble white noise on NumPy or CuPy and return CPU arrays for EMD libraries."""

    if xp is None:
        return [rng.normal(size=n_samples) for _ in range(ensemble_size)]

    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=ensemble_size, dtype=np.uint32)
    noises = []
    for seed in seeds:
        gpu_rng = xp.random.default_rng(int(seed))
        noises.append(xp.asnumpy(gpu_rng.normal(size=n_samples)))
    return noises


def _mean_stack(arrays: list[np.ndarray], xp) -> np.ndarray:
    """Average an ensemble on the GPU when available, otherwise use NumPy."""

    if xp is None:
        return np.mean(arrays, axis=0)
    return xp.asnumpy(xp.mean(xp.asarray(arrays), axis=0))


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


def _from_emd_python_modes(modes: np.ndarray, n_samples: int, max_imfs: int | None) -> np.ndarray:
    """Convert EMD-Python's samples x modes output to modes x samples."""

    if isinstance(modes, tuple):
        arr = np.asarray(modes[0], dtype=float)
    else:
        arr = np.asarray(modes, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.size == 0:
        return np.empty((0, n_samples))
    if arr.shape[0] == n_samples:
        arr = arr.T
    if arr.shape[1] != n_samples:
        raise ValueError("EMD-Python returned IMFs with an unexpected sample dimension")
    if max_imfs is not None:
        arr = arr[:max_imfs]
    return arr


def _emd_with_emd_python(
    signal: np.ndarray,
    *,
    max_imfs: int | None,
    max_siftings: int,
    stop_sd: float,
    sift_method: EMDPythonSift,
    ensemble_size: int,
    noise_width: float,
    random_state: int | None,
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
            "noise_seed": random_state,
            "imf_opts": imf_opts,
        }
    elif sift_method == "complete_ensemble_sift":
        kwargs = {
            "max_imfs": max_imfs,
            "nensembles": ensemble_size,
            "ensemble_noise": noise_width,
            "noise_seed": random_state,
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
    elif sift_method == "iterated_mask_sift":
        kwargs = {
            "max_imfs": max_imfs,
            "mask_0": mask_freqs,
            "mask_amp": mask_amp,
            "mask_amp_mode": mask_amp_mode,
            "imf_opts": imf_opts,
        }
    else:
        raise ValueError(
            "sift_method must be 'sift', 'ensemble_sift', 'complete_ensemble_sift', "
            "'mask_sift', or 'iterated_mask_sift'"
        )
    try:
        modes = sift_function(x, **kwargs)
    except TypeError:
        kwargs.pop("imf_opts", None)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        modes = sift_function(x, **kwargs)
    except UnboundLocalError:
        # Catch the emd-python bug when it fails to find any IMFs
        return np.empty((0, x.size)), x.copy()
    modes = _from_emd_python_modes(modes, x.size, max_imfs)
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
    random_state: int | None = None,
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
                    random_state=random_state,
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
    random_state: int | None = None,
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
        random_state=random_state,
    )


def complete_ensemble_sift(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    max_siftings: int = 50,
    stop_sd: float = 0.2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with EMD-Python's complete ensemble sift."""

    return emd(
        signal,
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        backend="emd-python",
        sift_method="complete_ensemble_sift",
        ensemble_size=ensemble_size,
        noise_width=noise_width,
        random_state=random_state,
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


def iterated_mask_sift(
    signal: np.ndarray,
    *,
    max_imfs: int | None = None,
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
    max_siftings: int = 50,
    stop_sd: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with EMD-Python's iterated mask sift."""

    return emd(
        signal,
        max_imfs=max_imfs,
        max_siftings=max_siftings,
        stop_sd=stop_sd,
        backend="emd-python",
        sift_method="iterated_mask_sift",
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
    sift_acceleration: SiftAcceleration = "none",
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with PyEMD's CEEMDAN implementation.

    ``sift_acceleration="cpu"`` enables PyEMD's parallel CEEMDAN trials.
    """

    x = _as_1d(signal)
    sift_acceleration = _validate_sift_acceleration(sift_acceleration)
    if sift_acceleration == "gpu":
        raise ValueError("PyEMD CEEMDAN does not expose GPU sifting; use sift_acceleration='cpu' or ICEEMDAN")
    pyemd = import_module("PyEMD")
    ext_emd = pyemd.EMD()
    ext_emd.MAX_ITERATION = max_siftings
    ext_emd.std_thr = stop_sd
    ext_emd.range_thr = envelope_mean_tol
    parallel = sift_acceleration in {"cpu", "auto"}
    ceemdan_kwargs = {
        "trials": ensemble_size,
        "epsilon": noise_width,
        "ext_EMD": ext_emd,
        "parallel": parallel,
        "seed": random_state,
    }
    if parallel:
        ceemdan_kwargs["processes"] = _worker_count(n_jobs)
    decomposer = pyemd.CEEMDAN(**ceemdan_kwargs)
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
    sift_acceleration: SiftAcceleration = "none",
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved CEEMDAN-style decomposition.

    This follows the Colominas-style MATLAB code structure:
    pre-decompose each white-noise realization, estimate the first local mean
    from noisy copies of the normalized signal, then obtain every next mode as
    the difference between consecutive averaged local means.

    The sifting iterations themselves stay with EMD-Python/PyEMD. The
    acceleration modes target the independent ensemble EMD calls and reductions,
    which are the parallel part of noise-assisted EMD variants.
    """

    x = _as_1d(signal)
    sift_acceleration = _validate_sift_acceleration(sift_acceleration)
    xp = _cupy_module(sift_acceleration)
    if snr_flag not in {1, 2}:
        raise ValueError("snr_flag must be 1 or 2")
    x_std = np.std(x)
    if x_std == 0:
        return np.empty((0, x.size)), x.copy()
    x_norm = x / x_std
    rng = np.random.default_rng(random_state)
    noise_max_imfs = None if max_imfs is None else max_imfs + 1

    def decompose_noise(noise: np.ndarray) -> np.ndarray:
        modes, residue = emd(
            noise,
            max_imfs=noise_max_imfs,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            envelope_mean_tol=envelope_mean_tol,
            backend=emd_backend,
        )
        return np.vstack((modes, residue[np.newaxis, :]))

    noises = _batched_noise(rng, ensemble_size, x.size, xp)
    noise_modes = _parallel_map(
        decompose_noise,
        noises,
        sift_acceleration=sift_acceleration,
        n_jobs=n_jobs,
    )

    def first_mean(modes: np.ndarray) -> np.ndarray:
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
        return noisy_residue if first.size else noisy_signal

    first_means = _parallel_map(
        first_mean,
        noise_modes,
        sift_acceleration=sift_acceleration,
        n_jobs=n_jobs,
    )
    current_mean = _mean_stack(first_means, xp)
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

        def next_mean(modes: np.ndarray) -> np.ndarray:
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
            return noisy_residue

        next_means = _parallel_map(
            next_mean,
            noise_modes,
            sift_acceleration=sift_acceleration,
            n_jobs=n_jobs,
        )
        next_mean_array = _mean_stack(next_means, xp)
        imf = current_mean - next_mean_array
        if np.allclose(imf, 0):
            break
        imfs.append(imf)
        current_mean = next_mean_array
        mode_index += 1

    modes = np.vstack(imfs) * x_std
    residue = current_mean * x_std
    return modes, residue


def decompose_signal(
    signal: np.ndarray,
    method: DecompositionMethod,
    *,
    max_imfs: int | None = 10,
    ensemble_size: int = 64,
    noise_width: float = 0.2,
    random_state: int | None = 13,
    max_siftings: int = 20,
    stop_sd: float = 0.2,
    emd_backend: EMDBackend = "auto",
    mask_freqs: np.ndarray | float | None = None,
    mask_amp: float = 1.0,
    mask_amp_mode: str = "ratio_sig",
    sift_acceleration: SiftAcceleration = "none",
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a signal with one supported HHSA decomposition method."""

    sift_acceleration = _validate_sift_acceleration(sift_acceleration)

    if method in {"emd", "sift"}:
        return emd(
            signal,
            max_imfs=max_imfs,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            backend=emd_backend,
            sift_method="sift",
        )
    if method == "ensemble_sift":
        return ensemble_sift(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            random_state=random_state,
        )
    if method == "complete_ensemble_sift":
        return complete_ensemble_sift(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            random_state=random_state,
        )
    if method == "mask_sift":
        return mask_sift(
            signal,
            max_imfs=max_imfs,
            mask_freqs=mask_freqs,
            mask_amp=mask_amp,
            mask_amp_mode=mask_amp_mode,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    if method == "iterated_mask_sift":
        return iterated_mask_sift(
            signal,
            max_imfs=max_imfs,
            mask_freqs=mask_freqs,
            mask_amp=mask_amp,
            mask_amp_mode=mask_amp_mode,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
        )
    if method == "ceemdan":
        return ceemdan(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            emd_backend=emd_backend,
            sift_acceleration=sift_acceleration,
            n_jobs=n_jobs,
        )
    if method == "iceemdan":
        return iceemdan(
            signal,
            max_imfs=max_imfs,
            ensemble_size=ensemble_size,
            noise_width=noise_width,
            random_state=random_state,
            max_siftings=max_siftings,
            stop_sd=stop_sd,
            emd_backend=emd_backend,
            sift_acceleration=sift_acceleration,
            n_jobs=n_jobs,
        )
    raise ValueError(
        "method must be 'emd', 'sift', 'ensemble_sift', 'complete_ensemble_sift', "
        "'mask_sift', 'iterated_mask_sift', 'ceemdan', or 'iceemdan'"
    )
