

class ICEEMDAN:

    """Class-based ICEEMDAN decomposer with a CEEMDAN-like interface.



    The callable object returns a complete component matrix: all extracted IMFs

    followed by the final residue. Use :meth:`get_imfs_and_residue` after a run

    when the split representation is more convenient.

    """



    trials: int = 100

    epsilon: float = 0.2

    max_imf: int = 100

    snr_flag: int = 1

    seed: int | None = None

    max_siftings: int = 50

    stop_sd: float = 0.2

    envelope_mean_tol: float = 0.1



    def __post_init__(self) -> None:

        if self.trials < 1:

            raise ValueError("trials must be at least 1")

        if self.epsilon < 0:

            raise ValueError("epsilon must be non-negative")

        if self.max_imf == 0 or self.max_imf < -1:

            raise ValueError("max_imf must be -1 or a positive integer")

        if self.snr_flag not in {1, 2}:

            raise ValueError("snr_flag must be 1 or 2")

        self.C_IMF: np.ndarray | None = None

        self.residue: np.ndarray | None = None



    def __call__(self, signal: np.ndarray, max_imf: int | None = None) -> np.ndarray:

        return self.iceemdan(signal, max_imf=max_imf)



    def noise_seed(self, seed: int | None) -> None:

        """Set the random seed used for the ensemble noise."""



        self.seed = seed



    def iceemdan(self, signal: np.ndarray, max_imf: int | None = None) -> np.ndarray:

        """Perform ICEEMDAN and return IMFs with the residue as the last row."""



        if max_imf is None:

            max_imf = None if self.max_imf == -1 else self.max_imf



        imfs, residue = iceemdan(

            signal,

            max_imfs=max_imf,

            ensemble_size=self.trials,

            noise_width=self.epsilon,

            random_state=self.seed,

            max_siftings=self.max_siftings,

            stop_sd=self.stop_sd,

            envelope_mean_tol=self.envelope_mean_tol,

            snr_flag=self.snr_flag,

        )



        if imfs.size:

            components = np.vstack((imfs, residue[np.newaxis, :]))

        else:

            components = residue[np.newaxis, :]



        self.C_IMF = components

        self.residue = residue

        return components



    def get_imfs_and_residue(self) -> tuple[np.ndarray, np.ndarray]:

        """Return the IMFs and final residue from the most recent run."""



        if self.C_IMF is None or self.residue is None:

            raise ValueError("No IMF found. Please run iceemdan first.")

        if self.C_IMF.shape[0] == 1:

            return np.empty((0, self.C_IMF.shape[1])), self.residue

        return self.C_IMF[:-1], self.residue


