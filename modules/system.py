"""
This class provides tools for setting up, analyzing and performing the general
algorithms of the system.
"""
from configs import config
from modules import optimization as opt

import numpy as np

noise_power = config.NOISE_POWER
class System:
    """
    System class to manage the wireless communication system with TX, RX, RIS 
    and channel modeling.
    """

    def __init__(self, tx, ris, rx) -> None:
        """
        Initialize a system with transmitter, RIS, and receiver.

        Args:
            tx (TX): Transmitter object
            ris (RIS): Reconfigurable Intelligent Surface object
            rx (RX): Receiver object
        """
        self.tx = tx
        self.ris = ris
        self.rx = rx

        # Channel will be initialized when gen_channels is called
        self.h_0 = None
        self.h = None
        # f(x) = x.T M_true x + w_true.T x + c
        self.M_true = None
        self.w_true = None
        self.c_true = None

    def cal_distance(self, pos1, pos2) -> float:
        """
        Calculate Euclidean distance between two positions
        """
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

    def _build_user_terms(self, tx, ris, rx, is_los=False):
        """
        Build (M_true, w_true, c_true) terms for one TX-RIS-RX triplet.

        Args:
            tx (TX): Transmitter object
            ris (RIS): Reconfigurable Intelligent Surface object
            rx (RX): Receiver object
            is_los (bool): Whether LOS direct path is enabled
        """
        N = ris.N

        # Calculate distances
        d_tx_rx = self.cal_distance(tx, rx)
        d_tx_ris = self.cal_distance(tx, ris)
        d_ris_rx = self.cal_distance(ris, rx)

        # pathloss
        PL_1 = 30 + 22 * np.log10(d_tx_ris)
        PL_2 = 30 + 22 * np.log10(d_ris_rx)

        # LOS component
        if is_los:
            # Calculate pathloss for direct path
            PL_0 = 32.6 + 36.7 * np.log10(d_tx_rx)
            # Generate LOS component
            zeta_0 = np.random.normal(0, 1/np.sqrt(2)) + 1j * np.random.normal(0, 1/np.sqrt(2))
            h_0 = 10**(-PL_0/20) * zeta_0
        else:
            h_0 = 0

        # NLOS components
        zeta_1 = np.random.normal(0, 1/np.sqrt(2), size=(N, 1)) + 1j * np.random.normal(0, 1/np.sqrt(2), size=(N, 1))
        zeta_2 = np.random.normal(0, 1/np.sqrt(2), size=(N, 1)) + 1j * np.random.normal(0, 1/np.sqrt(2), size=(N, 1))
        h = 10**(-(PL_1 + PL_2)/20) * zeta_1 * zeta_2

        # forming M_true, w_true and c
        v_0 = np.array([[np.real(h_0)], [np.imag(h_0)]])
        V = np.concatenate([np.real(h), np.imag(h)], axis=-1).T
        M_true = tx.power/noise_power * V.T @ V
        c_true = tx.power/noise_power * np.linalg.norm(v_0)**2
        w_true = tx.power/noise_power * 2 * V.T @ v_0

        return M_true, w_true, c_true, h_0, h

    def gen_channels(self, is_los=False) -> None:
        """
        Generate channel coefficients h. Note that the shape of h is N for NLOS, N + 1 for LOS.

        Args:
            is_los (bool): Whether there is line-of-sight direct path (True) or not (False)
        """
        self.M_true, self.w_true, self.c_true, h_0, h = self._build_user_terms(
            self.tx,
            self.ris,
            self.rx,
            is_los=is_los
        )
        self.h_0 = h_0
        self.h = h

    def cal_snr(self, sigma_db: float = 0.0, delta_db: float = None) -> float:
        """
        Calculate the received SNR.

        Args:
            sigma_db: Standard deviation of additive Gaussian noise in dB.
            delta_db: Quantization resolution in dB. If set, the noisy dB SNR is
                quantized to this step; if None, no quantization is applied.
        """
        snr = self.ris.vector_x.T @ self.M_true @ self.ris.vector_x + self.w_true.T @ self.ris.vector_x + self.c_true
        snr = float(snr.item())

        # Ideal measurement (default behavior): no added noise and no quantization.
        if sigma_db == 0 and delta_db is None:
            return snr

        # Guard against numerical issues if the ideal value is extremely small.
        snr_db = 10 * np.log10(max(snr, 1e-30))
        snr_db_noisy = snr_db + np.random.normal(0, sigma_db)

        if delta_db is None:
            return 10 ** (snr_db_noisy / 10)

        snr_db_quant = delta_db * np.round(snr_db_noisy / delta_db)
        return 10 ** (snr_db_quant / 10)
    
    def run_GCSM(
        self,
        N_samples=None,
        sigma_db: float = 0.0,
        delta_db: float = None
    ) -> None:
        """
        Perform GCSM (grouped conditional sample mean) algorithm: https://arxiv.org/abs/2305.18998
        GCSM is not implemented in other modules due to the protocols including at least 3 stages.
        
        Args:
            N_samples (_type_, optional): Number of samples in total. Defaults to None.
            sigma_db: Standard deviation of additive Gaussian noise in dB used for measurements.
            delta_db: Quantization step in dB for measurements. If None, no quantization.
        """
        T1 = int(N_samples/2) # only use a half of number of samples for the 1st phase
        num_phase = int(self.ris.N/2) # number of phase needs to find
        X = np.zeros(shape=(num_phase, T1))
        y = np.zeros(shape=(T1, 1))

        for i in range(T1):
            # generate a random ris setting for a half of elements
            r = np.random.choice([-1, 1], size=(num_phase, 1))
            self.ris.vector_x = np.ones(shape=(self.ris.N, 1))
            self.ris.vector_x[:num_phase] = r # only randomize a half of it
            X[:, i] = r.flatten()
            y[i] = self.cal_snr(sigma_db=sigma_db, delta_db=delta_db) - self.c_true

        first_half = opt.CSM(X, y)
        
        fix_num = int(self.ris.N/4) # number of element with known value (half of first_half)
        unknown_num = self.ris.N - fix_num # number of element with unknown value (the rest)
        
        fixed_elements = first_half[:fix_num] # fixing a half of the first half
        # group 2:
        X = np.zeros(shape=(unknown_num, N_samples))
        y = np.zeros(shape=(N_samples, 1))

        for i in range(T1):
            # generate a random ris setting, but we fix a half of the first half
            self.ris.vector_x = np.random.choice([-1, 1], self.ris.N)
            self.ris.vector_x[:fix_num] = fixed_elements.reshape(1, -1)
            X[:, i] = self.ris.vector_x[-unknown_num:]
            y[i] = self.cal_snr(sigma_db=sigma_db, delta_db=delta_db) - self.c_true

        unknown_elements= opt.CSM(X, y)

        final_state = np.concatenate([fixed_elements, unknown_elements])

        self.ris.vector_x = final_state
        self.ris.theta = np.arccos(final_state)


class MultiUserSystem(System):
    """
    Multi-user extension that reuses single-user System behavior and solves a
    summed objective over multiple users.
    """

    def __init__(self, tx, ris, rxs, user_weights=None, normalize_weights=False) -> None:
        """
        Initialize a system with one TX, one RIS and multiple RX users.

        Args:
            tx (TX): Transmitter object.
            ris (RIS): Reconfigurable Intelligent Surface object.
            rxs (list[RX] | tuple[RX]): List of receiver users.
            user_weights (array-like | None): Optional positive user weights.
                If None, all users are weighted equally.
            normalize_weights (bool): If True, normalize user weights to unit sum.
        """
        rxs = list(rxs)
        if len(rxs) == 0:
            raise ValueError("MultiUserSystem needs at least one RX user.")

        self.rxs = rxs
        self.num_users = len(rxs)
        self.normalize_weights = normalize_weights
        self.user_weights = self._normalize_user_weights(self.num_users, user_weights, normalize_weights)

        # Use the first RX for base class initialization state compatibility.
        super().__init__(tx, ris, rxs[0])

        self.M_true_users = None
        self.w_true_users = None
        self.c_true_users = None
        self.h0_users = None
        self.h_users = None

    @staticmethod
    def _normalize_user_weights(num_users, weights=None, normalize=False):
        """
        Build and validate user weights.
        """
        if weights is None:
            weights = np.ones(num_users, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if weights.size != num_users:
                raise ValueError(
                    f"user_weights has shape {weights.shape}, expected ({num_users},)"
                )
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                raise ValueError("user_weights must be finite numbers.")
            if np.any(weights < 0):
                raise ValueError("user_weights must be non-negative.")

        if normalize:
            s = float(np.sum(weights))
            if s == 0:
                raise ValueError("Cannot normalize zero user weights.")
            weights = weights / s

        return weights

    def gen_channels(self, is_los=False, user_weights=None, normalize_weights=None) -> None:
        """
        Generate channel coefficients for all users and aggregate them.

        The aggregated objective is:
            sum_k (x^T M_k x + w_k^T x + c_k)
            with optional user weights alpha_k:
            x^T (sum alpha_k M_k) x + (sum alpha_k w_k)^T x + sum alpha_k c_k.
        """
        if normalize_weights is None:
            normalize_weights = self.normalize_weights
        if user_weights is None:
            user_weights = self.user_weights

        user_weights = self._normalize_user_weights(self.num_users, user_weights, normalize_weights)
        self.user_weights = user_weights

        M_user = []
        w_user = []
        c_user = []
        h0_user = []
        h_user = []

        for rx in self.rxs:
            M_i, w_i, c_i, h0_i, h_i = self._build_user_terms(self.tx, self.ris, rx, is_los=is_los)
            M_user.append(M_i)
            w_user.append(w_i)
            c_user.append(c_i)
            h0_user.append(h0_i)
            h_user.append(h_i)

        self.M_true_users = M_user
        self.w_true_users = w_user
        self.c_true_users = c_user
        self.h0_users = h0_user
        self.h_users = h_user

        # For compatibility with existing single-user methods.
        self.M_true = np.sum([w * M_i for w, M_i in zip(user_weights, M_user)], axis=0)
        self.w_true = np.sum([w * w_i for w, w_i in zip(user_weights, w_user)], axis=0)
        self.c_true = float(np.sum([w * c_i for w, c_i in zip(user_weights, c_user)]))

        # Keep last generated user objects for debug/compatibility.
        self.h_0 = h0_user[-1]
        self.h = h_user[-1]

    def cal_multi_snr(self, x) -> float:
        """
        Compute weighted sum SNR over all users at RIS state x.
        """
        x_col = np.asarray(x).reshape(-1, 1)
        total_snr = 0.0
        for alpha_k, M_i, w_i, c_i in zip(self.user_weights, self.M_true_users, self.w_true_users, self.c_true_users):
            total_snr += float(alpha_k * (x_col.T @ M_i @ x_col + w_i.T @ x_col + c_i))
        return total_snr

    def benchmark_sum_snr(
        self, 
        num_trials=200,
        rank=2,
        is_los=False,
        user_weights=None,
        normalize_weights=None,
        init_from_rank2=False,
        random_init=False,
        track_min_snr=False,
        plot=True
    ):
        """
        Benchmark multi-user sum-SNR distribution over random channel realizations.
        For rank > 2, init_from_rank2 controls alternating solver initialization.
        Set random_init=True to force random x initialization in the alternating solver
        for r > 2.
        If track_min_snr=True, also returns per-trial minimum user SNR.

        Returns:
            sum_snr_samples (np.ndarray): Array of achieved sum SNR per trial.
            x_solutions (list): RIS solutions for each trial.
            min_snr_samples (np.ndarray): Array of minimum per-trial user SNR.
                Returned only when track_min_snr=True.
        """
        sum_snr_samples = []
        x_solutions = []
        min_snr_samples = []

        for t in range(num_trials):
            self.gen_channels(
                is_los=is_los,
                user_weights=user_weights,
                normalize_weights=normalize_weights
            )
            x_star = opt.solve_our_optimization(
                self.M_true,
                self.w_true,
                r=rank,
                init_from_rank2=init_from_rank2 if rank > 2 else False,
                random_init=random_init and (rank > 2),
            )
            self.ris.vector_x = x_star
            self.ris.theta = np.arccos(x_star)

            sum_snr_samples.append(self.cal_multi_snr(x_star))
            x_solutions.append(x_star)
            if track_min_snr:
                x_col = np.asarray(x_star).reshape(-1, 1)
                user_snrs = [
                    float(x_col.T @ M_i @ x_col + w_i.T @ x_col + c_i)
                    for M_i, w_i, c_i in zip(
                        self.M_true_users, self.w_true_users, self.c_true_users
                    )
                ]
                min_snr_samples.append(float(np.min(user_snrs)))

        sum_snr_samples = np.array(sum_snr_samples)
        if track_min_snr:
            min_snr_samples = np.array(min_snr_samples)

        if plot:
            import matplotlib.pyplot as plt
            sorted_vals = np.sort(sum_snr_samples)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            plt.figure()
            plt.plot(sorted_vals, cdf, label=f"{self.num_users} users")
            plt.xlabel("Sum SNR")
            plt.ylabel("CDF")
            plt.title("Multi-user sum-SNR CDF")
            plt.grid(True)
            plt.legend()
            plt.show()

        if track_min_snr:
            return sum_snr_samples, x_solutions, min_snr_samples
        return sum_snr_samples, x_solutions
