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

    def gen_channels(self, is_los=False) -> None:
        """
        Generate channel coefficients h. Note that the shape of h is N for NLOS, N + 1 for LOS.

        Args:
            is_los (bool): Whether there is line-of-sight direct path (True) or not (False)
        """
        tx = self.tx
        ris = self.ris
        rx = self.rx
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
            self.h_0 = 10**(-PL_0/20) * zeta_0
        else:
            self.h_0 = 0

        # NLOS components
        zeta_1 = np.random.normal(0, 1/np.sqrt(2), size=(N, 1)) + 1j * np.random.normal(0, 1/np.sqrt(2), size=(N, 1))
        zeta_2 = np.random.normal(0, 1/np.sqrt(2), size=(N, 1)) + 1j * np.random.normal(0, 1/np.sqrt(2), size=(N, 1))
        self.h = 10**(-(PL_1 + PL_2)/20) * zeta_1 * zeta_2

        # forming M_true, w_true and c
        v_0 = np.array([[np.real(self.h_0)], [np.imag(self.h_0)]])
        V = np.concatenate([np.real(self.h), np.imag(self.h)], axis=-1).T
        self.M_true = tx.power/noise_power * V.T @ V
        self.c_true = tx.power/noise_power * np.linalg.norm(v_0)**2
        self.w_true = tx.power/noise_power * 2 * V.T @ v_0

    def cal_snr(self) -> float:
        """
        Calculate the received SNR.
        """
        snr = self.ris.vector_x.T @ self.M_true @ self.ris.vector_x + self.w_true.T @ self.ris.vector_x + self.c_true
        return snr.item()
    
    def run_GCSM(self, N_samples=None) -> None:
        """
        Perform GCSM (grouped conditional sample mean) algorithm: https://arxiv.org/abs/2305.18998
        GCSM is not implemented in other modules due to the protocols including at least 3 stages.
        
        Args:
            N_samples (_type_, optional): Number of samples in total. Defaults to None.
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
            y[i] = self.cal_snr() - self.c_true

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
            y[i] = self.cal_snr() - self.c_true

        unknown_elements= opt.CSM(X, y)

        final_state = np.concatenate([fixed_elements, unknown_elements])

        self.ris.vector_x = final_state
        self.ris.theta = np.arccos(final_state)
