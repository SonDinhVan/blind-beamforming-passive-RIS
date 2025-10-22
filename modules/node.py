"""
This class provides the basic set-up for TX, RX and RIS.
"""
import numpy as np
from configs import config


class TX:
    def __init__(self, x, y, z, power=None) -> None:
        """
        Initialize a transmitter.

        Args:
           x (float): x-coordinate position
           y (float): y-coordinate position
           z (float): z-coordinate position
           power (float, optional): Transmit power in dBm. If None, uses config.P
       """
        self.x = x
        self.y = y
        self.z = z
        self.power = power if power is not None else config.P

    def __str__(self) -> None:
        return f"Transmitter at position ({self.x}, {self.y}) with power {self.power} dBm"


class RX:
    def __init__(self, x, y, z) -> None:
        """
         Initialize a receiver.

        Args:
            x (float): x-coordinate position
            y (float): y-coordinate position
            z (float): z-coordinate position
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> None:
        return f"Receiver at position ({self.x}, {self.y})"


class RIS:
    def __init__(self, x, y, z, N=None) -> None:
        """
        Initialize a Reconfigurable Intelligent Surface.

        Args:
            x (float): x-coordinate position
            y (float): y-coordinate position
            z (float): z-coordinate position
            N (int): Number of reflective elements
        """
        self.x = x
        self.y = y
        self.z = z
        self.N = N
        # Initialize phase shifts to zero
        self.theta = np.zeros(shape=(self.N, 1))
        # This is the feature vector x in the paper. This name is used to differentiate
        # with x as x-coordinate
        self.vector_x = None

    def set_random_binary_phase_shifts(self) -> None:
        """
        Set random binary phase shifts for all reflective elements.
        """
        self.theta = np.random.choice([0, np.pi], size=(self.N, 1))
        self.vector_x = np.real(np.exp(self.theta * 1j))

    def __str__(self) -> None:
        return f"RIS at position ({self.x}, {self.y}) with {self.N} reflective elements"
