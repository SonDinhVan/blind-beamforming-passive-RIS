"""
This file contains the constants and parameters of the system.
"""
# Constants
# Noise power
NOISE_POWER_dbm = -90
NOISE_POWER = 10 ** (NOISE_POWER_dbm/10) * 10**-3

# Transmit power
P_dbm = 30
P = 10**(P_dbm/10) * 10**-3
