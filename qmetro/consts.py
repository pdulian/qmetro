import numpy as np
from numpy import kron




PAULI_X = np.array((
    (0, 1),
    (1, 0)
), dtype=np.complex128)
PAULI_Y = np.array((
    (0, -1j),
    (1j, 0)
), dtype=np.complex128)
PAULI_Z = np.array((
    (1, 0),
    (0, -1)
), dtype=np.complex128)
PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)


# Generators of counterclockwise rotations in mathematical notation
# (without i in the exponens).
LX = np.array((
    (0, 0, 0),
    (0, 0, -1),
    (0, 1, 0)
), dtype=np.float64)
LY = np.array((
    (0, 0, 1),
    (0, 0, 0),
    (-1, 0, 0)
), dtype=np.float64)
LZ = np.array((
    (0, -1, 0),
    (1, 0, 0),
    (0, 0, 0)
), dtype=np.float64)
LS = (LX, LY, LZ)


class QubitStates():
    ZERO = np.array([1, 0])
    ONE = np.array([0, 1])
    PLUS = np.array((1, 1)) / np.sqrt(2)
    MINUS = np.array((1, -1)) / np.sqrt(2)
    PLUS_I = np.array((1, 1j)) / np.sqrt(2)
    MINUS_I = np.array((1, -1j)) / np.sqrt(2)


class TwoQubitStates():
    ZERO = np.array([1, 0, 0, 0])
    ONE = np.array([0, 1, 0, 0])
    TWO = np.array([0, 0, 1, 0])
    THREE = np.array([0, 0, 0, 1])

    # Bell states
    PHI_PLUS = (
        kron(QubitStates.ZERO, QubitStates.ZERO)
        + kron(QubitStates.ONE, QubitStates.ONE)
    ) / np.sqrt(2)
    PHI_MINUS = (
        kron(QubitStates.ZERO, QubitStates.ZERO)
        - kron(QubitStates.ONE, QubitStates.ONE)
    ) / np.sqrt(2)
    PSI_PLUS = (
        kron(QubitStates.ZERO, QubitStates.ONE)
        + kron(QubitStates.ONE, QubitStates.ZERO)
    ) / np.sqrt(2)
    PSI_MINUS = (
        kron(QubitStates.ZERO, QubitStates.ONE)
        - kron(QubitStates.ONE, QubitStates.ZERO)
    ) / np.sqrt(2)


LINE_WIDTH = 120
