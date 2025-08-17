from enum import IntEnum
import numpy as np


class QDState(IntEnum):
    """Quantum Dot eigenstates."""

    G = 0  # Ground state
    X1 = 1  # Exciton 1
    X2 = 2  # Exciton 2
    XX = 3  # Biexciton


def transition_operator(from_state: QDState, to_state: QDState) -> np.ndarray:
    """
    Construct a transition operator |to⟩⟨from| for QD states.

    Parameters
    ----------
    from_state : QDState
        Initial state of the QD (⟨from|).
    to_state : QDState
        Final state of the QD (|to⟩).

    Returns
    -------
    np.ndarray
        A 4x4 complex NumPy array representing the transition operator.
    """
    dim = len(QDState)
    op = np.zeros((dim, dim), dtype=complex)
    op[to_state.value, from_state.value] = 1.0
    return op
