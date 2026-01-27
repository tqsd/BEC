from __future__ import annotations
from typing import Optional, Tuple, Literal
from enum import IntEnum
import numpy as np
import jax.numpy as jnp


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


def outer_operator(
    ket: np.ndarray, bra: Optional[np.ndarray] = None
) -> np.ndarray:
    ket = np.asarray(ket, dtype=complex).reshape(-1)
    if bra is None:
        bra = ket
    else:
        bra = np.asarray(bra, dtype=complex).reshape(-1)
    if ket.shape[0] != 4 or bra.shape[0] != 4:
        raise ValueError(
            "outer_operator expects 4D kets/bras for the QD space")

    return np.outer(ket, np.conjugate(bra))


def exciton_eigenkets_4d(Hx_2x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize the exciton block Hamiltonian in basis (X1, X2) and return
    4D kets (in basis G, X1, X2, XX).

    Returns (ket_low, ket_high) corresponding to eigenvalues (low, high).
    """
    Hx = np.asarray(Hx_2x2, dtype=complex)
    if Hx.shape != (2, 2):
        raise ValueError("Hx_2x2 must be 2x2")

    # Hermitian diagonalization
    evals, evecs = np.linalg.eigh(Hx)  # columns are eigenvectors
    v_low = evecs[:, 0]
    v_high = evecs[:, 1]

    ket_low = np.array([0.0, v_low[0], v_low[1], 0.0], dtype=complex)
    ket_high = np.array([0.0, v_high[0], v_high[1], 0.0], dtype=complex)

    # Normalize (should already be, but keep it safe)
    ket_low /= np.linalg.norm(ket_low)
    ket_high /= np.linalg.norm(ket_high)

    return ket_low, ket_high


def exciton_relaxation_operator(
    Hx_2x2: np.ndarray,
    direction: Literal["down", "up"] = "down",
) -> np.ndarray:
    """
    Build |Xm><Xp| ("down": high->low) or |Xp><Xm| ("up": low->high)
    in the full 4D QD space.

    Conventions:
      - "down": |low><high|
      - "up"  : |high><low|
    """
    ket_low, ket_high = exciton_eigenkets_4d(Hx_2x2)

    if direction == "down":
        return outer_operator(ket_low, ket_high)  # |low><high|
    elif direction == "up":
        return outer_operator(ket_high, ket_low)  # |high><low|
    else:
        raise ValueError("direction must be 'down' or 'up'")


def basis_ket_4d(state: QDState) -> jnp.ndarray:
    v = jnp.zeros((4,), dtype=jnp.complex128)
    return v.at[int(state)].set(1.0)
