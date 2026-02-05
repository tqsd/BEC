from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bec.quantum_dot.enums import QDState


@dataclass(frozen=True)
class FrameConstraint:
    """
    eps_dst(t) - eps_src(t) = rhs(t)

    Here rhs(t) is in rad/s (physical units).
    """

    src: QDState
    dst: QDState
    rhs_rad_s: np.ndarray
    meta: Mapping[str, Any]


def solve_state_energies_ls(
    *,
    states: Sequence[QDState],
    constraints: Sequence[FrameConstraint],
    gauge_state: QDState,
) -> dict[QDState, np.ndarray]:
    """
    Solve eps_s(t) in rad/s for each state s, with gauge eps[gauge_state]=0.

    Uses per-time-slice least squares via pseudoinverse.
    Works for trees and overconstrained loops.

    Returns:
        dict: QDState -> eps_rad_s(t) array shape (T,)
    """
    states_t = tuple(states)
    if gauge_state not in states_t:
        raise ValueError("gauge_state must be in states")

    if not constraints:
        return {st: np.zeros(0, dtype=float) for st in states_t}

    # Validate lengths
    T = int(np.asarray(constraints[0].rhs_rad_s, dtype=float).reshape(-1).size)
    for c in constraints:
        v = np.asarray(c.rhs_rad_s, dtype=float).reshape(-1)
        if v.size != T:
            raise ValueError("Constraint rhs length mismatch")

    unknown_states = [st for st in states_t if st is not gauge_state]
    n = len(unknown_states)
    idx = {st: i for i, st in enumerate(unknown_states)}

    m = len(constraints)
    A = np.zeros((m, n), dtype=float)
    b = np.zeros((m, T), dtype=float)

    for row, c in enumerate(constraints):
        if c.dst is not gauge_state:
            A[row, idx[c.dst]] += 1.0
        if c.src is not gauge_state:
            A[row, idx[c.src]] -= 1.0
        b[row, :] = np.asarray(c.rhs_rad_s, dtype=float).reshape(-1)

    pinvA = np.linalg.pinv(A)
    x = pinvA @ b  # (n, T)

    out: dict[QDState, np.ndarray] = {gauge_state: np.zeros(T, dtype=float)}
    for st in unknown_states:
        out[st] = x[idx[st], :].copy()
    return out


def eps_to_solver_coeff(
    eps_rad_s: np.ndarray, time_unit_s: float
) -> np.ndarray:
    """
    Convert eps(t) from rad/s -> solver units coefficient array (complex).
    """
    s = float(time_unit_s)
    if s <= 0.0:
        raise ValueError("time_unit_s must be > 0")
    return (np.asarray(eps_rad_s, dtype=float).reshape(-1) * s).astype(complex)
