from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

from bec.quantum_dot.mode_registry import ModeRegistry
from bec.quantum_dot.enums import QDState  # canonical enum

# Your robust transition_operator from earlier patch
from bec.quantum_dot.operators.qd import transition_operator  # type: ignore


def _eye(d: int) -> np.ndarray:
    return np.eye(int(d), dtype=np.complex128)


def _destroy(d: int) -> np.ndarray:
    d = int(d)
    a = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def _create(d: int) -> np.ndarray:
    return _destroy(d).conj().T


def _vacuum_projector(d: int) -> np.ndarray:
    d = int(d)
    P0 = np.zeros((d, d), dtype=np.complex128)
    P0[0, 0] = 1.0
    return P0


def _embed_single_mode_op(
    dims: list[int], mode_index: int, op: np.ndarray
) -> np.ndarray:
    """Embed an op acting on fock factor `mode_index` (0-based among channels) into full fock space."""
    n_modes = len(dims) - 1
    mats = []
    for k in range(n_modes):
        d = int(dims[1 + k])
        mats.append(op if k == mode_index else np.eye(d, dtype=np.complex128))
    out = mats[0]
    for M in mats[1:]:
        out = np.kron(out, M)
    return out


def build_symbol_table(
    *,
    modes: ModeRegistry,
    theta: float = 0.0,
    phi: float = 0.0,
) -> Dict[str, Callable[[list[int]], np.ndarray]]:
    """
    Symbol table for PhotonWeave (and other engines).

    IMPORTANT: This matches your current ModeRegistry:
      - each LightChannel is one mode already (pol is in ChannelKey)
      - dims layout is [4, d0, d1, d2, ...]

    Symbols:
      QD:  s_<TO>_<FROM>, idq
      Mode i: a{i}, a{i}_dag, n{i}, if{i}, vac{i}
    """
    ctx: Dict[str, Callable[[list[int]], np.ndarray]] = {}

    # --- QD ops (fixed 4x4) ---
    states = [QDState.G, QDState.X1, QDState.X2, QDState.XX]
    for to in states:
        for fr in states:
            key = f"s_{to.name}_{fr.name}"
            mat = np.asarray(transition_operator(to, fr), dtype=np.complex128)
            ctx[key] = lambda _dims, A=mat: A

    ctx["idq"] = lambda _dims: np.eye(4, dtype=np.complex128)

    # --- Per-mode ops ---
    # mode i lives at dims index 1+i
    n_modes = len(modes.channels)

    for i in range(n_modes):

        def a_i(dims: list[int], _i=i) -> np.ndarray:
            d = int(dims[1 + _i])
            return _destroy(d)

        def adag_i(dims: list[int], _i=i) -> np.ndarray:
            d = int(dims[1 + _i])
            return _create(d)

        def n_i(dims: list[int], _i=i) -> np.ndarray:
            d = int(dims[1 + _i])
            return _create(d) @ _destroy(d)

        def if_i(dims: list[int], _i=i) -> np.ndarray:
            d = int(dims[1 + _i])
            return _eye(d)

        def vac_i(dims: list[int], _i=i) -> np.ndarray:
            d = int(dims[1 + _i])
            return _vacuum_projector(d)

        ctx[f"a{i}"] = a_i
        ctx[f"a{i}_dag"] = adag_i
        ctx[f"n{i}"] = n_i
        ctx[f"if{i}"] = if_i
        ctx[f"vac{i}"] = vac_i

    # assumes two channels exist: iH=0, iV=1 for this test setup
    def a_plus(
        dims: list[int],
        iH: int = 0,
        iV: int = 1,
        th: float = theta,
        ph: float = phi,
    ) -> np.ndarray:
        dH = int(dims[1 + iH])
        dV = int(dims[1 + iV])
        if dH != dV:
            raise ValueError("Rotation test assumes equal dims for H and V")
        a = _destroy(dH)
        c, s = np.cos(th), np.sin(th)
        eip = np.exp(1j * ph)

        AH = _embed_single_mode_op(dims, iH, a)
        AV = _embed_single_mode_op(dims, iV, a)
        return c * AH + eip * s * AV

    ctx["a_plus"] = a_plus
    ctx["a_plus_dag"] = lambda dims: a_plus(dims).conj().T

    return ctx
