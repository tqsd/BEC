from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.ops.symbols import QDSymbol

LocalDims = tuple[int, ...]
SymbolKey = tuple[LocalDims, str]
Builder = Callable[[LocalDims], np.ndarray]


def canon_symbol(x: Any) -> str:
    """
    Canonicalize a symbol identifier to a string.

    Accepts:
      - str
      - Enum (uses .value)
      - anything else (uses str(x))
    """
    if isinstance(x, Enum):
        return str(x.value)
    return str(x)


def as_dims(dims: Sequence[int]) -> LocalDims:
    return tuple(int(d) for d in dims)


def ket(dim: int, i: int) -> np.ndarray:
    v = np.zeros((int(dim),), dtype=complex)
    v[int(i)] = 1.0 + 0.0j
    return v


def op_ij(dim: int, i: int, j: int) -> np.ndarray:
    # |i><j|
    return np.outer(ket(dim, i), np.conjugate(ket(dim, j)))


def qd_basis_index(state: QDState) -> int:
    """
    Single source of truth for the 4-level dot basis ordering.

    Order: [G, X1, X2, XX]
    """
    if state is QDState.G:
        return 0
    if state is QDState.X1:
        return 1
    if state is QDState.X2:
        return 2
    if state is QDState.XX:
        return 3
    raise KeyError(state)


def qd4_projector(state: QDState) -> np.ndarray:
    i = qd_basis_index(state)
    return op_ij(4, i, i)


def qd4_transition_op(tr: Transition) -> np.ndarray:
    """
    Transition.SRC_DST encodes SRC -> DST.
    Operator convention: |DST><SRC|.
    """
    src_s, dst_s = str(tr.value).split("_", 1)
    src = QDState(src_s)
    dst = QDState(dst_s)
    i_src = qd_basis_index(src)
    i_dst = qd_basis_index(dst)
    return op_ij(4, i_dst, i_src)


def qd4_sigma_x(pair: Transition) -> np.ndarray:
    """
    Convenience: build sigma_x-like operator for a 2-level subspace from a directed
    transition family name.

    Example: pair = Transition.G_XX or Transition.XX_G. We will infer the reverse.
    """
    if pair is Transition.G_XX:
        up = Transition.G_XX
        down = Transition.XX_G
    elif pair is Transition.XX_G:
        up = Transition.G_XX
        down = Transition.XX_G
    else:
        raise ValueError(
            "qd4_sigma_x expects Transition.G_XX or Transition.XX_G"
        )
    return qd4_transition_op(up) + qd4_transition_op(down)


@lru_cache(maxsize=64)
def fock_ops(dim: int) -> dict[str, np.ndarray]:
    """
    Truncated Fock operators for dimension dim >= 1.

    Returns dict with: a, adag, n, I.
    """
    d = int(dim)
    if d <= 0:
        raise ValueError("dim must be >= 1")

    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(float(n))
    adag = np.conjugate(a.T)
    n_op = adag @ a
    ident = np.eye(d, dtype=complex)
    return {"a": a, "adag": adag, "n": n_op, "I": ident}


@lru_cache(maxsize=8)
def qd4_named_ops() -> dict[str, np.ndarray]:
    """
    Friendly names for 4-level dot local operators.

    Includes:
      - projectors: proj_G, proj_X1, proj_X2, proj_XX
      - directed transitions: names equal to Transition values (e.g. "X1_G")
      - aliases: "t_<Transition>" for convenience/backwards-compat
      - sx_G_XX
    """
    ops: dict[str, np.ndarray] = {}

    # projectors (store string keys, not Enum objects)
    ops[str(QDSymbol.PROJ_G.value)] = qd4_projector(QDState.G)
    ops[str(QDSymbol.PROJ_X1.value)] = qd4_projector(QDState.X1)
    ops[str(QDSymbol.PROJ_X2.value)] = qd4_projector(QDState.X2)
    ops[str(QDSymbol.PROJ_XX.value)] = qd4_projector(QDState.XX)

    # all directed transitions (canonical names)
    for tr in Transition:
        name = str(tr.value)  # e.g. "X1_G"
        ops[name] = qd4_transition_op(tr)

        # alias: "t_X1_G" -> same operator
        ops["t_" + name] = ops[name]

    # convenience
    ops[str(QDSymbol.SX_G_XX.value)] = qd4_sigma_x(Transition.G_XX)

    return ops


@dataclass(frozen=True)
class SymbolLibrary:
    """
    Small, testable symbol library for resolving local operators by (dims, symbol).

    This is intentionally independent of SMEF's OpMaterializeContextProto.
    Materializers can wrap this later.
    """

    builders: Mapping[SymbolKey, Builder]

    def resolve(self, symbol: Any, dims: Sequence[int]) -> np.ndarray:
        d = as_dims(dims)
        s = canon_symbol(symbol)
        key = (d, s)
        b = self.builders.get(key)

        if b is None:
            supported = sorted(
                {k[1] for k in self.builders.keys() if k[0] == d}
            )
            raise KeyError(
                "Unknown symbol for dims=%s: %s. Supported: %s"
                % (d, s, supported)
            )

        m = np.asarray(b(d), dtype=complex)
        return m


def build_default_symbol_library(
    *, register_fock_dims: Sequence[int] = (2, 2, 2, 2)
) -> SymbolLibrary:
    """
    Build a default library containing:
      - QD local ops for dims (4,)
      - Fock ops for dims (N,) for N in register_fock_dims
    """
    builders: dict[SymbolKey, Builder] = {}

    # QD (4,)
    qd_ops = qd4_named_ops()
    for name, mat in qd_ops.items():
        builders[((4,), name)] = lambda _d, m=mat: m

    # Fock (N,)
    for d in register_fock_dims:
        for sym in ("a", "adag", "n", "I"):
            builders[((int(d),), sym)] = lambda dims, sym=sym: fock_ops(
                int(dims[0])
            )[sym]

    return SymbolLibrary(builders=builders)
