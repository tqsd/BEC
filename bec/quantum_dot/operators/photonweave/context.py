from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from photon_weave.extra import interpreter

from bec.quantum_dot.mode_registry import ModeRegistry
from bec.quantum_dot.operators.photonweave.emit import PW_KRON


SymbolFn = Callable[[Sequence[int]], np.ndarray]
PWCtx = Dict[str, SymbolFn]


def _a(dim: int) -> np.ndarray:
    out = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        out[n - 1, n] = np.sqrt(float(n))
    return out


def _adag(dim: int) -> np.ndarray:
    return _a(dim).conj().T


def _n(dim: int) -> np.ndarray:
    return np.diag(np.arange(dim, dtype=float)).astype(complex)


def _vac(dim: int) -> np.ndarray:
    # Projector onto |0><0|
    out = np.zeros((dim, dim), dtype=complex)
    out[0, 0] = 1.0 + 0.0j
    return out


def _qd_sigma(bra: int, ket: int, dim: int = 4) -> np.ndarray:
    out = np.zeros((dim, dim), dtype=complex)
    out[bra, ket] = 1.0 + 0.0j
    return out


def _parse_qd_symbol(sym: str) -> Tuple[int, int]:
    # Expected "s_X1_G" etc.
    if not sym.startswith("s_"):
        raise ValueError(f"Not a QD operator symbol: {sym!r}")
    parts = sym.split("_")
    if len(parts) != 3:
        raise ValueError(f"QD symbol must look like 's_A_B', got {sym!r}")

    # Keep this consistent with your QD basis ordering used everywhere.
    # Common ordering in your code is (G, X1, X2, XX).
    name_to_idx = {"G": 0, "X1": 1, "X2": 2, "XX": 3}

    bra_name = parts[1]
    ket_name = parts[2]
    if bra_name not in name_to_idx or ket_name not in name_to_idx:
        raise ValueError(f"Unknown QD state in symbol {sym!r}")

    return name_to_idx[bra_name], name_to_idx[ket_name]


def build_photonweave_symbol_table(modes: ModeRegistry) -> PWCtx:
    """
    Build ctx: symbol -> callable(dims) -> local operator matrix.

    Conventions:
      - dims[0] is QD dimension (should be 4)
      - dims[1 + i] is the local dimension for mode i in modes.channels order
    """
    ctx: PWCtx = {}

    # QD coherences/projectors: "s_X1_G", "s_XX_XX", etc.
    def qd_factory(sym: str) -> SymbolFn:
        bra, ket = _parse_qd_symbol(sym)

        def fn(dims: Sequence[int]) -> np.ndarray:
            qd_dim = int(dims[0])
            if qd_dim != 4:
                # If you ever generalize QD dim, change _qd_sigma and parsing accordingly.
                raise ValueError(f"Expected QD dim 4, got {qd_dim}")
            return _qd_sigma(bra, ket, dim=qd_dim)

        return fn

    # We do not pre-enumerate all s_*_* symbols; instead, we register them on demand
    # in PhotonWeaveMaterializeContext.resolve_symbol via _ensure_qd_symbol.

    # Identity placeholders for each photonic subsystem: "if0", "if1", ...
    # These are local identities for the kron slots.
    for i in range(len(modes.channels)):
        key = f"if{i}"

        def make_if(ii: int) -> SymbolFn:
            def fn(dims: Sequence[int]) -> np.ndarray:
                d = int(dims[1 + ii])
                return np.eye(d, dtype=complex)

            return fn

        ctx[key] = make_if(i)

    # Photonic operators per channel index: a{i}, a{i}_dag, n{i}, vac{i}
    for i in range(len(modes.channels)):

        def make_a(ii: int) -> SymbolFn:
            def fn(dims: Sequence[int]) -> np.ndarray:
                return _a(int(dims[1 + ii]))

            return fn

        def make_adag(ii: int) -> SymbolFn:
            def fn(dims: Sequence[int]) -> np.ndarray:
                return _adag(int(dims[1 + ii]))

            return fn

        def make_n(ii: int) -> SymbolFn:
            def fn(dims: Sequence[int]) -> np.ndarray:
                return _n(int(dims[1 + ii]))

            return fn

        def make_vac(ii: int) -> SymbolFn:
            def fn(dims: Sequence[int]) -> np.ndarray:
                return _vac(int(dims[1 + ii]))

            return fn

        ctx[f"a{i}"] = make_a(i)
        ctx[f"a{i}_dag"] = make_adag(i)
        ctx[f"n{i}"] = make_n(i)
        ctx[f"vac{i}"] = make_vac(i)

    # Expose the on-demand qd_factory via a private hook
    ctx["_qd_factory"] = qd_factory  # type: ignore[assignment]
    return ctx


@dataclass
class PhotonWeaveMaterializeContext:
    """
    Provides:
      - pw_ctx: symbol table for interpret(...)
      - resolve_symbol(key, dims): returns FULL embedded operator (D,D)

    resolve_symbol always embeds using a ('kron', key, 'if0', 'if1', ...) expression,
    so callers do not need to care if 'key' is local or already embedded.
    """

    modes: ModeRegistry
    pw_ctx: PWCtx = field(init=False)
    _cache: MutableMapping[Tuple[str, Tuple[int, ...]], np.ndarray] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.pw_ctx = build_photonweave_symbol_table(self.modes)

    def _ensure_qd_symbol(self, key: str) -> None:
        if key in self.pw_ctx:
            return
        qd_factory = self.pw_ctx.get("_qd_factory", None)
        if qd_factory is None:
            raise RuntimeError("PhotonWeave ctx missing _qd_factory hook")
        self.pw_ctx[key] = qd_factory(key)  # type: ignore[misc]

    def resolve_symbol(self, key: str, dims: Sequence[int]) -> np.ndarray:
        dkey = (str(key), tuple(int(x) for x in dims))
        hit = self._cache.get(dkey, None)
        if hit is not None:
            return hit

        # If it looks like a QD symbol, ensure it is registered.
        if key.startswith("s_"):
            self._ensure_qd_symbol(key)

        # Build a FULL embedding expression: ("kron", key, "if0", "if1", ...)
        # Slot 0 is QD; slots 1.. are photonic channels.
        atoms: list[Any] = [PW_KRON, key]
        for i in range(len(self.modes.channels)):
            atoms.append(f"if{i}")
        expr = tuple(atoms)

        A = interpreter(expr, self.pw_ctx, list(dims))
        out = np.asarray(A, dtype=complex)
        self._cache[dkey] = out
        return out

    def materialize_expr(self, expr: Any, dims: Sequence[int]) -> np.ndarray:
        A = interpreter(expr, self.pw_ctx, list(dims))
        return np.asarray(A, dtype=complex)
