from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Iterable, List, Sequence

try:
    from qutip import Qobj
except Exception:
    Qobj = None  # allow import without QuTiP in non-runtime contexts


# ---------- low-level utilities (little-endian tensor layout) ----------


def _to_ndarray(rho) -> np.ndarray:
    """Return a dense ndarray view of `rho` (Qobj or array-like)."""
    if Qobj is not None and isinstance(rho, Qobj):
        return np.array(rho.full())
    return np.asarray(rho)


def _strides(dims: Sequence[int]) -> List[int]:
    """Little-endian strides for column-stacked tensor factors."""
    s = [1]
    for d in dims[:-1]:
        s.append(s[-1] * d)
    return s


def _flat_index(occ: Sequence[int], dims: Sequence[int]) -> int:
    """
    Map a bitstring/occupation list (little-endian) to the flat basis index.
    `occ[k]` is the local occupation in factor k; dims[k] its local dimension.
    """
    st = _strides(dims)
    return int(sum(o * si for o, si in zip(occ, st)))


# ---------- pattern/projector constructors in NumPy and QuTiP ----------


def all_bitstrings_with_weight(M: int, w: int) -> Iterable[List[int]]:
    """Generate all 0/1 occupation lists of length M with Hamming weight w."""
    for idxs in combinations(range(M), w):
        occ = [0] * M
        for k in idxs:
            occ[k] = 1
        yield occ


def projector_on_patterns_np(
    patterns: Iterable[Sequence[int]], dims: Sequence[int]
) -> np.ndarray:
    """
    Build a NumPy projector P = Σ |pattern⟩⟨pattern| over given occupation lists.
    Each pattern is a 0/1 vector (length = len(dims)) in the number basis.
    """
    D = int(np.prod(dims))
    P = np.zeros((D, D), dtype=complex)
    st = _strides(dims)
    for occ in patterns:
        i = int(sum(o * si for o, si in zip(occ, st)))
        P[i, i] += 1.0
    return P


def projector_on_patterns_qobj(
    patterns: Iterable[Sequence[int]], dims: Sequence[int]
):
    """
    QuTiP version of the projector builder. Returns a Qobj with dims=[dims, dims].
    """
    if Qobj is None:
        raise RuntimeError(
            "QuTiP is not available; cannot build Qobj projectors."
        )
    D = int(np.prod(dims))
    data = np.zeros((D, D), dtype=complex)
    st = _strides(dims)
    for occ in patterns:
        i = int(sum(o * si for o, si in zip(occ, st)))
        data[i, i] = 1.0
    return Qobj(data, dims=[list(dims), list(dims)]).to("csr")


def trace_proj_np(
    rho, patterns: Iterable[Sequence[int]], dims: Sequence[int]
) -> float:
    """
    Compute Tr[P rho] in pure NumPy. Accepts `rho` as Qobj or array-like.
    """
    rhoA = _to_ndarray(rho)
    P = projector_on_patterns_np(patterns, dims)
    return float(np.trace(P @ rhoA).real)


def _trace_proj(
    rho, patterns: Iterable[Sequence[int]], dims: Sequence[int]
) -> float:
    """
    Compute Tr[P(patterns) rho]. Uses QuTiP ops if `rho` is a Qobj, else NumPy.
    """
    if Qobj is not None and isinstance(rho, Qobj):
        P = projector_on_patterns_qobj(patterns, dims)
        return float((P * rho).tr().real)
    # numpy path
    return trace_proj_np(rho, patterns, dims)


# ---------- main metrics ----------


def two_photon_metrics_multimode(
    rho,
    dims: Sequence[int],
    early_modes: Sequence[int],
    late_modes: Sequence[int],
    H_modes: Sequence[int] | None = None,
    V_modes: Sequence[int] | None = None,
):
    """
    Phase-agnostic, multimode two-photon diagnostics on a truncated 0/1-per-factor space.

    Args
    ----
    rho : Qobj | ndarray
        Photonic density matrix on ⊗_k C^{dims[k]} (QD already traced out).
    dims : list[int]
        Local dimensions of photonic factors; for 4 modes × 2 pol with {0,1} cutoff: [2]*8.
    early_modes : list[int]
        Factor indices (across both polarisations) that belong to the first jump (XX→X_i).
    late_modes : list[int]
        Factor indices (across both polarisations) that belong to the second jump (X_i→G).
    H_modes, V_modes : list[int] | None
        Optional polarity partition (e.g., “+”/“−” or H/V) across all involved modes.

    Returns
    -------
    dict with keys:
        P2   : total two-photon probability across all factors
        Pel  : probability of one photon in early and one in late (proper cascade)
        Pee  : probability of two photons both in early factors
        Pll  : probability of two photons both in late factors
        (optional) P_HH, P_VV, P_HV, P_VH within the (early, late) pairing
    """
    M = len(dims)
    if any(d != 2 for d in dims):
        # This implementation assumes {0,1} occupancy in each factor
        raise ValueError(
            "two_photon_metrics_multimode requires dims[k]==2 for all k."
        )

    # --- global N = 2 ---
    patterns_N2 = list(all_bitstrings_with_weight(M, 2))
    P2 = _trace_proj(rho, patterns_N2, dims)

    # --- early + late (proper cascade) ---
    patterns_EL = []
    Eset, Lset = list(early_modes), list(late_modes)
    for i in Eset:
        for j in Lset:
            if i == j:
                continue
            occ = [0] * M
            occ[i] = occ[j] = 1
            patterns_EL.append(occ)
    Pel = _trace_proj(rho, patterns_EL, dims)

    # --- both early (EE) and both late (LL) ---
    patterns_EE, patterns_LL = [], []
    for i, j in combinations(Eset, 2):
        occ = [0] * M
        occ[i] = occ[j] = 1
        patterns_EE.append(occ)
    for i, j in combinations(Lset, 2):
        occ = [0] * M
        occ[i] = occ[j] = 1
        patterns_LL.append(occ)
    Pee = _trace_proj(rho, patterns_EE, dims)
    Pll = _trace_proj(rho, patterns_LL, dims)

    out = dict(P2=P2, Pel=Pel, Pee=Pee, Pll=Pll)

    # --- optional polarization-resolved split within (early, late) pairing ---
    if H_modes is not None and V_modes is not None:
        Hset, Vset = set(H_modes), set(V_modes)
        Eset, Lset = set(early_modes), set(late_modes)

        def pats(first_set, second_set):
            pats_ = []
            for i in Eset & first_set:
                for j in Lset & second_set:
                    occ = [0] * M
                    occ[i] = occ[j] = 1
                    pats_.append(occ)
            return pats_

        P_HH = _trace_proj(rho, pats(Hset, Hset), dims)
        P_VV = _trace_proj(rho, pats(Vset, Vset), dims)
        P_HV = _trace_proj(rho, pats(Hset, Vset), dims)
        P_VH = _trace_proj(rho, pats(Vset, Hset), dims)
        out.update(dict(P_HH=P_HH, P_VV=P_VV, P_HV=P_HV, P_VH=P_VH))

    return out


# ---------- optional: quick moment-based estimate (when P(N>=3)≈0) ----------


def approx_P2_from_moments(rho, number_ops: Sequence[np.ndarray]) -> float:
    """
    Approximate P(N=2) via ½⟨N(N−1)⟩ with N = Σ n_i.
    Only valid when probability of N≥3 is negligible.

    Args
    ----
    rho : ndarray (dense). If Qobj, pass rho.full().
    number_ops : list of dense matrices (same size as rho) for each factor.

    Returns
    -------
    float : ~ P2
    """
    rhoA = _to_ndarray(rho)
    N_op = np.zeros_like(rhoA)
    for n in number_ops:
        N_op = N_op + n
    I = np.eye(N_op.shape[0], dtype=N_op.dtype)
    N2m1 = N_op @ (N_op - I)
    return float(0.5 * np.trace(N2m1 @ rhoA).real)


__all__ = [
    "all_bitstrings_with_weight",
    "projector_on_patterns_np",
    "projector_on_patterns_qobj",
    "trace_proj_np",
    "two_photon_metrics_multimode",
    "approx_P2_from_moments",
]
