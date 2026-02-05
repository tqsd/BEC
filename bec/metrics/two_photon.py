from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bec.metrics.linops import _dims_prod, partial_trace


@dataclass(frozen=True)
class TwoPhotonResult:
    p11: float
    # (4,4) postselected polarization state, if p11>0 else zeros
    rho_pol: np.ndarray


def _select_one_photon_subspace_two_modes(
    rho_2m: np.ndarray, d: int
) -> np.ndarray:
    """
    Given rho on two Fock modes (H,V) with local dim d,
    return the projected density matrix onto total photon number == 1
    in the basis {|1,0>, |0,1>} (i.e. H or V single photon).
    Output is 2x2 (not normalized).
    """
    d = int(d)
    if rho_2m.shape != (d * d, d * d):
        raise ValueError("rho_2m shape mismatch")

    # basis indices for |1,0> and |0,1> in flattening with dims (d,d)
    i10 = np.ravel_multi_index((1, 0), (d, d))
    i01 = np.ravel_multi_index((0, 1), (d, d))
    idx = [int(i10), int(i01)]
    return rho_2m[np.ix_(idx, idx)]


def two_photon_postselect(
    rho_full: np.ndarray,
    dims: Sequence[int],
    early_indices: Sequence[int],
    late_indices: Sequence[int],
) -> TwoPhotonResult:
    """
    early_indices: [idx_H, idx_V] for early band (e.g. XX_H, XX_V)
    late_indices:  [idx_H, idx_V] for late band (e.g. GX_H, GX_V)

    Steps:
      - trace out everything except early+late optical modes (4 modes)
      - project early total n==1 and late total n==1
      - return p11 and normalized 4x4 polarization rho in basis
        {|H_e H_l>, |H_e V_l>, |V_e H_l>, |V_e V_l>}
    """
    dims = [int(d) for d in dims]
    e = [int(x) for x in early_indices]
    l = [int(x) for x in late_indices]
    keep = e + l

    rho_opt = partial_trace(rho_full, dims, keep=keep)

    # reorder rho_opt into tensor order (eH,eV,lH,lV) already if keep is in that order
    de = dims[e[0]]
    if dims[e[1]] != de:
        raise ValueError("early H/V dims mismatch")
    dl = dims[l[0]]
    if dims[l[1]] != dl:
        raise ValueError("late H/V dims mismatch")

    # Now we need reduced states for early pair and late pair, but projection is joint.
    # We'll do it by reshaping rho_opt to 8 indices and extracting the 2x2 blocks for each side.

    # rho_opt acts on (eH,eV,lH,lV)
    d_opt = [de, de, dl, dl]
    Dopt = _dims_prod(d_opt)
    if rho_opt.shape != (Dopt, Dopt):
        raise ValueError("rho_opt shape mismatch")

    # Project early to {|10>,|01>} and late to {|10>,|01>}.
    # This is equivalent to selecting indices where early basis is i10 or i01
    # and late basis is j10 or j01.
    i10_e = np.ravel_multi_index((1, 0), (de, de))
    i01_e = np.ravel_multi_index((0, 1), (de, de))
    i10_l = np.ravel_multi_index((1, 0), (dl, dl))
    i01_l = np.ravel_multi_index((0, 1), (dl, dl))

    # Build flat indices in the 4-mode space: (early_pair_index, late_pair_index)
    # where early_pair_index in [0..de*de-1], late_pair_index in [0..dl*dl-1]
    def flat(ie: int, il: int) -> int:
        return int(np.ravel_multi_index((int(ie), int(il)), (de * de, dl * dl)))

    basis = [
        flat(i10_e, i10_l),  # H_e, H_l
        flat(i10_e, i01_l),  # H_e, V_l
        flat(i01_e, i10_l),  # V_e, H_l
        flat(i01_e, i01_l),  # V_e, V_l
    ]

    rho_pol_unnorm = rho_opt[np.ix_(basis, basis)]
    p11 = float(np.real_if_close(np.trace(rho_pol_unnorm)))

    if p11 <= 0.0:
        return TwoPhotonResult(p11=0.0, rho_pol=np.zeros((4, 4), dtype=complex))

    rho_pol = rho_pol_unnorm / p11
    return TwoPhotonResult(p11=p11, rho_pol=rho_pol)
