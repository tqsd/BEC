from qutip import Qobj
import numpy as np


def flat_index_from_occ(occ, dims):
    # little-endian as in your projector code
    s = [1]
    for d in dims[:-1]:
        s.append(s[-1] * d)
    return sum(o * si for o, si in zip(occ, s))


def coherence_HV_VH(rho, dims, e_plus, e_minus, l_plus, l_minus):
    """
    Returns (p_HV, p_VH, c) where:
      p_HV ≡ P(|e+, l->), p_VH ≡ P(|e-, l+|), c ≡ <e+,l-| rho |e-,l+>
    Indices (e_plus, e_minus, l_plus, l_minus) are factor positions in `dims`.
    """
    M = len(dims)

    def ket_occ(indices_one):
        occ = [0] * M
        for k in indices_one:
            occ[k] = 1
        return occ

    # |HV> ≡ |e+, l->
    occ_HV = ket_occ([e_plus, l_minus])
    iHV = flat_index_from_occ(occ_HV, dims)

    # |VH> ≡ |e-, l+>
    occ_VH = ket_occ([e_minus, l_plus])
    iVH = flat_index_from_occ(occ_VH, dims)

    if isinstance(rho, Qobj):
        mat = rho.full()
    else:
        mat = np.asarray(rho)

    p_HV = float(mat[iHV, iHV].real)
    p_VH = float(mat[iVH, iVH].real)
    # same as mat[iHV, iVH]; use explicit:
    c = complex(mat[iHV, iVH * 0 + iVH])
    c = complex(mat[iHV, iVH])
    return p_HV, p_VH, c
