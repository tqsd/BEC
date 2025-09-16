from qutip import basis, tensor
import numpy as np


def pretty_ket(state, dims):
    M = int(np.prod(dims))
    coeffs = state.full().flatten()
    basis_labels = []
    for i in range(M):
        occ = np.base_repr(i, base=max(dims))
        occ = occ.zfill(len(dims))
        basis_labels.append("|" + ",".join(occ) + ">")
    # Build nonzero terms
    terms = []
    for c, lbl in zip(coeffs, basis_labels):
        if abs(c) > 1e-12:
            terms.append(f"({c:.3g}){lbl}")
    return " + ".join(terms)


def pretty_density(rho, dims):
    coeffs = rho.full()
    basis_labels = []
    M = int(np.prod(dims))
    for i in range(M):
        occ = np.base_repr(i, base=max(dims))
        occ = occ.zfill(len(dims))
        basis_labels.append("|" + ",".join(occ) + ">")
    terms = []
    for i in range(M):
        for j in range(M):
            if abs(coeffs[i, j]) > 1e-12:
                terms.append(
                    f"({coeffs[i, j]:.3g}){basis_labels[i]}⟨{basis_labels[j]}|"
                )
    return " + ".join(terms)
