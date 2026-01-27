from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _as_cd(x: Any) -> complex:
    return complex(x)


def _trapz_complex(y: np.ndarray, x: np.ndarray) -> complex:
    # numpy.trapezoid supports complex in recent numpy, but keep explicit.
    return complex(np.trapezoid(y.real, x)) + 1j * complex(
        np.trapezoid(y.imag, x)
    )


def _fmt_c(z: complex, digits: int = 6) -> str:
    return (
        f"{z.real:.{digits}g}{'+' if z.imag >= 0 else ''}{z.imag:.{digits}g}j"
    )


def _topk_abs(
    A: np.ndarray, k: int = 10
) -> List[Tuple[float, Tuple[int, int], complex]]:
    # Return top-k entries by absolute value: (abs, (i,j), val)
    flat = A.reshape(-1)
    if flat.size == 0:
        return []
    idx = np.argpartition(np.abs(flat), -min(k, flat.size))[
        -min(k, flat.size) :
    ]
    idx = idx[np.argsort(-np.abs(flat[idx]))]
    out = []
    n = A.shape[1]
    for p in idx:
        i = int(p // n)
        j = int(p % n)
        out.append((float(abs(A[i, j])), (i, j), complex(A[i, j])))
    return out


def _basis_index(dims: List[int], ket: List[int]) -> int:
    """
    Convert tensor-product basis label (list of local indices) to flat index.
    Convention: dims = [d0,d1,...], ket = [i0,i1,...]
    Flat index = (((i0*d1 + i1)*d2 + i2)*...) .
    """
    if len(dims) != len(ket):
        raise ValueError(f"ket rank {len(ket)} != dims rank {len(dims)}")
    idx = 0
    for d, i in zip(dims, ket):
        if not (0 <= i < d):
            raise ValueError(f"local index {i} out of range for dim {d}")
        idx = idx * d + i
    return idx


def _ket_vector(dims: List[int], ket: List[int]) -> np.ndarray:
    v = np.zeros((int(np.prod(dims)),), dtype=np.complex128)
    v[_basis_index(dims, ket)] = 1.0 + 0j
    return v


# ---------------------------------------------------------------------
# Drive/Hamiltonian Audit
# ---------------------------------------------------------------------


@dataclass
class TermAudit:
    label: str
    kind: str
    op_shape: Tuple[int, int]
    op_fro: float
    coeff_peak: complex
    coeff_peak_t: float
    coeff_minabs: float
    coeff_maxabs: float
    coeff_area: complex
    # a few largest matrix elements
    op_top: List[Tuple[float, Tuple[int, int], complex]]
    # optional specific matrix elements
    matrix_elems: Dict[str, complex]


@dataclass
class DriveAudit:
    """
    Audit compiled MEProblem terms.

    What it tells you for each selected Hamiltonian term:
      - operator norms and top matrix elements (structure sanity)
      - coefficient min/max/peak and integral on *solver* tlist
      - optional specific bra/ket matrix elements (e.g. <XX,vac|H_op|G,vac>)
    """

    me: Any  # MEProblem
    tlist: np.ndarray
    args: Dict[str, Any]

    def audit_h_terms(
        self,
        *,
        label_substr: Optional[str] = None,
        kind_contains: Optional[str] = "DRIVE",
        max_terms: int = 10,
        sample_points: int = 200,
        topk: int = 12,
        # Optional explicit matrix element checks.
        # Provide global basis kets as lists of local indices, same length as me.dims.
        bra_ket_checks: Optional[List[Tuple[str, List[int], List[int]]]] = None,
    ) -> List[TermAudit]:
        ts_full = np.asarray(self.tlist, dtype=float)
        # Downsample for peak/min/max discovery cheaply
        if sample_points >= len(ts_full):
            ts = ts_full
        else:
            ts = np.linspace(ts_full[0], ts_full[-1], sample_points)

        out: List[TermAudit] = []

        for term in self.me.h_terms:
            lab = str(getattr(term, "label", "") or "")
            kind = getattr(term, "kind", None)
            kind_name = getattr(kind, "name", str(kind))

            if (
                label_substr is not None
                and label_substr.lower() not in lab.lower()
            ):
                continue
            if kind_contains is not None and kind_contains not in kind_name:
                continue

            op = np.asarray(term.op, dtype=np.complex128)
            op_fro = float(np.linalg.norm(op))

            # coeff stats
            if term.coeff is None:
                coeff_vals = np.zeros_like(ts, dtype=np.complex128)
                coeff_vals_full = np.zeros_like(ts_full, dtype=np.complex128)
            else:
                coeff_vals = np.array(
                    [term.coeff(float(t), dict(self.args)) for t in ts],
                    dtype=np.complex128,
                )
                coeff_vals_full = np.array(
                    [term.coeff(float(t), dict(self.args)) for t in ts_full],
                    dtype=np.complex128,
                )

            abs_vals = np.abs(coeff_vals)
            imax = int(np.argmax(abs_vals)) if abs_vals.size else 0
            coeff_peak = (
                complex(coeff_vals[imax]) if coeff_vals.size else 0.0 + 0.0j
            )
            coeff_peak_t = float(ts[imax]) if ts.size else float(ts_full[0])
            coeff_minabs = (
                float(np.min(np.abs(coeff_vals_full)))
                if coeff_vals_full.size
                else 0.0
            )
            coeff_maxabs = (
                float(np.max(np.abs(coeff_vals_full)))
                if coeff_vals_full.size
                else 0.0
            )

            coeff_area = _trapz_complex(coeff_vals_full, ts_full)

            # operator structure
            op_top = _topk_abs(op, k=topk)

            # explicit matrix elements
            elems: Dict[str, complex] = {}
            if bra_ket_checks:
                dims = list(self.me.dims)
                for name, bra, ket in bra_ket_checks:
                    vb = _ket_vector(dims, bra)
                    vk = _ket_vector(dims, ket)
                    val = complex(vb.conj() @ (op @ vk))
                    elems[name] = val

            out.append(
                TermAudit(
                    label=lab,
                    kind=kind_name,
                    op_shape=tuple(op.shape),
                    op_fro=op_fro,
                    coeff_peak=coeff_peak,
                    coeff_peak_t=coeff_peak_t,
                    coeff_minabs=coeff_minabs,
                    coeff_maxabs=coeff_maxabs,
                    coeff_area=coeff_area,
                    op_top=op_top,
                    matrix_elems=elems,
                )
            )

            if len(out) >= max_terms:
                break

        return out

    def pretty_print(self, audits: List[TermAudit]) -> None:
        print("\n" + "=" * 80)
        print(" DRIVE / HAMILTONIAN TERM AUDIT")
        print("=" * 80)
        print(f"dims: {list(self.me.dims)}  D={int(np.prod(self.me.dims))}")
        print(
            f"tlist: N={len(self.tlist)}  [{float(self.tlist[0])}, {
              float(self.tlist[-1])}] (solver units)"
        )
        print("-" * 80)

        if not audits:
            print("No matching Hamiltonian terms found.")
            print("=" * 80)
            return

        for i, a in enumerate(audits):
            print(f"[{i:02d}] kind={a.kind}  label={a.label}")
            print(f"     op: shape={a.op_shape}  ||op||_F={a.op_fro:.6g}")
            print(
                f"     coeff: |min|={
                  a.coeff_minabs:.6g}  |max|={a.coeff_maxabs:.6g}"
            )
            print(
                f"            peak @ t={a.coeff_peak_t:.6g}: {_fmt_c(a.coeff_peak)}"
            )
            print(
                f"            area ∫coeff dt (solver units): {
                  _fmt_c(a.coeff_area)}"
            )
            print("     op top entries (abs, (i,j), val):")
            for absval, (r, c), v in a.op_top:
                print(f"        {absval:.6g}  ({r:4d},{c:4d})  {_fmt_c(v)}")
            if a.matrix_elems:
                print("     selected matrix elements:")
                for k, v in a.matrix_elems.items():
                    print(f"        {k}: {_fmt_c(v)}")
            print("-" * 80)

        print("=" * 80)


# ---------------------------------------------------------------------
# Example usage inside your script (minimal)
# ---------------------------------------------------------------------
#
#   audit = DriveAudit(me=me, tlist=np.asarray(me.tlist, float), args=dict(me.args))
#   checks = [
#       # Example for your ordering: [QD, m0+, m0-, m1+, m1-, ...]
#       # If you want <XX,vac|H_op|G,vac> == 1+0j:
#       ("<XX,vac|H_op|G,vac>", [3, 0,0, 0,0, 0,0, 0,0], [0, 0,0, 0,0, 0,0, 0,0]),
#   ]
#   audits = audit.audit_h_terms(label_substr="TPE", kind_contains="DRIVE", bra_ket_checks=checks)
#   audit.pretty_print(audits)
#
# Adjust the QD basis indices: here I assumed QD local basis indices:
#   G=0, X1=1, X2=2, XX=3
# and vacuum in each mode local index = 0.
