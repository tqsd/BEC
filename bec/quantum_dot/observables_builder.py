from typing import Dict, Any, List

import numpy as np
import jax.numpy as jnp
from qutip import Qobj

from photon_weave.extra import interpreter
from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.protocols import ModeProvider, ObservableProvider
from bec.operators.qd_operators import QDState, transition_operator


class ObservablesBuilder(ObservableProvider):
    """
    Builds QD and per-mode (two-pol) projectors/observables as QuTiP Qobj,
    using the current symbolic context and a KronPad helper.
    """

    def __init__(
        self,
        context: Dict[str, Any],
        kron: KronPad,
        mode_provider: ModeProvider,
    ):
        self._ctx = context
        self._kron = kron
        self._modes = mode_provider

    # -------------------- QD projectors --------------------

    def qd_projectors(self, dims: List[int]) -> Dict[str, Qobj]:
        """
        Return QD-only projectors as full-dimension Qobj:
            P_G, P_X1, P_X2, P_XX
        """
        # Local 4x4 (QD-space) projectors
        local = {
            "P_G": jnp.array(transition_operator(QDState.G, QDState.G)),
            "P_X1": jnp.array(transition_operator(QDState.X1, QDState.X1)),
            "P_X2": jnp.array(transition_operator(QDState.X2, QDState.X2)),
            "P_XX": jnp.array(transition_operator(QDState.XX, QDState.XX)),
        }

        out: Dict[str, Qobj] = {}
        for name, op in local.items():
            kron_expr = self._kron.pad(op, "i", -1)  # QD ⊗ I_fock_all
            arr = interpreter(kron_expr, self._ctx, dims)
            out[name] = Qobj(np.array(arr), dims=[dims, dims]).to("csr")

        return out

    # -------------------- Per-mode projectors --------------------
    def _to_photonic_operator(
        self, op_full: Qobj, dims_full: list[int]
    ) -> Qobj:
        """
        Given an operator of the form I_QD ⊗ A_phot on the full space with dims_full,
        return the photonic-only operator A_phot by tracing out the QD (first factor)
        and dividing by d_QD.
        """
        d_qd = int(dims_full[0])
        if op_full.dims != [dims_full, dims_full]:
            op_full = Qobj(op_full.full(), dims=[dims_full, dims_full]).to(
                "csr"
            )
        keep = list(range(1, len(dims_full)))  # keep photonic factors
        op_phot = op_full.ptrace(keep) / d_qd  # ptrace(I ⊗ A) = d_QD * A
        return op_phot.to("csr")

    def light_mode_projectors(
        self, dims: List[int], include_qd: bool = True
    ) -> Dict[str, Qobj]:
        """
        For each light mode (each carrying two polarizations '+' and '-') return:

        N[label]          : total photon number (N+ + N-)
        N+[label], N-[label]
        Pvac[label]       : |0,0><0,0|
        P10[label]        : |1,0><1,0|   (one '+' photon)
        P01[label]        : |0,1><1,1|   (one '-' photon)
        P11[label]        : |1,1><1,1|
        S0[label], S1[label] (optional Stokes-like intensities)

        If include_qd is False, return the same operators reduced to the photonic
        subspace (QD traced out and divided by d_QD), with dims set to dims[1:].
        """
        ops: Dict[str, Qobj] = {}
        # --- build full-space operators exactly as you do now ---
        for i, m in enumerate(self._modes.modes):
            label = getattr(m, "label", f"mode_{i}")

            # Symbolic expressions from your DSL, padded across all modes (QD included)
            n_plus_expr = self._kron.pad("idq", "n+", i)
            n_minus_expr = self._kron.pad("idq", "n-", i)
            p_vac_expr = self._kron.pad("idq", "vac", i)
            I_mode_expr = self._kron.pad("idq", "i", i)

            # Evaluate and wrap as Qobj on the full dims
            Np = Qobj(
                np.array(interpreter(n_plus_expr, self._ctx, dims)),
                dims=[dims, dims],
            )
            Nm = Qobj(
                np.array(interpreter(n_minus_expr, self._ctx, dims)),
                dims=[dims, dims],
            )
            P0 = Qobj(
                np.array(interpreter(p_vac_expr, self._ctx, dims)),
                dims=[dims, dims],
            )
            I = Qobj(
                np.array(interpreter(I_mode_expr, self._ctx, dims)),
                dims=[dims, dims],
            )

            # 0/1 subspace projectors per pol: |1><1| = n, |0><0| = I - n
            P1p, P0p = Np, I - Np
            P1m, P0m = Nm, I - Nm

            # Joint occupancy projectors
            P10 = P1p @ P0m
            P01 = P0p @ P1m
            P11 = P1p @ P1m

            # Store (full-space for now)
            ops[f"N[{label}]"] = (Np + Nm).to("csr")
            ops[f"N+[{label}]"] = Np.to("csr")
            ops[f"N-[{label}]"] = Nm.to("csr")
            ops[f"Pvac[{label}]"] = P0.to("csr")
            ops[f"P10[{label}]"] = P10.to("csr")
            ops[f"P01[{label}]"] = P01.to("csr")
            ops[f"P11[{label}]"] = P11.to("csr")

            # Optional Stokes-like intensities
            ops[f"S0[{label}]"] = (Np + Nm).to("csr")
            ops[f"S1[{label}]"] = (Np - Nm).to("csr")

        if include_qd:
            return ops

        # --- reduce to photonic-only and reset dims to dims_phot = dims[1:] ---
        dims_phot = dims[1:]
        reduced: Dict[str, Qobj] = {}
        for k, Op in ops.items():
            A = self._to_photonic_operator(Op, dims)
            # ensure dims metadata is photonic-only
            reduced[k] = Qobj(A.full(), dims=[dims_phot, dims_phot]).to("csr")
        return reduced
