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

    def light_mode_projectors(self, dims: List[int]) -> Dict[str, Qobj]:
        """
        For each light mode (each carrying two polarizations '+' and '-') return:

          N[label]          : total photon number (N+ + N-)
          N+[label], N-[label]
          Pvac[label]       : |0,0><0,0|
          P10[label]        : |1,0><1,0|   (one '+' photon)
          P01[label]        : |0,1><0,1|   (one '-' photon)
          P11[label]        : |1,1><1,1|
          S0[label], S1[label] (optional Stokes-like intensities)

        Notes
        -----
        Assumes 0/1 truncation per polarization subspace where:
            |1><1| ≡ n,   |0><0| ≡ I - n
        """
        ops: Dict[str, Qobj] = {}

        for i, m in enumerate(self._modes.modes):
            label = getattr(m, "label", f"mode_{i}")

            # Symbolic expressions from your DSL, padded across all modes
            n_plus_expr = self._kron.pad("idq", "n+", i)
            n_minus_expr = self._kron.pad("idq", "n-", i)
            p_vac_expr = self._kron.pad("idq", "vac", i)
            I_mode_expr = self._kron.pad("idq", "i", i)

            # Evaluate and wrap
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

            # Joint occupancy for the two-pol mode (tensor already implied in dims)
            P10 = P1p @ P0m
            P01 = P0p @ P1m
            P11 = P1p @ P1m

            # Store
            ops[f"N[{label}]"] = Np + Nm
            ops[f"N+[{label}]"] = Np
            ops[f"N-[{label}]"] = Nm
            ops[f"Pvac[{label}]"] = P0
            ops[f"P10[{label}]"] = P10
            ops[f"P01[{label}]"] = P01
            ops[f"P11[{label}]"] = P11

            # Optional Stokes-like intensities
            ops[f"S0[{label}]"] = Np + Nm
            ops[f"S1[{label}]"] = Np - Nm

        return ops
