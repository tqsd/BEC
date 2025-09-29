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
    Constructs a QD and photonic mode observables as QuTiP Qobj.

    This builder uses the context and KronPad to create:
    - QD projectors (P_G, P_X1, P_X2, P_XX)
    - Photonic mode numer operators and projectors for the two polarizations
      N[label], N+[label], N-[label], Pvac[label], P10[label], P01[label],
      P11[label]
    - Optional Stokes-like intensities S0[label], S1[label]

    It can also return photonic only operators by tracing out the
    QD factor

    Parameters:
    -----------
    context: dict[str, callable]
        Symbolic operator context (`photon_weave.extra.interpreter`)
    kron: KronPad
        Helper that pads local QD and photonic mode operators
    mode_provider: ModeProvider
        provides the list of light modes (each with a label and two
        polarizations)
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

    def qd_projectors(self, dims: List[int]) -> Dict[str, Qobj]:
        r"""
        Build QD-only projectors and embed them into the full space.

        The following projectors are returned, each padded as
        `QD_op \otimes I_phot`:
        - "P_G"  : |G><G|
        - "P_X1" : |X1><X1|
        - "P_X2" : |X2><X2|
        - "P_XX" : |XX><XX|

        Parameters
        ----------
        dims : list[int]
            Composite Hilbert space dimensions. The first entry is the QD
            dimension; the remaining entries are per-mode photonic factors.

        Returns
        -------
        Dict[str, qutip.Qobj]
            Mapping from label to CSR Qobj with `dims=[dims, dims]`.
        """
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

    def _to_photonic_operator(
        self, op_full: Qobj, dims_full: list[int]
    ) -> Qobj:
        r"""
        Reduce a full-space operator `I_QD \otimes A_phot` to the photonic
        subspace.

        The reduction is `ptrace` over the QD factor (index 0), followed by
        division by the QD dimension so that:
        .. math::
            \text{ptrace}(I_{QD}\otimes A_{phot}) / d_{QD} = A_{phot}

        Parameters
        ----------
        op_full : qutip.Qobj
            Operator defined on the full composite space
        dims_full : list[int]
            Full composite dimensions,

        Returns
        -------
        qutip.Qobj
            Photonic-only operator with (CSR)

        Raises
        ------
        ValueError
            If the operator cannot be reshaped to the provided `dims_full`.
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
        Build per-mode projectors and number operators for both polarizations.

        For each registered light mode with label `L`, the following operators
        are constructed on the full composite space:

        Counts
        ------
        - "N[L]"   : total photon number on the mode (N+ + N-)
        - "N+[L]"  : photon number in '+' polarization
        - "N-[L]"  : photon number in '-' polarization

        Occupancy projectors (0/1 subspace per polarization)
        ----------------------------------------------------
        - "Pvac[L]": |0,0><0,0|
        - "P10[L]" : |1,0><1,0|  (one '+' photon, zero '-' photons)
        - "P01[L]" : |0,1><0,1|  (zero '+' photons, one '-' photon)
        - "P11[L]" : |1,1><1,1|  (one '+' and one '-' photon)

        Stokes-like intensities (optional)
        ----------------------------------
        - "S0[L]"  : N+ + N-
        - "S1[L]"  : N+ - N-

        If `include_qd` is False, each operator is reduced to the photonic
        subspace by tracing out the QD factor and dividing by the QD dimension;
        the returned Qobj then have `dims=[dims[1:], dims[1:]]`.

        Parameters
        ----------
        dims : list[int]
            Composite Hilbert space dimensions. `dims[0]` is the QD dimension;
        include_qd : bool, optional
            If True (default), return full-space operators. If False, return
            photonic-only operators as described above.

        Returns
        -------
        Dict[str, qutip.Qobj]
            Dictionary of per-mode operators in CSR format with appropriate
            `dims`.
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
