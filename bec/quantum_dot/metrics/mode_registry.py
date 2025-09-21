# bec/quantum_dot/mode_registry.py
from __future__ import annotations
from typing import Any, List
import numpy as np
from qutip import Qobj, basis, qeye, tensor

from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.quantum_dot.metrics.registry import PhotonicRegistry


def build_registry(
    qd: Any, mode_provider: Any, observable_provider: Any
) -> PhotonicRegistry:
    """
    Construct a PhotonicRegistry from the QD object and providers.

    Assumptions:
      - Each spatio-temporal mode has two polarization factors in order: '+' then '-'.
      - observable_provider.light_mode_projectors(...) returns keys "N+[label]" and "N-[label]".
    """
    early_facts, late_facts, _p, _m, dims_phot, offset = (
        infer_index_sets_from_registry(qd, rho_has_qd=False)
    )

    # labels for each mode index (used for mapping factor -> "N±[label]")
    modes = getattr(mode_provider, "intrinsic", None) or mode_provider.modes
    labels: List[str] = [
        getattr(m, "label", f"mode_{i}") for i, m in enumerate(modes)
    ]

    # pull number projectors once (photonic-only operators)
    dims_full = [4] + list(dims_phot)  # 4 = QD dimension
    ops_phot = observable_provider.light_mode_projectors(
        dims_full, include_qd=False
    )

    # map photonic factor index -> number operator
    def factor_to_N(f_idx: int) -> Qobj:
        k = f_idx - offset  # 0-based within photonic block
        i_mode, rem = divmod(k, 2)  # 2 polarizations per mode
        pol = "+" if rem == 0 else "-"
        label = getattr(modes[i_mode], "label", f"mode_{i_mode}")
        key = f"N{pol}[{label}]"  # matches your observables_builder keys
        return ops_phot[key].to("csr")

    all_facts = list(early_facts) + list(late_facts)
    number_op_by_factor = {f: factor_to_N(f) for f in all_facts}

    proj0_by_factor = {}
    proj1_by_factor = {}

    for f_idx in all_facts:
        p_factor = f_idx - offset  # 0-based index into the photonic tail
        d_local = dims_phot[
            p_factor
        ]  # local dim for THIS factor (not the mode)

        ket0 = basis(d_local, 0)
        ket1 = basis(d_local, 1)
        P0 = ket0 * ket0.dag()  # dims [[d_local],[d_local]]
        P1 = ket1 * ket1.dag()

        ops0, ops1 = [], []
        for p, d_p in enumerate(dims_phot):
            if p == p_factor:
                ops0.append(P0)
                ops1.append(P1)
            else:
                ops0.append(qeye(d_p))
                ops1.append(qeye(d_p))

        proj0_by_factor[f_idx] = tensor(ops0).to("csr")
        proj1_by_factor[f_idx] = tensor(ops1).to("csr")

    # identity on photonic space
    Dp = int(np.prod(dims_phot))
    I_phot = Qobj(np.eye(Dp, dtype=complex), dims=[dims_phot, dims_phot]).to(
        "csr"
    )

    return PhotonicRegistry(
        dims_phot=list(dims_phot),
        early_factors=list(early_facts),
        late_factors=list(late_facts),
        offset=offset,
        labels_by_mode_index=labels,
        number_op_by_factor=number_op_by_factor,
        proj0_by_factor=proj0_by_factor,
        proj1_by_factor=proj1_by_factor,
        I_phot=I_phot,
    )
