from typing import Dict
from qutip import Qobj
from bec.params.transitions import TransitionType
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import ExpectationLayout


class DefaultExpectationLayout(ExpectationLayout):
    """Mirrors your QDSimulator._make_eops index layout."""

    def select(
        self, qd_proj: Dict[str, Qobj], lm_proj: Dict[str, Qobj], qd: QuantumDot
    ) -> tuple[list[Qobj], Dict[str, slice]]:
        qd_eops = [qd_proj[k] for k in ("P_G", "P_X1", "P_X2", "P_XX")]

        flying = [
            m.label
            for m in qd.modes.modes
            if getattr(m, "source", None) == TransitionType.EXTERNAL
        ]
        intrinsic = [
            m.label
            for m in qd.modes.modes
            if getattr(m, "source", None) == TransitionType.INTERNAL
        ]

        fly_T = [
            lm_proj[f"N[{lbl}]"] for lbl in flying if f"N[{lbl}]" in lm_proj
        ]
        fly_H = [
            lm_proj[f"N-[{lbl}]"] for lbl in flying if f"N-[{lbl}]" in lm_proj
        ]
        fly_V = [
            lm_proj[f"N+[{lbl}]"] for lbl in flying if f"N+[{lbl}]" in lm_proj
        ]
        out_T = [
            lm_proj[f"N[{lbl}]"] for lbl in intrinsic if f"N[{lbl}]" in lm_proj
        ]
        out_H = [
            lm_proj[f"N-[{lbl}]"]
            for lbl in intrinsic
            if f"N-[{lbl}]" in lm_proj
        ]
        out_V = [
            lm_proj[f"N+[{lbl}]"]
            for lbl in intrinsic
            if f"N+[{lbl}]" in lm_proj
        ]

        e_ops = [*qd_eops, *fly_T, *fly_H, *fly_V, *out_T, *out_H, *out_V]
        # index map for unpacking
        i = 0
        idx: Dict[str, slice] = {}
        idx["qd"] = slice(i, i + len(qd_eops))
        i += len(qd_eops)
        idx["fly_T"] = slice(i, i + len(fly_T))
        i += len(fly_T)
        idx["fly_H"] = slice(i, i + len(fly_H))
        i += len(fly_H)
        idx["fly_V"] = slice(i, i + len(fly_V))
        i += len(fly_V)
        idx["out_T"] = slice(i, i + len(out_T))
        i += len(out_T)
        idx["out_H"] = slice(i, i + len(out_H))
        i += len(out_H)
        idx["out_V"] = slice(i, i + len(out_V))
        i += len(out_V)
        return e_ops, idx
