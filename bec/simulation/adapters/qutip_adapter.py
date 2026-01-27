from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
from qutip import Qobj, mesolve, Options, QobjEvo

from bec.simulation.types import MEProblem, MESimulationResult

QutipCoeff = Callable[[float, Dict[str, Any]], complex]
QutipHItem = Union[Qobj, List[object]]  # [Qobj, QutipCoeff]
QutipCItem = Union[Qobj, List[object]]  # [Qobj, QutipCoeff]


@dataclass
class QuTiPAdapter:
    backend_name: str = "qutip"
    store_states: bool = False
    store_final_state: bool = True
    sort_observables: bool = True
    progress: bool = True
    force_csr: bool = True
    _op_cache: Optional[Dict[tuple, Qobj]] = None  # add in __post_init__

    def __post_init__(self):
        object.__setattr__(self, "_op_cache", {})

    def _sample_coeff(
        self, coeff, tlist: np.ndarray, args_default: Dict[str, Any]
    ) -> np.ndarray:
        merged = dict(args_default)
        # IMPORTANT: your coeff signature is coeff(t_solver, args) -> complex
        # We sample exactly on the solver tlist.
        out = np.empty(len(tlist), dtype=np.complex128)
        for i, t in enumerate(tlist):
            out[i] = complex(coeff(float(t), merged))
        return out

    def _to_qobj(self, A: np.ndarray, dims: List[int]) -> Qobj:
        A = np.asarray(A, dtype=np.complex128)
        key = (
            A.shape,
            A.strides,
            A.__array_interface__["data"][0],
            tuple(dims),
            self.force_csr,
        )
        q = self._op_cache.get(key)
        if q is not None:
            return q
        q = Qobj(A, dims=[dims, dims])
        if self.force_csr:
            q = q.to("csr")
        self._op_cache[key] = q
        return q

    def _wrap_coeff(self, coeff, args_default: Dict[str, Any]) -> QutipCoeff:
        # Your CoeffExpr is: coeff(t: float, args: Optional[dict]) -> complex
        def f(t: float, args: Dict[str, Any]) -> complex:
            merged = dict(args_default)
            if args:
                merged.update(args)
            return complex(coeff(float(t), merged))

        return f

    def _build_H(self, me: MEProblem):
        dims = list(me.dims)
        tlist = np.asarray(me.tlist, dtype=float)

        # If constant, just sum them (fastest)
        if all(term.coeff is None for term in me.h_terms):
            H0 = 0
            for t in me.h_terms:
                H0 = H0 + self._to_qobj(t.op, dims)
            return H0

        H = []
        for term in me.h_terms:
            op_q = self._to_qobj(term.op, dims)
            if term.coeff is None:
                H.append(op_q)
            else:
                c = self._sample_coeff(term.coeff, tlist, me.args)
                # If it's real, make it float64 (helps QuTiP + speed)
                if np.all(np.isreal(c)):
                    c = np.asarray(c.real, dtype=float)
                H.append([op_q, c])
        return H

    def _build_cops(self, me: MEProblem):
        dims = list(me.dims)
        tlist = np.asarray(me.tlist, dtype=float)
        c_ops = []
        for term in me.c_terms:
            op_q = self._to_qobj(term.op, dims)
            if term.coeff is None:
                c_ops.append(op_q)
            else:
                c = self._sample_coeff(term.coeff, tlist, me.args)
                if np.all(np.isreal(c)):
                    c = np.asarray(c.real, dtype=float)
                c_ops.append([op_q, c])
        return c_ops

    def _build_eops(self, me: "MEProblem") -> Tuple[List[str], List[Qobj]]:
        if me.observables is None:
            return [], []

        dims = list(me.dims)

        e_ops: Dict[str, np.ndarray] = {}
        e_ops.update(me.observables.qd)
        e_ops.update(me.observables.modes)
        if me.observables.extra:
            e_ops.update(me.observables.extra)

        keys = list(e_ops.keys())
        if self.sort_observables:
            keys = sorted(keys)

        ops = [self._to_qobj(e_ops[k], dims) for k in keys]
        return keys, ops

    # --- required protocol method ---
    def simulate(
        self,
        me: MEProblem,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> MESimulationResult:
        dims = list(me.dims)
        rho0_q = self._to_qobj(me.rho0, dims)

        H = self._build_H(me)

        c_ops = self._build_cops(me)

        print("[qutip] num_c_terms =", len(me.c_terms))
        print("[qutip] num_c_ops   =", len(c_ops))
        print("[qutip] first c_op type:", type(c_ops[0]) if c_ops else None)

        keys, e_ops = self._build_eops(me)

        options = dict(options or {})
        options["progress_bar"] = "tqdm"  # or True / "enhanced" / False
        options["store_final_state"] = bool(self.store_final_state)
        # tolerances: relax slightly if you can
        options.setdefault("method", "adams")
        options.setdefault("atol", 1e-8)
        options.setdefault("rtol", 1e-6)

        # limit internal stepping overhead
        # raise if it errors, but don't keep it gigantic
        options.setdefault("nsteps", 5000)
        options.setdefault("max_step", np.inf)  # allow adaptive stepping
        res = mesolve(
            H,
            rho0=rho0_q,
            tlist=np.asarray(me.tlist, dtype=float),
            c_ops=c_ops,
            e_ops=e_ops,
            args=dict(me.args),
            options=options,
        )

        expect: Dict[str, np.ndarray] = {}
        if keys:
            # QuTiP returns a list of arrays aligned to e_ops order
            for k, arr in zip(keys, res.expect):
                expect[k] = np.asarray(arr, dtype=float)

        states: Optional[Tuple[np.ndarray, ...]] = None
        if self.store_states and getattr(res, "states", None) is not None:
            states = tuple(
                np.asarray(st.full(), dtype=np.complex128) for st in res.states
            )

        final_state: Optional[np.ndarray] = None
        fs = getattr(res, "final_state", None)
        if fs is not None:
            final_state = np.asarray(fs.full(), dtype=np.complex128)

        meta = dict(me.meta)
        meta.update(
            {
                "backend": self.backend_name,
                "num_h_terms": len(me.h_terms),
                "num_c_terms": len(me.c_terms),
                "num_observables": len(keys),
                "stored_states": bool(states is not None),
            }
        )

        return MESimulationResult(
            tlist=np.asarray(me.tlist, dtype=float),
            expect=expect,
            states=states,
            final_state=np.asarray(fs.full(), dtype=np.complex128),
            meta=meta,
        )
