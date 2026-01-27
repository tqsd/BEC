from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.me.coeffs import CoeffExpr, as_coeff
from bec.quantum_dot.me.types import CollapseTerm, CollapseTermKind
from bec.simulation.protocols import CollapseComposer
from bec.simulation.types import RatesBundle


def _as_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as exc:
        raise TypeError(f"{name} must be float-like, got {type(x)!r}") from exc


def _coeff_const(val: float) -> CoeffExpr:
    v = float(val)

    def f(t: float, args: Any = None, *, _v: float = v) -> float:
        return _v

    return as_coeff(f)


def _coeff_sqrt(gamma: CoeffExpr) -> CoeffExpr:
    def f(t: float, args: Any = None) -> complex:
        z = gamma(float(t), args)
        if isinstance(z, complex):
            if abs(z.imag) > 1e-12 * max(1.0, abs(z.real)):
                raise ValueError(
                    f"Rate coefficient returned complex with imag: {z}"
                )
            z = z.real
        g = float(z)
        if g < 0.0:
            raise ValueError(f"Encountered negative rate in sqrt: {g}")
        return complex(np.sqrt(g))

    return as_coeff(f)


def _coeff_eval_at0(gamma: CoeffExpr) -> float:
    z = gamma(0.0, None)
    if isinstance(z, complex):
        if abs(z.imag) > 1e-12 * max(1.0, abs(z.real)):
            raise ValueError(
                f"Rate coefficient returned complex with imag: {z}"
            )
        z = z.real
    g = float(z)
    if g < 0.0:
        raise ValueError(f"Rate coefficient returned negative value: {g}")
    return g


@dataclass(frozen=True)
class CollapseCompositionPolicy:
    radiative_enabled: bool = True
    coherent_xx_decay: bool = True
    xx_relative_phase_rad: float = 0.0
    phonon_pure_dephasing_enabled: bool = False
    eid_enabled: bool = True


class DefaultCollapseComposer(CollapseComposer):
    def __init__(self, policy: Optional[CollapseCompositionPolicy] = None):
        self.policy = policy or CollapseCompositionPolicy()

    def compose(
        self,
        *,
        qd: QuantumDot,
        dims: List[int],
        rates: RatesBundle,
        time_unit_s: float,
    ) -> List[CollapseTerm]:
        _as_float(time_unit_s, "time_unit_s")

        proto = qd.c_builder.build_catalog(dims=dims)

        radiative = [t for t in proto if t.kind == CollapseTermKind.RADIATIVE]
        phonon = [t for t in proto if t.kind == CollapseTermKind.PHONON]
        eid_proto = [
            t for t in proto if t.kind == CollapseTermKind.PHENOMENOLOGICAL
        ]
        others = [
            t
            for t in proto
            if t.kind
            not in (
                CollapseTermKind.RADIATIVE,
                CollapseTermKind.PHONON,
                CollapseTermKind.PHENOMENOLOGICAL,
            )
        ]

        out: List[CollapseTerm] = []
        if self.policy.radiative_enabled:
            out.extend(self._emit_radiative_from_catalog(radiative, rates))

        if self.policy.phonon_pure_dephasing_enabled:
            out.extend(self._attach_rates_for_catalog_terms(phonon, rates))

        if self.policy.eid_enabled:
            # easiest: have EID prototypes in catalog with rate_key="gamma_eid_X"
            out.extend(self._attach_rates_for_catalog_terms(eid_proto, rates))
        else:
            out.extend(eid_proto)  # or drop them

        out.extend(others)
        return out

    def _attach_rates_for_catalog_terms(
        self, terms: List[CollapseTerm], rates: RatesBundle
    ) -> List[CollapseTerm]:
        out: List[CollapseTerm] = []
        for t in terms:
            rk = (t.meta or {}).get("rate_key")
            if not rk:
                out.append(t)
                continue

            gamma = rates.get(str(rk))
            if gamma is None:
                raise KeyError(
                    f"Missing rate for collapse term {
                        t.label} (rate_key={rk})"
                )

            out.append(
                CollapseTerm(
                    kind=t.kind,
                    op=t.op,
                    coeff=_coeff_sqrt(gamma),
                    label=t.label,
                    meta=dict(t.meta or {}),
                )
            )
        return out

    def _emit_radiative_from_catalog(
        self, radiative: List[CollapseTerm], rates: RatesBundle
    ) -> List[CollapseTerm]:
        if not radiative:
            return []

        groups: Dict[str, List[CollapseTerm]] = {}
        for t in radiative:
            mk = (t.meta or {}).get("merge_key")
            groups.setdefault(
                (
                    str(mk)
                    if mk is not None
                    else f"__solo__:{
                        t.label}"
                ),
                [],
            ).append(t)

        out: List[CollapseTerm] = []
        phi = float(
            rates.meta.get(
                "xx_relative_phase_rad", self.policy.xx_relative_phase_rad
            )
        )

        for mk, terms in groups.items():
            is_xx_group = mk.startswith("XX->X")
            do_coherent = bool(
                is_xx_group
                and self.policy.coherent_xx_decay
                and len(terms) == 2
            )

            if not do_coherent:
                out.extend(self._attach_rates_for_catalog_terms(terms, rates))
                continue

            # deterministic ordering: X1 then X2
            order = {"X1": 0, "X2": 1}
            terms_sorted = sorted(
                terms,
                key=lambda t: order.get(
                    str((t.meta or {}).get("branch", "")), 99
                ),
            )
            t1, t2 = terms_sorted

            rk1 = str((t1.meta or {})["rate_key"])
            rk2 = str((t2.meta or {})["rate_key"])
            g1c = rates.get(rk1)
            g2c = rates.get(rk2)
            if g1c is None or g2c is None:
                out.extend(self._attach_rates_for_catalog_terms(terms, rates))
                continue

            g1 = _coeff_eval_at0(g1c)
            g2 = _coeff_eval_at0(g2c)

            Lop = (np.sqrt(g1) * t1.op) + (
                np.exp(1j * phi) * np.sqrt(g2) * t2.op
            )

            out.append(
                CollapseTerm(
                    kind=CollapseTermKind.RADIATIVE,
                    op=Lop,
                    coeff=_coeff_const(1.0),
                    label=f"L_{mk}_coherent",
                    meta={
                        "type": "radiative",
                        "mode": "coherent",
                        "merge_key": mk,
                        "phi_rad": float(phi),
                        rk1: float(g1),
                        rk2: float(g2),
                    },
                )
            )

        return out
