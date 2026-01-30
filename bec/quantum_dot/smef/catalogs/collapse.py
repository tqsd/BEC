from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind
from smef.core.model.protocols import TermCatalogProto

from bec.quantum_dot.enums import RateKey, Transition, TransitionPair
from bec.quantum_dot.transitions import RAD_RATE_TO_TRANSITION
from bec.quantum_dot.smef.catalogs.base import FrozenCatalog
from bec.quantum_dot.smef.modes import QDModes, QDModeKey


def _qd_sym(tr: Transition) -> str:
    return "t_" + str(tr.value)


def _qd_op(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _band_of(tr: Transition) -> str:
    s = str(tr.value)
    if s.startswith("XX_"):
        return "XX"
    if (s.startswith("X1_") or s.startswith("X2_")) and s.endswith("_G"):
        return "GX"
    raise KeyError(tr)


def _adag_on_mode(mode_index: int) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(mode_index,), locals=(LocalSymbolOp("adag"),))
    )


def _sigma_on_qd(qd_index: int, tr: Transition) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(_qd_sym(tr)),))
    )


def _exp_i(phi: float) -> complex:
    return complex(float(np.cos(phi)), float(np.sin(phi)))


def _rotated_adag(
    modes: QDModes,
    *,
    band: str,
    sign: str,
    theta: float,
    phi: float = 0.0,
) -> OpExpr:
    """
    Build rotated creation operator a_sign^\dagger(theta) as an OpExpr sum of
    H/V creation operators on the two separate subsystems.

    Convention (edit here to match your paper/interpreter exactly):

      a_plus^dag  = cos(theta) * a_H^dag + exp(+i phi) * sin(theta) * a_V^dag
      a_minus^dag = -exp(-i phi) * sin(theta) * a_H^dag + cos(theta) * a_V^dag

    This is a unitary SU(2) rotation on the (H,V) polarization modes.
    """
    iH = modes.index_of(QDModeKey(kind="mode", band=band, pol="H"))
    iV = modes.index_of(QDModeKey(kind="mode", band=band, pol="V"))

    adagH = _adag_on_mode(iH)
    adagV = _adag_on_mode(iV)

    c = float(np.cos(float(theta)))
    s = float(np.sin(float(theta)))
    eip = _exp_i(float(phi))
    eimp = np.conjugate(eip)

    if sign == "+":
        return OpExpr.summation(
            [OpExpr.scale(c, adagH), OpExpr.scale(eip * s, adagV)]
        )
    if sign == "-":
        return OpExpr.summation(
            [OpExpr.scale(-eimp * s, adagH), OpExpr.scale(c, adagV)]
        )
    raise ValueError("sign must be '+' or '-'.")


def _get_rate_value(rates: Mapping[Any, Any], rk: RateKey, *, units) -> float:
    r = rates.get(rk, rates.get(rk.value))
    if r is None:
        raise KeyError(rk)
    return float(units.rate_to_solver(r))


@dataclass(frozen=True)
class QDCollapseCatalog(FrozenCatalog):
    """
    Two-channel collapse catalog (paper-style):

      L_XX : XX -> X emission channel (sum of X1 and X2 branches)
      L_GX : X  -> G emission channel (sum of X1 and X2 branches)

    Each branch couples to a rotated polarization creation operator (+/-),
    implemented as a superposition of H/V mode creation operators.
    """

    @classmethod
    def from_qd(
        cls,
        qd,
        *,
        modes: QDModes,
        units,
        theta: Optional[float] = None,
        phi: float = 0.0,
        # Which branch uses which rotated polarization.
        # This matches your old code: XX_X1 uses "-", XX_X2 uses "+" etc.
        xx_branch_signs: Tuple[str, str] = ("-", "+"),  # (XX_X1, XX_X2)
        gx_branch_signs: Tuple[str, str] = ("-", "+"),  # (X1_G,  X2_G)
    ) -> TermCatalogProto:
        derived = qd.derived

        rates: Mapping[Any, Any] = getattr(derived, "rates", None) or {}
        if not rates:
            raise AttributeError("No rates found in derived.rates")

        th = (
            float(theta)
            if theta is not None
            else float(getattr(derived, "exciton_theta_rad", 0.0))
        )

        qd_i = modes.index_of(QDModeKey.qd())

        # Identify the four radiative transitions we care about
        # (we intentionally do not create one Lindblad operator per transition)
        g_xx_x1 = _get_rate_value(rates, RateKey.RAD_XX_X1, units=units)
        g_xx_x2 = _get_rate_value(rates, RateKey.RAD_XX_X2, units=units)
        g_x1_g = _get_rate_value(rates, RateKey.RAD_X1_G, units=units)
        g_x2_g = _get_rate_value(rates, RateKey.RAD_X2_G, units=units)

        # Build L_XX = sqrt(g1) sigma_XX_X1 adag_-(theta) + sqrt(g2) sigma_XX_X2 adag_+(theta)
        sig_xx_x1 = _sigma_on_qd(qd_i, Transition.XX_X1)
        sig_xx_x2 = _sigma_on_qd(qd_i, Transition.XX_X2)

        adag_xx_1 = _rotated_adag(
            modes, band="XX", sign=xx_branch_signs[0], theta=th, phi=phi
        )
        adag_xx_2 = _rotated_adag(
            modes, band="XX", sign=xx_branch_signs[1], theta=th, phi=phi
        )

        L_xx = OpExpr.summation(
            [
                OpExpr.scale(
                    complex(np.sqrt(max(g_xx_x1, 0.0))),
                    OpExpr.product([sig_xx_x1, adag_xx_1]),
                ),
                OpExpr.scale(
                    complex(np.sqrt(max(g_xx_x2, 0.0))),
                    OpExpr.product([sig_xx_x2, adag_xx_2]),
                ),
            ]
        )

        # Build L_GX = sqrt(g1) sigma_X1_G adag_-(theta) + sqrt(g2) sigma_X2_G adag_+(theta)
        sig_x1_g = _sigma_on_qd(qd_i, Transition.X1_G)
        sig_x2_g = _sigma_on_qd(qd_i, Transition.X2_G)

        adag_gx_1 = _rotated_adag(
            modes, band="GX", sign=gx_branch_signs[0], theta=th, phi=phi
        )
        adag_gx_2 = _rotated_adag(
            modes, band="GX", sign=gx_branch_signs[1], theta=th, phi=phi
        )

        L_gx = OpExpr.summation(
            [
                OpExpr.scale(
                    complex(np.sqrt(max(g_x1_g, 0.0))),
                    OpExpr.product([sig_x1_g, adag_gx_1]),
                ),
                OpExpr.scale(
                    complex(np.sqrt(max(g_x2_g, 0.0))),
                    OpExpr.product([sig_x2_g, adag_gx_2]),
                ),
            ]
        )

        terms: list[Term] = [
            Term(
                kind=TermKind.C,
                op=L_xx,
                coeff=None,
                label="L_XX",
                meta={
                    "band": "XX",
                    "theta": th,
                    "phi": float(phi),
                    "style": "rotated_2ch",
                },
            ),
            Term(
                kind=TermKind.C,
                op=L_gx,
                coeff=None,
                label="L_GX",
                meta={
                    "band": "GX",
                    "theta": th,
                    "phi": float(phi),
                    "style": "rotated_2ch",
                },
            ),
        ]

        # ---- phonon-induced pure dephasing (phenomenological) ----
        # L = sqrt(gamma_phi) * P_state

        def _maybe_rate_key(k):
            return rates.get(k, rates.get(getattr(k, "value", str(k))))

        # X1 dephasing
        r = _maybe_rate_key(RateKey.PH_DEPH_X1)
        if r is not None:
            g = float(units.rate_to_solver(r))
            if g > 0.0:
                P = _qd_op(qd_i, "proj_X1")
                terms.append(
                    Term(
                        kind=TermKind.C,
                        op=OpExpr.scale(complex(np.sqrt(g)), P),
                        coeff=None,
                        label="L_ph_deph_X1",
                        meta={"kind": "phonon_deph", "state": "X1"},
                    )
                )

        # X2 dephasing
        r = _maybe_rate_key(RateKey.PH_DEPH_X2)
        if r is not None:
            g = float(units.rate_to_solver(r))
            if g > 0.0:
                P = _qd_op(qd_i, "proj_X2")
                terms.append(
                    Term(
                        kind=TermKind.C,
                        op=OpExpr.scale(complex(np.sqrt(g)), P),
                        coeff=None,
                        label="L_ph_deph_X2",
                        meta={"kind": "phonon_deph", "state": "X2"},
                    )
                )

        # XX dephasing
        r = _maybe_rate_key(RateKey.PH_DEPH_XX)
        if r is not None:
            g = float(units.rate_to_solver(r))
            if g > 0.0:
                P = _qd_op(qd_i, "proj_XX")
                terms.append(
                    Term(
                        kind=TermKind.C,
                        op=OpExpr.scale(complex(np.sqrt(g)), P),
                        coeff=None,
                        label="L_ph_deph_XX",
                        meta={"kind": "phonon_deph", "state": "XX"},
                    )
                )

        return cls(_terms=tuple(terms))
