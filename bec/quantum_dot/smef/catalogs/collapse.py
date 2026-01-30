from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind
from smef.core.model.protocols import TermCatalogProto

from bec.quantum_dot.enums import RateKey, Transition
from bec.quantum_dot.smef.catalogs.base import FrozenCatalog
from bec.quantum_dot.smef.modes import QDModes, QDModeKey
from .base import (
    _get_rate_value,
    _rotated_adag,
    _sigma_on_qd,
    _maybe_get_rate_solver,
    _exciton_theta_rad_from_qd,
)


@dataclass(frozen=True)
class QDCollapseCatalog(FrozenCatalog):
    r"""
    Collapse-term catalog for a four-level biexciton cascade with explicit field modes.

    This catalog constructs SMEF IR collapse terms from the unitful rates exposed by
    ``qd.rates``. The only unit boundary is the conversion ``units.rate_to_solver(rate)``,
    after which the catalog uses float solver-units.

    Radiative emission (two-channel form)
    ------------------------------------
    The biexciton cascade is represented using two Lindblad operators that each
    coherently sum over the fine-structure branches:

    .. math::

        L_{XX} &= \sqrt{\gamma_{XX\to X1}} \, \sigma_{XX\to X1} \,
                 a_{XX,s_1}^\dagger(\theta,\phi)
              +  \sqrt{\gamma_{XX\to X2}} \, \sigma_{XX\to X2} \,
                 a_{XX,s_2}^\dagger(\theta,\phi) \\

        L_{GX} &= \sqrt{\gamma_{X1\to G}} \, \sigma_{X1\to G} \,
                 a_{GX,s_3}^\dagger(\theta,\phi)
              +  \sqrt{\gamma_{X2\to G}} \, \sigma_{X2\to G} \,
                 a_{GX,s_4}^\dagger(\theta,\phi)

    where :math:`\sigma_{src\to dst}` is the QD transition operator and
    :math:`a^\dagger` creates a photon in the corresponding polarization mode.

    Polarization rotation
    ---------------------
    Each optical band (``"XX"`` and ``"GX"``) is modeled as two independent
    polarization subsystems (H and V). The rotated creation operators are defined as:

    .. math::

        a_+^\dagger &= \cos\theta \, a_H^\dagger + e^{i\phi}\sin\theta \, a_V^\dagger \\
        a_-^\dagger &= -e^{-i\phi}\sin\theta \, a_H^\dagger + \cos\theta \, a_V^\dagger

    The branch-to-sign assignment is controlled by ``xx_branch_signs`` and
    ``gx_branch_signs``. This matches the common “paper-style” mapping where
    the two branches emit into orthogonal polarizations.

    Choosing theta
    --------------
    If ``theta`` is not provided explicitly, it is derived from QD parameters
    (fine-structure splitting and exciton mixing) via:

    .. math::

        \theta = \tfrac{1}{2}\operatorname{atan2}(2\delta', \mathrm{FSS})

    using ``qd.energy`` and ``qd.mixing``.

    Phenomenological phonon terms
    -----------------------------
    If present in ``qd.rates``, additional constant Lindblad operators are emitted:

    - pure dephasing (projector form):

      .. math::

        L = \sqrt{\gamma_\phi}\, P_{state}

      for X1, X2, and XX.

    - exciton relaxation:

      .. math::

        L = \sqrt{\gamma}\, |dst\rangle\langle src|

      implemented using the registered transition symbols ``t_<Transition.value>``.

    Notes
    -----
    - This catalog only builds IR. All physics (radiative and phonon rates)
      must be computed by unitful models (DecayModel, PhononModel) and exposed
      through ``qd.rates``.
    - The materializer must register:
      - QD local symbols: ``t_<Transition.value>``, ``proj_X1``, ``proj_X2``, ``proj_XX``
      - Mode local symbol: ``adag`` on each optical mode subsystem.
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
        # Default mapping: XX_X1 uses "-", XX_X2 uses "+"; X1_G uses "-", X2_G uses "+".
        xx_branch_signs: Tuple[str, str] = ("-", "+"),  # (XX_X1, XX_X2)
        gx_branch_signs: Tuple[str, str] = ("-", "+"),  # (X1_G,  X2_G)
    ) -> TermCatalogProto:
        if units is None:
            raise ValueError("units must be provided")
        if modes is None:
            raise ValueError("modes must be provided")

        rates: Mapping[Any, Any] = getattr(qd, "rates", None) or {}
        if not rates:
            raise AttributeError("No rates found in qd.rates")

        th = (
            float(theta)
            if theta is not None
            else _exciton_theta_rad_from_qd(qd)
        )
        qd_i = int(modes.index_of(QDModeKey.qd()))

        # ---- radiative emission (required) ----
        g_xx_x1 = _get_rate_value(rates, RateKey.RAD_XX_X1, units=units)
        g_xx_x2 = _get_rate_value(rates, RateKey.RAD_XX_X2, units=units)
        g_x1_g = _get_rate_value(rates, RateKey.RAD_X1_G, units=units)
        g_x2_g = _get_rate_value(rates, RateKey.RAD_X2_G, units=units)

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
                    "theta": float(th),
                    "phi": float(phi),
                    "style": "rotated_2ch",
                    "xx_branch_signs": tuple(xx_branch_signs),
                },
            ),
            Term(
                kind=TermKind.C,
                op=L_gx,
                coeff=None,
                label="L_GX",
                meta={
                    "band": "GX",
                    "theta": float(th),
                    "phi": float(phi),
                    "style": "rotated_2ch",
                    "gx_branch_signs": tuple(gx_branch_signs),
                },
            ),
        ]

        # ---- phonon-induced pure dephasing (optional) ----
        # L = sqrt(gamma_phi) * P_state
        g = _maybe_get_rate_solver(rates, RateKey.PH_DEPH_X1, units=units)
        if g is not None:
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

        g = _maybe_get_rate_solver(rates, RateKey.PH_DEPH_X2, units=units)
        if g is not None:
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

        g = _maybe_get_rate_solver(rates, RateKey.PH_DEPH_XX, units=units)
        if g is not None:
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

        # ---- phonon-induced exciton relaxation (optional) ----
        # L = sqrt(gamma) * |dst><src| == sqrt(gamma) * t_src_dst

        g = _maybe_get_rate_solver(rates, RateKey.PH_RELAX_X1_X2, units=units)
        if g is not None:
            # Transition.X1_X2 corresponds to |X2><X1| in your convention.
            op = _sigma_on_qd(qd_i, Transition.X1_X2)
            terms.append(
                Term(
                    kind=TermKind.C,
                    op=OpExpr.scale(complex(np.sqrt(g)), op),
                    coeff=None,
                    label="L_ph_relax_X1_X2",
                    meta={"kind": "phonon_relax", "tr": "X1_X2"},
                )
            )

        g = _maybe_get_rate_solver(rates, RateKey.PH_RELAX_X2_X1, units=units)
        if g is not None:
            # Transition.X2_X1 corresponds to |X1><X2| in your convention.
            op = _sigma_on_qd(qd_i, Transition.X2_X1)
            terms.append(
                Term(
                    kind=TermKind.C,
                    op=OpExpr.scale(complex(np.sqrt(g)), op),
                    coeff=None,
                    label="L_ph_relax_X2_X1",
                    meta={"kind": "phonon_relax", "tr": "X2_X1"},
                )
            )

        return cls(_terms=tuple(terms))
