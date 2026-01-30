from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term
from smef.core.model.protocols import TermCatalogProto
from smef.core.units import magnitude

from bec.quantum_dot.enums import RateKey, Transition
from bec.quantum_dot.smef.modes import QDModeKey, QDModes


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


def _qd_sym(tr: Transition) -> str:
    # qd4_named_ops registers "t_" + Transition.value as an alias
    return "t_" + str(tr.value)


def _qd_op(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _adag_on_mode(mode_index: int) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(mode_index,), locals=(LocalSymbolOp("adag"),))
    )


def _sigma_on_qd(qd_index: int, tr: Transition) -> OpExpr:
    return _qd_op(qd_index, _qd_sym(tr))


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
    r"""
    Build rotated creation operator :math:`a_{\pm}^\dagger(\theta)` as an OpExpr.

    We represent each optical band as two independent polarization subsystems
    (H and V), each with its own creation operator ``adag``. The rotated
    creation operators are defined as the SU(2) rotation:

    .. math::

        a_+^\dagger &= \cos\theta \, a_H^\dagger + e^{i\phi}\sin\theta \, a_V^\dagger \\
        a_-^\dagger &= -e^{-i\phi}\sin\theta \, a_H^\dagger + \cos\theta \, a_V^\dagger

    Parameters
    ----------
    modes:
        The compiled mode layout. Must contain H and V modes for the requested band.
    band:
        Optical band identifier used by :class:`~bec.quantum_dot.smef.modes.QDModeKey`
        (typically ``"XX"`` for XX->X emission and ``"GX"`` for X->G emission).
    sign:
        Either ``"+"`` or ``"-"`` to select which rotated mode is returned.
    theta:
        Polarization rotation angle (radians).
    phi:
        Relative phase between H and V contributions.

    Returns
    -------
    OpExpr
        An IR expression representing :math:`a_\pm^\dagger`.

    Raises
    ------
    ValueError
        If ``sign`` is not ``"+"`` or ``"-"``.
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
    """
    Fetch a rate from a mapping and convert it into solver units.

    The mapping may be keyed by :class:`~bec.quantum_dot.enums.RateKey` or by
    its string value. The stored value is expected to be unitful (1/s), and
    conversion into solver units is delegated to ``units.rate_to_solver``.

    Parameters
    ----------
    rates:
        Mapping of rates, typically ``qd.rates``.
    rk:
        Rate key to fetch.
    units:
        UnitSystem-like object that provides ``rate_to_solver(rate_1_s)``.

    Returns
    -------
    float
        Rate in solver units.

    Raises
    ------
    KeyError
        If the rate is missing.
    """
    r = rates.get(rk, rates.get(rk.value))
    if r is None:
        raise KeyError(rk)
    return float(units.rate_to_solver(r))


def _maybe_get_rate_solver(
    rates: Mapping[Any, Any], rk: RateKey, *, units
) -> Optional[float]:
    """
    Like :func:`_get_rate_value` but returns None if missing and clamps <= 0 to None.
    """
    r = rates.get(rk, rates.get(rk.value))
    if r is None:
        return None
    g = float(units.rate_to_solver(r))
    if g <= 0.0:
        return None
    return g


def _exciton_theta_rad_from_qd(qd) -> float:
    """
    Compute exciton mixing rotation angle from QD parameters.

    Uses the standard 2x2 exciton Hamiltonian in the {X1, X2} basis with
    diagonal splitting given by FSS and real off-diagonal coupling delta_prime.

    .. math::

        \\theta = \\tfrac{1}{2} \\operatorname{atan2}(2\\delta', \\mathrm{FSS})

    Notes
    -----
    - If ``qd.mixing`` is None, returns 0.
    - If ``qd.energy`` does not expose ``fss``, we compute it as ``E_X1 - E_X2``.
    """
    mp = getattr(qd, "mixing", None)
    if mp is None:
        return 0.0

    try:
        fss_eV = float(qd.energy.fss.to("eV").magnitude)
    except Exception:
        fss_eV = float(qd.energy.X1.to("eV").magnitude) - float(
            qd.energy.X2.to("eV").magnitude
        )

    dp = getattr(mp, "delta_prime", 0.0)
    try:
        dp_eV = float(dp.to("eV").magnitude)
    except Exception:
        dp_eV = float(dp)

    return 0.5 * float(np.arctan2(2.0 * dp_eV, fss_eV))


def _qd_local(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _proj(qd_index: int, name: str) -> OpExpr:
    # expected symbols: proj_X1, proj_X2, ...
    return _qd_local(qd_index, "proj_" + name)


def _t(qd_index: int, tr: Transition) -> OpExpr:
    # convention: "t_" + Transition.value is registered by your materializer
    return _qd_local(qd_index, "t_" + str(tr.value))


def _delta_prime_eV(qd) -> float:
    """
    Read exciton mixing parameter delta_prime in eV.

    Contract:
    - QuantumDot stores it as qd.mixing (ExcitonMixingParams) or None.
    - Return 0.0 if not provided.
    """
    mp = getattr(qd, "mixing", None)
    if mp is None:
        return 0.0
    dp = getattr(mp, "delta_prime", 0.0)
    try:
        return float(magnitude(dp, "eV"))
    except Exception:
        return float(dp)
