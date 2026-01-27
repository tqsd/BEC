from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from bec.light.core.polarization import JonesState
from bec.quantum_dot.enums import Transition
from bec.units import QuantityLike, as_quantity  # adjust import path if needed

HVVec = np.ndarray  # shape (2,), complex


def _as_Cm(x: Any) -> QuantityLike:
    """
    Coerce to a dipole moment quantity in C*m.
    - float interpreted as C*m
    """
    return as_quantity(x, "C*m")


def _as_hv_vec(x: Any) -> HVVec:
    """
    Coerce to normalized complex 2-vector in HV basis.
    Accepts: (2,), list/tuple, numpy array, or JonesState-like with as_array().
    """
    if hasattr(x, "as_array"):
        x = x.as_array()
    v = np.asarray(x, dtype=complex).reshape(2)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Dipole polarization vector must be non-zero.")
    return v / n


@dataclass(frozen=True)
class DipoleParams:
    r"""
    Transition dipole moment parameters (unit-aware).

    Supports 3 levels of detail:

    1) Global magnitude default:
       - mu_default : QuantityLike (C·m)
         * float interpreted as C·m

    2) Per-transition scalar magnitudes:
       - mu_by_transition[Transition] : QuantityLike (C·m)

    3) Per-transition HV polarization vectors (dimensionless complex 2-vectors):
       - pol_hv_by_transition[Transition] : (dH, dV)
         Stored normalized; magnitude comes from (1)/(2).

    Interpretation
    --------------
    For directed transition `tr`:

        d_vec(tr) = mu(tr) * e_d(tr)

    where:
      - mu(tr) is scalar magnitude in C·m
      - e_d(tr) is normalized complex Jones-like vector in HV basis.

    Notes
    -----
    - Polarization vectors are dimensionless; only magnitudes carry units.
    - If you want different conventions (e.g. store μ in Debye), just change
      the unit coercion in _as_Cm and/or add alternate constructors.
    """

    # Level 1: simplest
    mu_default: QuantityLike = field(default_factory=lambda: _as_Cm(1e-29))

    # Level 2: per-transition override magnitudes
    mu_by_transition: Dict[Transition, QuantityLike] = field(
        default_factory=dict
    )

    # Level 3: per-transition polarization directions (HV basis)
    pol_hv_by_transition: Dict[Transition, Tuple[complex, complex]] = field(
        default_factory=dict
    )

    # Default polarization when none is specified
    default_pol: JonesState = field(default_factory=JonesState.H)

    # ---------- constructors ----------

    @classmethod
    def from_values(
        cls,
        *,
        mu_default_Cm: Any = 1e-29,
        mu_by_transition_Cm: Mapping[Transition, Any] | None = None,
        pol_hv_by_transition: (
            Mapping[Transition, Tuple[complex, complex]] | None
        ) = None,
        default_pol: JonesState | None = None,
    ) -> "DipoleParams":
        return cls(
            mu_default=_as_Cm(mu_default_Cm),
            mu_by_transition={
                tr: _as_Cm(val)
                for tr, val in (mu_by_transition_Cm or {}).items()
            },
            pol_hv_by_transition=dict(pol_hv_by_transition or {}),
            default_pol=(
                default_pol if default_pol is not None else JonesState.H()
            ),
        )

    @classmethod
    def biexciton_cascade_defaults(
        cls,
        *,
        mu_default_Cm: Any = 1e-29,
        mu_by_transition_Cm: Mapping[Transition, Any] | None = None,
    ) -> "DipoleParams":
        pol = {
            Transition.G_X1: (1.0 + 0j, 0.0 + 0j),  # H
            Transition.G_X2: (0.0 + 0j, 1.0 + 0j),  # V
            Transition.X1_XX: (0.0 + 0j, 1.0 + 0j),  # V
            Transition.X2_XX: (1.0 + 0j, 0.0 + 0j),  # H
        }
        return cls.from_values(
            mu_default_Cm=mu_default_Cm,
            mu_by_transition_Cm=mu_by_transition_Cm,
            pol_hv_by_transition=pol,
            default_pol=JonesState.H(),  # fallback only
        )

    # ---------- core accessors ----------

    def mu(self, tr: Transition) -> QuantityLike:
        """Scalar magnitude μ(tr) in C·m (QuantityLike)."""
        return self.mu_by_transition.get(tr, self.mu_default)

    def mu_Cm(self, tr: Transition) -> float:
        """Scalar magnitude μ(tr) as a float in SI units (C·m)."""
        return float(self.mu(tr).to("C*m").magnitude)

    def e_pol_hv(self, tr: Transition) -> HVVec:
        """
        Normalized polarization direction e_d(tr) in HV basis (complex 2-vector).
        """
        if tr in self.pol_hv_by_transition:
            dH, dV = self.pol_hv_by_transition[tr]
            return _as_hv_vec([dH, dV])
        return _as_hv_vec(self.default_pol.as_array())

    def d_vec_hv(self, tr: Transition) -> HVVec:
        """
        Full dipole vector in HV basis as complex 2-vector in C·m:
            μ(tr) [C·m] * e_d(tr) [1]
        Returned as numpy array (complex) with values in C·m.
        """
        return self.mu_Cm(tr) * self.e_pol_hv(tr)

    # ---------- validation ----------

    def validate(self) -> None:
        # Validate magnitudes
        mu0 = float(self.mu_default.to("C*m").magnitude)
        if mu0 <= 0:
            raise ValueError(f"mu_default must be > 0, got {self.mu_default}")

        for tr, mu in self.mu_by_transition.items():
            val = float(mu.to("C*m").magnitude)
            if val <= 0:
                raise ValueError(
                    f"mu_by_transition[{tr}] must be > 0, got {mu}"
                )

        # Validate pol vectors (non-zero; normalization happens in accessor)
        for tr, (dH, dV) in self.pol_hv_by_transition.items():
            _ = _as_hv_vec([dH, dV])  # will raise if zero

        # Validate default pol is non-zero
        _ = _as_hv_vec(self.default_pol.as_array())

    # ---------- numeric view ----------

    def as_floats(self) -> dict[str, Any]:
        """
        Float-only view for hot numeric code.

        Returns:
          - mu_default_Cm: float
          - mu_by_transition_Cm: dict[Transition, float]
          - pol_hv_by_transition: dict[Transition, (complex, complex)] (already numeric)
        """
        return {
            "mu_default_Cm": float(self.mu_default.to("C*m").magnitude),
            "mu_by_transition_Cm": {
                tr: float(mu.to("C*m").magnitude)
                for tr, mu in self.mu_by_transition.items()
            },
            "pol_hv_by_transition": dict(self.pol_hv_by_transition),
        }
