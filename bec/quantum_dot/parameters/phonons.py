from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from bec.units import QuantityLike
from bec.quantum_dot.units import as_quantity


# ---- unit coercers ----


def _as_K(x: Any) -> QuantityLike:
    return as_quantity(x, "K")


def _as_rate_1_s(x: Any) -> QuantityLike:
    return as_quantity(x, "1/s")


def _as_s2(x: Any) -> QuantityLike:
    return as_quantity(x, "s**2")


def _as_rad_s(x: Any) -> QuantityLike:
    # Pint typically understands "rad" as dimensionless; rad/s is usually fine.
    # If your registry doesn't have rad, use "1/s".
    return as_quantity(x, "rad/s")


def _as_dimless(x: Any) -> float:
    return float(x)


class PhononModelType(str, Enum):
    """
    Selection of phonon modeling strategy.

    - PHENOMENOLOGICAL:
      Add user-specified GKSL channels (e.g. pure dephasing)
      with constant rates. Fast, flexible, but not predictive.

    - POLARON:
      Use LA-phonon spectral density to compute the polaron dressing
      factor B(T) (and optionally scattering terms in later extensions).
      Predictive in the driven regime and consistent with polaron-frame
      master-equation treatments.
    """

    PHENOMENOLOGICAL = "phenomenological"
    POLARON = "polaron"


@dataclass(frozen=True)
class PhononParams:
    r"""
    Parameters describing phonon-induced decoherence / dressing in the QD model (unit-aware).

    Units
    -----
    - temperature: K
    - dephasing rates: 1/s
    - alpha: s^2  (for J(ω)=α ω^3 exp[-(ω/ωc)^2] with ω in rad/s)
    - omega_c: rad/s (often effectively 1/s in Pint because rad is dimensionless)

    Safety defaults
    --------------
    Default model is POLARON, but with alpha=0 the dressing factor is <B>=1
    so the polaron model is effectively disabled.
    """

    # Strategy
    model: PhononModelType = PhononModelType.POLARON

    # Shared
    temperature: QuantityLike = 4.0  # float interpreted as K

    # --- Phenomenological GKSL channels (constant rates) ---
    gamma_phi_Xp: QuantityLike = 0.0  # float interpreted as 1/s
    gamma_phi_Xm: QuantityLike = 0.0
    gamma_phi_XX: QuantityLike = 0.0
    gamma_phi_eid_scale: float = 0.0  # dimensionless scaling knob

    # --- Polaron model (LA phonons, super-Ohmic) ---
    enable_polaron_renorm: bool = True
    alpha: QuantityLike = 0.0  # float interpreted as s^2
    omega_c: QuantityLike = 1.0e12  # float interpreted as rad/s

    # --- Polaron model structure (state-dependent displacement) ---
    phi_G: float = 0.0
    phi_X: float = 1.0  # applies to X1 and X2
    phi_XX: float = 2.0

    # --- Optional (future / advanced) ---
    enable_exciton_relaxation: bool = False

    def __post_init__(self) -> None:
        # coerce units (dataclass is frozen → use object.__setattr__)
        object.__setattr__(self, "temperature", _as_K(self.temperature))

        object.__setattr__(
            self, "gamma_phi_Xp", _as_rate_1_s(self.gamma_phi_Xp)
        )
        object.__setattr__(
            self, "gamma_phi_Xm", _as_rate_1_s(self.gamma_phi_Xm)
        )
        object.__setattr__(
            self, "gamma_phi_XX", _as_rate_1_s(self.gamma_phi_XX)
        )

        object.__setattr__(self, "alpha", _as_s2(self.alpha))
        object.__setattr__(self, "omega_c", _as_rad_s(self.omega_c))

        object.__setattr__(
            self, "gamma_phi_eid_scale", _as_dimless(self.gamma_phi_eid_scale)
        )
        object.__setattr__(self, "phi_G", _as_dimless(self.phi_G))
        object.__setattr__(self, "phi_X", _as_dimless(self.phi_X))
        object.__setattr__(self, "phi_XX", _as_dimless(self.phi_XX))

        self.validate()

    # ---------- normalized / derived ----------

    @property
    def temperature_K(self) -> float:
        return float(self.temperature.to("K").magnitude)

    @property
    def omega_c_rad_s(self) -> float:
        # If your registry treats rad as dimensionless, this is effectively 1/s anyway.
        return float(self.omega_c.to("rad/s").magnitude)

    @property
    def alpha_s2(self) -> float:
        return float(self.alpha.to("s**2").magnitude)

    @property
    def gamma_phi_Xp_1_s(self) -> float:
        return float(self.gamma_phi_Xp.to("1/s").magnitude)

    @property
    def gamma_phi_Xm_1_s(self) -> float:
        return float(self.gamma_phi_Xm.to("1/s").magnitude)

    @property
    def gamma_phi_XX_1_s(self) -> float:
        return float(self.gamma_phi_XX.to("1/s").magnitude)

    def validate(self) -> None:
        # Non-negativity checks
        if self.temperature_K < 0.0:
            raise ValueError("temperature must be non-negative")
        for name, val in [
            ("gamma_phi_Xp", self.gamma_phi_Xp_1_s),
            ("gamma_phi_Xm", self.gamma_phi_Xm_1_s),
            ("gamma_phi_XX", self.gamma_phi_XX_1_s),
            ("gamma_phi_eid_scale", self.gamma_phi_eid_scale),
            ("alpha", self.alpha_s2),
            ("omega_c", self.omega_c_rad_s),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if self.model == PhononModelType.POLARON and self.enable_polaron_renorm:
            # With alpha=0 the renorm is a no-op; allowed.
            if self.omega_c_rad_s == 0.0:
                raise ValueError(
                    "omega_c must be > 0 when polaron renorm is enabled"
                )

    # ---------- numeric view ----------

    def as_floats(self) -> dict[str, float]:
        """
        Float-only view (SI-like) for hot numeric code.

        Notes:
        - omega_c is returned in rad/s (often equivalent dimensionally to 1/s in Pint).
        - alpha in s^2.
        """
        return {
            "temperature_K": self.temperature_K,
            "gamma_phi_Xp_1_s": self.gamma_phi_Xp_1_s,
            "gamma_phi_Xm_1_s": self.gamma_phi_Xm_1_s,
            "gamma_phi_XX_1_s": self.gamma_phi_XX_1_s,
            "gamma_phi_eid_scale": float(self.gamma_phi_eid_scale),
            "enable_polaron_renorm": float(self.enable_polaron_renorm),
            "alpha_s2": self.alpha_s2,
            "omega_c_rad_s": self.omega_c_rad_s,
            "phi_G": float(self.phi_G),
            "phi_X": float(self.phi_X),
            "phi_XX": float(self.phi_XX),
            "enable_exciton_relaxation": float(self.enable_exciton_relaxation),
        }
