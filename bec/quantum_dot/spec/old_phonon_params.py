from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from smef.core.units import QuantityLike

from bec.core.units import as_dimless, as_K, as_rad_s, as_rate_1_s, as_s2


class PhononModelType(str, Enum):
    PHENOMENOLOGICAL = "phenomenological"
    POLARON = "polaron"


@dataclass(frozen=True)
class PhenomenologicalPhononParams:
    gamma_phi_Xp: QuantityLike = as_rate_1_s(0.0)
    gamma_phi_Xm: QuantityLike = as_rate_1_s(0.0)
    gamma_phi_XX: QuantityLike = as_rate_1_s(0.0)
    gamma_relax_X1_X2: QuantityLike = as_rate_1_s(0.0)
    gamma_relax_X2_X1: QuantityLike = as_rate_1_s(0.0)
    gamma_phi_eid_scale: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma_phi_Xp", as_rate_1_s(self.gamma_phi_Xp))
        object.__setattr__(self, "gamma_phi_Xm", as_rate_1_s(self.gamma_phi_Xm))
        object.__setattr__(self, "gamma_phi_XX", as_rate_1_s(self.gamma_phi_XX))
        object.__setattr__(
            self, "gamma_phi_eid_scale", as_dimless(self.gamma_phi_eid_scale)
        )
        self.validate()

    @property
    def gamma_relax_X1_X2_1_s(self) -> float:
        return float(self.gamma_relax_X1_X2.to("1/s").magnitude)

    @property
    def gamma_relax_X2_X1_1_s(self) -> float:
        return float(self.gamma_relax_X2_X1.to("1/s").magnitude)

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
        for name, val in [
            ("gamma_phi_Xp", self.gamma_phi_Xp_1_s),
            ("gamma_phi_Xm", self.gamma_phi_Xm_1_s),
            ("gamma_phi_XX", self.gamma_phi_XX_1_s),
            ("gamma_phi_eid_scale", float(self.gamma_phi_eid_scale)),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def as_floats(self) -> dict[str, float]:
        return {
            "gamma_phi_Xp_1_s": self.gamma_phi_Xp_1_s,
            "gamma_phi_Xm_1_s": self.gamma_phi_Xm_1_s,
            "gamma_phi_XX_1_s": self.gamma_phi_XX_1_s,
            "gamma_phi_eid_scale": float(self.gamma_phi_eid_scale),
        }


@dataclass(frozen=True)
class PolaronPhononParams:
    enable_polaron_renorm: bool = True
    alpha: QuantityLike = as_s2(0.0)
    omega_c: QuantityLike = as_rad_s(1.0e12)
    enable_eid: bool = False
    enable_exciton_relaxation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", as_s2(self.alpha))
        object.__setattr__(self, "omega_c", as_rad_s(self.omega_c))
        self.validate()

    @property
    def alpha_s2(self) -> float:
        return float(self.alpha.to("s**2").magnitude)

    @property
    def omega_c_rad_s(self) -> float:
        return float(self.omega_c.to("rad/s").magnitude)

    def validate(self) -> None:
        if self.alpha_s2 < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.omega_c_rad_s < 0.0:
            raise ValueError("omega_c must be non-negative")
        if self.enable_polaron_renorm and self.omega_c_rad_s == 0.0:
            raise ValueError(
                "omega_c must be > 0 when polaron renorm is enabled"
            )

    def as_floats(self) -> dict[str, float]:
        return {
            "enable_polaron_renorm": float(self.enable_polaron_renorm),
            "alpha_s2": self.alpha_s2,
            "omega_c_rad_s": self.omega_c_rad_s,
            "enable_exciton_relaxation": float(self.enable_exciton_relaxation),
        }


@dataclass(frozen=True)
class PhononParams:
    """
    Parameters describing phonon-induced decoherence / dressing (unit-aware).

    This is the public object you pass around. Internally it holds two
    sub-blocks (phenomenological and polaron) so extending either one later
    is painless.
    """

    model: PhononModelType = PhononModelType.POLARON
    temperature: QuantityLike = as_K(4.0)

    # state-dependent displacement (used by polaron dressing/scattering)
    phi_G: float = 0.0
    phi_X1: float = 1.0
    phi_X2: float = 1.0
    phi_XX: float = 2.0

    phenomenological: PhenomenologicalPhononParams = (
        PhenomenologicalPhononParams()
    )
    polaron: PolaronPhononParams = PolaronPhononParams()

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", as_K(self.temperature))
        object.__setattr__(self, "phi_G", as_dimless(self.phi_G))
        object.__setattr__(self, "phi_X", as_dimless(self.phi_X))
        object.__setattr__(self, "phi_XX", as_dimless(self.phi_XX))
        self.validate()

    @property
    def temperature_K(self) -> float:
        return float(self.temperature.to("K").magnitude)

    def validate(self) -> None:
        if self.temperature_K < 0.0:
            raise ValueError("temperature must be non-negative")
        for name, val in [
            ("phi_G", self.phi_G),
            ("phi_X1", self.phi_X1),
            ("phi_X2", self.phi_X2),
            ("phi_XX", self.phi_XX),
        ]:
            if float(val) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        # sub-blocks validate themselves in __post_init__
        if (
            self.model == PhononModelType.POLARON
            and self.polaron.enable_polaron_renorm
        ):
            # allow alpha=0 (no-op), but omega_c must be >0 for renorm
            if self.polaron.omega_c_rad_s == 0.0:
                raise ValueError(
                    "omega_c must be > 0 when polaron renorm is enabled"
                )

    def as_floats(self) -> dict[str, float]:
        out: dict[str, float] = {
            "temperature_K": self.temperature_K,
            "phi_G": float(self.phi_G),
            "phi_X": float(self.phi_X),
            "phi_XX": float(self.phi_XX),
        }
        out.update(self.phenomenological.as_floats())
        out.update(self.polaron.as_floats())
        # model is not a float; keep it out of this numeric dict
        return out
