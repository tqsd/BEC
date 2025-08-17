from typing import Literal, Optional, Set
import uuid

from bec.light.envelopes import GaussianEnvelope


class LightMode:
    def __init__(
        self,
        wavelength_nm: float,
        source: Literal["internal", "external"],
        transitions: list[int],
        gaussian: GaussianEnvelope,
        label: Optional[str] = None,
        *,
        role: Literal["single", "tpe"] = "single",
        tpe_eliminated: Optional[Set[str]] = None,  # {"X1","X2"} subset
        tpe_alpha_X1: float = 0.0,
        tpe_alpha_X2: float = 0.0,
    ):
        self.wavelength_nm = wavelength_nm
        self.label = label
        self.source = source
        self.transitions = transitions
        self.containerHV = None
        self.gaussian = gaussian
        self.role: Literal["single", "tpe"] = (
            "tpe"
            if (len(transitions) == 1 and transitions[0] == 4)
            else "single"
        )
        self.tpe_eliminated = tpe_eliminated or set()
        self.tpe_alpha_X1 = tpe_alpha_X1
        self.tpe_alpha_X2 = tpe_alpha_X2
        self.__id = uuid.uuid4()

    def __eq__(self, other):
        if not isinstance(other, LightMode):
            return NotImplemented
        return (self.wavelength_nm, self.source, self.label, self.__id) == (
            other.wavelength_nm,
            other.source,
            other.label,
            other.__id,
        )

    def __hash__(self):
        return hash((self.wavelength_nm, self.source, self.label, self.__id))

    def __repr__(self):
        return (
            f"LightMode(wavelength_nm={self.wavelength_nm}, "
            f"source={self.source!r}, label={self.label!r})"
        )
