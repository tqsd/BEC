from typing import Callable, Optional
from .base import LightModeBase
from photon_weave.state.envelope import Envelope


class LightMode:
    def __init__(
        self,
        wavelength_nm: float,
        label: Optional[str],
    ):
        self.wavelenth_nm = wavelength_nm
        self.label = label
