from dataclasses import dataclass
from qutip import Qobj
from .builders import PrepareFromVacuum, PrepareFromScalar
from .general import GeneralKrausChannel


@dataclass
class ChannelFactory:
    """
    Given a simulated photonic density matrix, produce a desired channel type.
    """

    def from_photonic_state_prepare_from_vacuum(
        self, rho_phot: Qobj
    ) -> GeneralKrausChannel:
        return PrepareFromVacuum(rho_phot).build()

    def from_photonic_state_prepare_from_scalar(
        self, rho_phot: Qobj
    ) -> GeneralKrausChannel:
        return PrepareFromScalar(rho_phot).build()
