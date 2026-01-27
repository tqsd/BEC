from typing import Callable, Dict, Optional
from bec.quantum_dot.derived import DerivedQD
from bec.quantum_dot.me.hamiltonian_builder import HamiltonianBuilder
from bec.quantum_dot.me.observables_builder import ObservablesBuilder
from bec.quantum_dot.mode_registry import ModeRegistry, SpectralResolution
from bec.quantum_dot.models.decay_model import DecayModel
from bec.quantum_dot.models.phonon_model import PhononModel
from bec.quantum_dot.parameters.cavity import CavityParams
from bec.quantum_dot.parameters.dipole import DipoleParams
from bec.quantum_dot.parameters.energy_structure import EnergyStructure
from bec.quantum_dot.parameters.exciton_mixing import ExcitonMixingParams
from bec.quantum_dot.parameters.phonons import PhononParams
from bec.quantum_dot.polarization import (
    PolarizationCoupling,
    exciton_rotation_params,
)
from bec.quantum_dot.transitions import (
    DEFAULT_REGISTRY as DEFAULT_TRANSITION_REGISTRY,
)


def default_pm_map(idx: int) -> str:
    return "+" if idx in (0, 2) else "-"


class QuantumDot:
    """
        High-level facade for a four-level quantum-dot (QD) model.

        This class is intentionally *drive-agnostic*. It wires static subsystems:
          - mode registry (intrinsic modes + any externally registered modes)
          - symbolic operator context builder
          - kron embedding helper
    - ME builders (Hamiltonian, Collapse, Observables)
          - physical models (DecayModel, PhononModel)

        Dynamic, drive-dependent compilation into a solver-ready master equation
        should live outside this class (e.g. in a compiler / simulation engine).
    """

    def __init__(
        self,
        *,
        energy_structure: EnergyStructure,
        phonon_params: Optional[PhononParams] = None,
        cavity_params: Optional[CavityParams] = None,
        dipole_params: Optional[DipoleParams] = None,
        exciton_mixing: Optional[ExcitonMixingParams] = None,
        spectral_resolution: Optional[SpectralResolution] = None,
    ):
        self.energy_structure = energy_structure
        self.phonon_params = phonon_params or PhononParams()
        self.cavity_params = cavity_params
        self.dipole_params = dipole_params or DipoleParams.from_values(
            mu_default_Cm=1e-29
        )
        self.exciton_mixing = exciton_mixing or ExcitonMixingParams.from_values(
            delta_prime_eV=0.0
        )

        theta, phi, Omega = exciton_rotation_params(
            fss=self.energy_structure.fss,
            delta_prime=self.exciton_mixing.delta_prime,
        )
        self.polarization = PolarizationCoupling(
            theta=theta, phi=phi, Omega=Omega
        )
        self.decay_model = DecayModel(
            energy_structure=self.energy_structure,
            dipole_params=self.dipole_params,
            cavity_params=self.cavity_params,
        )
        self.phonon_model = PhononModel(phonon_params=self.phonon_params)

        res = spectral_resolution or SpectralResolution.TWO_COLORS
        self.mode_registry = ModeRegistry.from_qd(
            energy_structure=self.energy_structure, resolution=res
        )

        # Builders
        self.hamiltonian_builder = HamiltonianBuilder(
            energy_structure=self.energy_structure,
            exciton_mixing=self.exciton_mixing,
            transitions=DEFAULT_TRANSITION_REGISTRY,
        )
        self.observables_builder = ObservablesBuilder(modes=self.mode_registry)

        self.derived = DerivedQD(self)

    def hamiltonian_catalog(self):
        return self.hamiltonian_builder.build_catalog()

    @property
    def observables_catalog(self):
        return self.observables_builder.build_catalog()
