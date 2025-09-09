from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from bec.light import LightMode
from bec.params.transitions import Transition, TransitionType


@dataclass
class EnergyLevels:
    """
    Quantum-dot energy level specification with derived transitions.

    Parameters
    ----------
    biexciton : float
        Energy of the |XX⟩ level (eV).
    exciton : float
        Mean energy of the exciton manifold (midpoint of X1/X2) in eV.
    fss : float
        Fine-structure splitting Δ between X1 and X2 (eV). X1 = exciton + Δ/2,
        X2 = exciton − Δ/2.
    delta_prime : float
        Off-diagonal coupling Δ' between the exciton sublevels (eV), e.g. due
        to anisotropy; appears as the sigma_x term in the exciton Hamiltonian.

    Notes
    -----
    - G is fixed at 0 by convention.
    - XX is taken to be `biexciton` (so transition energies from X to XX are
      `XX - Xi`).
    - Transition tuples are `(energy_eV, Transition)` for convenience.
    """

    biexciton: float
    exciton: float
    fss: float
    delta_prime: float
    # fixed or derived values
    G: float = field(init=False, default=0.0)
    X1: float = field(init=False)
    X2: float = field(init=False)
    XX: float = field(init=False)
    e_G_X1: Tuple[float, Transition, str] = field(init=False)
    e_G_X2: Tuple[float, Transition, str] = field(init=False)
    e_X1_XX: Tuple[float, Transition, str] = field(init=False)
    e_X2_XX: Tuple[float, Transition, str] = field(init=False)

    def __post_init__(self) -> None:
        """
        Compute derived level positions and transition energies.

        Sets:
            X1, X2: split exciton energies from mean `exciton` and `fss`.
            XX: equals `biexciton`.
            e_*: transition energy tuples (energy_eV, Transition).
            binding_energy: (X1 + X2) - XX.
        """
        self.X1 = self.exciton + self.fss / 2
        self.X2 = self.exciton - self.fss / 2
        self.XX = self.biexciton
        self.binding_energy = (self.X1 + self.X2) - self.biexciton
        self.e_G_X1 = (self.X1, Transition.G_X1, "G_X1")
        self.e_G_X2 = (self.X2, Transition.G_X2, "G_X2")
        self.e_X1_XX = (self.XX - self.X1, Transition.X1_XX, "X1_XX")
        self.e_X2_XX = (self.XX - self.X2, Transition.X2_XX, "X2_XX")

    def compute_modes(self) -> List[LightMode]:
        """
        Build `LightMode` objects for each allowed radiative transition.

        Returns
        -------
        List[LightMode]
            Modes for G<-->X1, G<-->X2, X1<-->XX, X2<-->XX with `energy_ev`
            filled and `source=TransitionType.INTERNAL`.
        """
        modes: List[LightMode] = []
        for t in [self.e_G_X1, self.e_G_X2, self.e_X1_XX, self.e_X2_XX]:
            modes.append(
                LightMode(
                    energy_ev=t[0],
                    source=TransitionType.INTERNAL,
                    transition=t[1],
                    label=t[2],
                )
            )
        return modes

    def exciton_rotation_params(self):
        """
        Diagonalize the exciton subspace Hamiltonian and return rotation
        parameters.

        The exciton subspace (|X1⟩, |X2⟩) is modeled by the effective
        Hamiltonian (in the {|X1⟩, |X2⟩} basis)
            H = 1/2 * [[ Δ,  Δ' ],
                       [ Δ', -Δ ]]
        where Δ = fss and Δ' = delta_prime.

        Diagonalizing H yields eigenvectors that define the rotation from the
        { |X1⟩, |X2⟩ } basis to the eigenbasis {|X+⟩, |X-⟩}. We parameterize
        the first eigenvector v = (v0, v1)^T by a mixing angle θ and a relative
        phase φ:
            θ = arctan(|v1| / |v0|),   φ = arg(v1) - arg(v0).

        Returns
        -------
        theta : float
            Mixing angle θ ∈ [0, π/2] between |X1⟩ and |X2⟩ in the first
            eigenstate.
        phi : float
            Relative phase φ between coefficients of |X2⟩ and |X1⟩ in the first
            eigenstate.
        eigvals : np.ndarray
            Eigenvalues (energies) of H in ascending order.

        Notes
        -----
        - If Δ' = 0, θ ≈ 0 and the eigenbasis aligns with {|X1⟩, |X2⟩}.
        - Nonzero Δ' mixes X1 and X2; θ encodes the mixing, φ the relative
          phase.
        - The global phase of eigenvectors is irrelevant; we use magnitudes
          for θ and a relative phase difference for φ.
        """
        delta = float(self.fss)
        delta_p = float(self.delta_prime)

        H = 0.5 * np.array([[delta, delta_p], [delta_p, -delta]], dtype=complex)

        eigvals, eigvecs = np.linalg.eigh(H)

        v = eigvecs[:, 0]
        v0, v1 = v[0], v[1]

        theta = np.arctan2(np.abs(v1), np.abs(v0))

        phi = np.angle(v1) - np.angle(v0)

        return float(theta), float(phi), eigvals
