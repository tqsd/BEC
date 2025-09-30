from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.constants import hbar as _hbar, e as _e

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
        Energy of the |XX> level (eV).
    exciton : float
        Mean energy of the exciton manifold (midpoint of X1 and X2) in eV.
    fss : float
        Fine-structure splitting between X1 and X2 (eV). The split levels are
        X1 = exciton + fss/2 and X2 = exciton - fss/2.
    delta_prime : float
        Off-diagonal coupling between the exciton sublevels (eV), for example
        due to anisotropy; appears as the sigma_x term in the exciton
        Hamiltonian.

    Notes
    -----
    - The ground state G is fixed at 0 by convention.
    - The biexciton level XX is taken to be `biexciton`.
    - Transition tuples are stored as `(energy_eV, Transition, label_str)` for
      convenience.
    """

    biexciton: float
    exciton: float
    fss: float
    delta_prime: float = field(default=0.0)
    enforce_2g_guard: bool = True
    min_binding_energy_meV: float = 1.0
    pulse_sigma_t_s: Optional[float] = None
    G: float = field(init=False, default=0.0)
    X1: float = field(init=False)
    X2: float = field(init=False)
    XX: float = field(init=False)
    e_G_X1: Tuple[float, Transition, str] = field(init=False)
    e_G_X2: Tuple[float, Transition, str] = field(init=False)
    e_X1_XX: Tuple[float, Transition, str] = field(init=False)
    e_X2_XX: Tuple[float, Transition, str] = field(init=False)

    def __post_init__(self) -> None:
        r"""
        Compute derived level positions and transition energies.

        Sets
        ----
        X1, X2
            Split exciton energies from mean ``exciton`` and ``fss``.
        XX
            Equals ``biexciton``.
        e_* tuples
            Transition entries ``(energy_eV, Transition, label)``.
        binding_energy : float
            Defined as ``(X1 + X2) - XX``.

        Math
        ----
        .. math::

           X_1 = E_X + \frac{\text{fss}}{2},\quad
           X_2 = E_X - \frac{\text{fss}}{2},\quad
           E_{\text{bind}} = (X_1 + X_2) - E_{XX}.
        """

        self.X1 = self.exciton + self.fss / 2
        self.X2 = self.exciton - self.fss / 2
        self.XX = self.biexciton
        self.binding_energy = (self.X1 + self.X2) - self.biexciton
        self.e_G_X1 = (self.X1, Transition.G_X1, "G_X1")
        self.e_G_X2 = (self.X2, Transition.G_X2, "G_X2")
        self.e_X1_XX = (self.XX - self.X1, Transition.X1_XX, "X1_XX")
        self.e_X2_XX = (self.XX - self.X2, Transition.X2_XX, "X2_XX")
        self.e_G_X = (self.X2, Transition.G_X, "G_X")
        self.e_X_XX = (self.XX - self.X1, Transition.X_XX, "X_XX")
        if self.enforce_2g_guard:
            self._validate_two_photon_regime()

    def _validate_two_photon_regime(self) -> None:
        r"""
        Guard for validity of an effective two-photon (2-gamma) model.

        Raises if the single-photon detuning is too small compared to a
        threshold. Uses ``|E_bind|`` as a proxy; on two-photon resonance,
        the single-photon detuning satisfies ``|Delta_1gamma| = |E_bind|/2``.

        If ``pulse_sigma_t_s`` is provided, a bandwidth-based floor is applied:

        .. math::

           \text{threshold} \ge 6\hbar/\sigma_t

        which is converted to meV for comparison.

        Raises
        ------
        Exception
            If ``abs(binding_energy_meV) < threshold_meV``.
        """
        thresh_meV = float(self.min_binding_energy_meV)
        if self.pulse_sigma_t_s and self.pulse_sigma_t_s > 0.0:
            sigma_omega = 1.0 / float(self.pulse_sigma_t_s)  # [rad/s]
            sigma_E_J = _hbar * sigma_omega  # [J]
            sigma_E_meV = 1e3 * sigma_E_J / _e  # [meV]
            thresh_meV = max(thresh_meV, 6.0 * sigma_E_meV)

        Eb_meV = 1e3 * float(self.binding_energy)  # [meV]
        if abs(Eb_meV) < thresh_meV:

            raise Exception(
                "Effective two-photon Hamiltonian invalid: single-photon detuning is too small.\n"
                f"  |E_b| = {
                    abs(Eb_meV):.3f} meV  (|Delta_1gamma| = |E_b|/2)\n"
                f"  threshold = {thresh_meV:.3f} meV "
                "(rule of thumb: |E_b| > 6 * hbar/sigma_t or > min_binding_energy_meV)\n"
                "Use the full four-level ladder Hamiltonian (with explicit 1gamma couplings) "
                "or increase pulse length/detuning."
            )

    def compute_modes(self) -> List[LightMode]:
        r"""
        Build ``LightMode`` objects for allowed radiative transitions.

        Returns
        -------
        List[LightMode]
            If `fss == 0` returns two modes for the degenerate case
            (`G <-> X` and `X <-> XX`). Otherwise returns four modes:
            `G <-> X1`, `G <-> X2, `X1 <-> XX`, `X2 <-> XX`.
            Each mode has `energy_ev` filled and
            `source = TransitionType.INTERNAL`.
        """
        modes: List[LightMode] = []
        if self.fss == 0:
            for t in [self.e_G_X, self.e_X_XX]:
                modes.append(
                    LightMode(
                        energy_ev=t[0],
                        source=TransitionType.INTERNAL,
                        transition=t[1],
                        label=t[2],
                        label_tex=t[1].tex(),
                    )
                )
        else:
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
        r"""
        Diagonalize the exciton subspace Hamiltonian and return rotation params.

        The exciton subspace in the basis `{|X1>, |X2>}` is modeled by

        .. math::

           H = \tfrac{1}{2}
               \begin{bmatrix}
                 \Delta & \Delta' \\\\
                 \Delta' & -\Delta
               \end{bmatrix},

        where `Delta = fss` and `Delta' = delta_prime`.

        Returns
        -------
        theta : float
            Mixing angle in radians.
        phi : float
            Relative phase in radians.
        eigvals : np.ndarray
            Eigenvalues of ``H`` in ascending order.

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
