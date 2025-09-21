from typing import Any, Dict, List

from photon_weave.state.custom_state import CustomState
from qutip import Qobj
from bec.light.detuning import two_photon_detuning_profile
from bec.operators.qd_operators import QDState
from bec.params.energy_levels import EnergyLevels
from bec.quantum_dot.collapse_builder import CollapseBuilder
from bec.quantum_dot.context_builder import QDContextBuilder
from bec.quantum_dot.decay_model import DecayModel
from bec.quantum_dot.hamiltonian_builder import HamiltonianBuilder
from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.mode_registry import ModeRegistry
from bec.quantum_dot.observables_builder import ObservablesBuilder
from bec.quantum_dot.diagnostics import Diagnostics


class QuantumDot:
    """
    High-level façade for a four-level quantum-dot (QD) model.

    This class composes all core subsystems—mode registry, symbolic context,
    Hamiltonian builder, decay/γ model (incl. Purcell), collapse operators, and
    observables—behind a stable and compact API.

    It does **not** implement physics itself; instead, it wires together the
    following components:
      - `ModeRegistry`           (intrinsic + external light modes)
      - `QDContextBuilder`       (symbolic operator context for QD+Fock)
      - `HamiltonianBuilder`     (FSS, light–matter, TPE, classical 2γ)
      - `DecayModel`             (ω, Purcell, Γ computation → simulation units)
      - `CollapseBuilder`        (Lindblad collapse operators)
      - `ObservablesBuilder`     (QD projectors and per-mode projectors)
      - `Diagnostics`            (Diagnostics class)

    Parameters
    ----------
    energy_levels : EnergyLevels
        Fully specified level structure with methods:
        `compute_modes()` and `exciton_rotation_params()`.
    cavity_params : Optional[CavityParams]
        Optical cavity parameters for Purcell enhancement.
        If `None`, Purcell=0.
    dipole_params : Optional[DipoleParams]
        Transition dipole (C·m). Required for nonzero spontaneous rates.
    time_unit_s : float, default 1e-9
        Simulation time unit (seconds per “1.0” in solver). All Γ are converted
        to this unit (dimensionless rates = [1/s] * time_unit_s).

    Notes
    -----
    - **Composite Hilbert space layout (dims):**
      The QuTiP `dims` list used across methods must match the context builder.
      Convention (example with `M` light modes, each with two polarizations):
          dims = [4, d1_plus, d1_minus, d2_plus, d2_minus, ...,
                  dM_plus, dM_minus]
      The QD space is always 4. Each mode contributes two factors (+/− pol).
    - **Rotation parameters (θ, φ):**
      Taken from `EnergyLevels.exciton_rotation_params()` and used to define
      the rotated polarization basis for ladder operators.
    - **Equality of interfaces:**
      This façade preserves your original system’s abilities (register modes,
      build Hamiltonians, get collapse ops / observables) while improving
      modularity and testability.

    Examples
    --------
    >>> qd = QuantumDot(EL, cavity_params=CP, dipole_params=DP)
    >>> dims = [4, 2, 2, 2, 2]  # QD + two modes (each truncated to {0,1})
    >>> H = qd.build_hamiltonians(dims)
    >>> C = qd.qutip_collapse_operators(dims)
    >>> Pqd = qd.qutip_projectors(dims)
    >>> Pm  = qd.qutip_light_mode_projectors(dims)
    """

    def __init__(
        self,
        energy_levels: EnergyLevels,
        initial_state: QDState,
        cavity_params=None,
        dipole_params=None,
        time_unit_s: float = 1e-9,
        N_cut: int = 2,
    ):
        self.N_cut = N_cut
        self.dot = CustomState(4)
        self.dot.state = int(initial_state)
        self._EL = energy_levels
        # prepare intrinsic modes and rotation params
        intrinsic_modes = energy_levels.compute_modes()
        theta, phi, _ = energy_levels.exciton_rotation_params()

        # pieces
        self.modes = ModeRegistry(intrinsic_modes, (theta, phi))
        self.context_builder = QDContextBuilder(self.modes, theta, phi)

        # EnergyLevels → dict for decay model (ensure you have a to_dict())
        el_dict = {
            "G": 0.0,
            "X1": energy_levels.X1,
            "X2": energy_levels.X2,
            "XX": energy_levels.XX,
        }
        self.decay = DecayModel(el_dict, cavity_params, dipole_params)

        # build context + helpers
        self._context = self.context_builder.build()
        self.kron = KronPad(self.modes)

        # polarization mapper (your _pm_of_transition)
        def pm(idx: int) -> str | None:
            return "+" if idx in (0, 2) else "-" if idx in (1, 3) else None

        self.hams = HamiltonianBuilder(
            self._context, self.kron, energy_levels, pm
        )
        self.gammas = self.decay.compute()
        self.collapses = CollapseBuilder(
            self.gammas, self._context, self.kron, self.modes
        )
        self.obs = ObservablesBuilder(self._context, self.kron, self.modes)
        self.diagnostics = Diagnostics(
            energy_levels=self._EL,
            gammas=self.gammas,
            mode_provider=self.modes,
            qd=self,
            observable_provider=self.obs,
        )

    def register_flying_mode(self, *_, **kwargs) -> None:
        """
        Register one external (runtime) light mode and refresh the operator
        context.

        This is a thin wrapper intended to be called by your higher-level
        classification logic (e.g., “near-resonant single-photon” vs
        “TPE arm”), which should construct a `LightMode` object and pass it
        as `light_mode=...`.

        Parameters
        ----------
        *args, **kwargs :
            Must contain `light_mode` (a `LightMode` instance). Any other
            arguments are ignored here (they belong to your upstream logic).

        Side Effects
        ------------
        - Appends the given mode to the `ModeRegistry` (external list).
        - Rebuilds the symbolic context to include new mode operators.

        Raises
        ------
        KeyError
            If `light_mode` is not present in `kwargs`.
        """
        lm = kwargs.pop(
            "light_mode"
        )  # e.g., after you classify near-resonant/TPE split
        self.modes.register_external(lm)
        self._context = (
            self.context_builder.build()
        )  # refresh after topology change

    def build_hamiltonians(
        self,
        dims: List[int],
        classical_2g=None,
        *,
        time_unit_s: float = 1.0,
    ) -> List[Qobj | list]:
        H: List[Any] = [self.hams.fss(dims, time_unit_s)]
        # external modes
        for m in [
            m
            for m in self.modes.modes
            if getattr(m, "source", None) == "external"
        ]:
            if m.role == "single":
                Hk = self.hams.lmi(m.label, dims)
                f = m.gaussian
                H.append([Hk, lambda t, _f=f, s=time_unit_s: float(_f(s * t))])
            elif m.role == "tpe":
                Hk = self.hams.tpe(m.label, dims)
                A = 0.0
                if "X1" in m.tpe_eliminated:
                    A += m.tpe_alpha_X1
                if "X2" in m.tpe_eliminated:
                    A += m.tpe_alpha_X2
                f = m.gaussian
                H.append(
                    [
                        Hk,
                        lambda t, _f=f, _A=A, s=time_unit_s: float(
                            _A * (_f(s * t) ** 2)
                        ),
                    ]
                )
        if classical_2g is not None:
            H.append(
                [
                    self.hams.classical_2g_flip(dims),
                    classical_2g.qutip_coeff(time_unit_s=time_unit_s),
                ]
            )

            det = classical_2g.detuning
            if callable(det):
                # det expects physical time [s] and returns Δ_phys(t) [rad/s]
                H.append(
                    [
                        self.hams.classical_2g_detuning(dims),
                        lambda t, det=det, s=time_unit_s: float(det(s * t))
                        * s,  # -> solver units
                    ]
                )
            elif det != 0.0:
                H.append(
                    [
                        self.hams.classical_2g_detuning(dims),
                        # constant coefficient in solver units
                        float(det) * time_unit_s,
                    ]
                )

        return H

    def qutip_collapse_operators(
        self, dims: List[int], time_unit_s: float = 1.0
    ) -> List[Qobj]:
        """
        Build Lindblad collapse operators as `Qobj` for the current mode
        layout.

        Parameters
        ----------
        dims : list[int]
            Composite dimensions as in `build_hamiltonians`.

        Returns
        -------
        list[Qobj]
            Collapse operators (CSR) in the full Hilbert space.

        Notes
        -----
        - Rates `γ` are computed from `DecayModel` and converted into
          simulation units (1/s → 1/Δt) using `time_unit_s`.
        - Operator placement across modes/polarizations follows the registry
          (intrinsic labels like 'XX<->X1', 'X1<->G', etc.).
        """
        return self.collapses.qutip_collapse_ops(dims, time_unit_s)

    def qutip_projectors(self, dims: List[int]) -> Dict[str, Qobj]:
        """
        Return QD population projectors embedded in the full space.

        Parameters
        ----------
        dims : list[int]
            Composite dimensions as in `build_hamiltonians`.

        Returns
        -------
        dict[str, Qobj]
            Keys: 'P_G', 'P_X1', 'P_X2', 'P_XX'.
        """
        return self.obs.qd_projectors(dims)

    def qutip_light_mode_projectors(self, dims: List[int]) -> Dict[str, Qobj]:
        """
        Return per-mode photon-number and occupancy projectors.

        For each light mode (two pol subspaces '+' and '−') this includes:
          - N[label]              : total photon number (N+ + N−)
          - N+[label], N−[label]  : pol-resolved numbers
          - Pvac[label]           : |0,0⟩⟨0,0|
          - P10[label]            : |1,0⟩⟨1,0|
          - P01[label]            : |0,1⟩⟨0,1|
          - P11[label]            : |1,1⟩⟨1,1|
          - S0[label], S1[label]  : Stokes-like intensities (optional)

        Parameters
        ----------
        dims : list[int]
            Composite dimensions as in `build_hamiltonians`.

        Returns
        -------
        dict[str, Qobj]
            Dictionary of `Qobj` observables keyed by descriptive names.
        """
        return self.obs.light_mode_projectors(dims)

    @property
    def context(self) -> Dict[str, Any]:
        """
        Current symbolic operator context (QD + all Fock modes).

        Returns
        -------
        dict[str, Callable | Any]
            A context mapping used by the symbolic interpreter. Rebuilt whenever
            mode topology changes (e.g., after `register_flying_mode`).

        Notes
        -----
        This is primarily useful for advanced users who want to evaluate custom
        symbolic expressions via the shared interpreter.
        """
        return self._context
