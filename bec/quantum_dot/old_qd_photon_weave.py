import numpy as np
import jax.numpy as jnp
from typing import Callable, List, Optional, Tuple, Dict

from photon_weave.extra import interpreter
from photon_weave.state.custom_state import CustomState
from photon_weave.state.envelope import Envelope
from qutip import Qobj


from bec.helpers.converter import (
    energy_to_wavelength_nm,
    omega_from_lambda_nm,
    omega_from_energy_eV,
)
from bec.light.base import LightMode
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.envelopes import GaussianEnvelope
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from .base import QuantumDotSystemBase


from bec.operators.fock_operators import (
    rotated_ladder_operator,
    Ladder,
    Pol,
    vacuum_projector,
)
from bec.operators.qd_operators import QDState, transition_operator


class QuantumDotSystem(QuantumDotSystemBase):
    def __init__(
        self,
        energy_levels: EnergyLevels,
        initial_state: QDState = QDState.G,
        cavity_params: Optional[CavityParams] = None,
        dipole_params: Optional[DipoleParams] = None,
        time_unit_s: float = 1e-9,
    ):
        super().__init__(
            energy_levels,
            initial_state,
            cavity_params,
            dipole_params,
            time_unit_s,
        )
        self.dot = CustomState(4)
        self.dot.state = int(initial_state)
        # Pass By Context
        self._pb_context = {}
        self._pb_modes = []

        self.build_context()

    @property
    def modes(self):
        return (*self._modes, *self._pb_modes)

    def filter_modes(self, label: str):
        candidates = [m for m in self.modes if m.label == label]
        if len(candidates) == 0:
            print(f"[QD] no mode found: Searching for {label}")
            print("[QD] known modes:", [m.label for m in self.modes])
            return None
        return candidates[0]

    @property
    def context(self):
        ctx = {}
        ctx.update(self._context)
        ctx.update(self._pb_context)
        return ctx

    def reset(self):
        self._pb_context = {}
        self._pb_modes = []

    def _interp(self, expr: tuple, dims: List[int]) -> np.ndarray:
        """
        Evaluate a symbolic expression with the current context into a dense
        array
        """
        return np.array(interpreter(expr, self.context, dims))

    def _qobj(self, expr: tuple, dims: List[int]) -> Qobj:
        """
        Evaluate a symbolic expression and wrap it as QuTiP Qobj with correct
        dims
        """
        return Qobj(self._interp(expr, dims), dims=[dims, dims]).to("csr")

    def _by_label(self, label: str) -> int:
        """
        Resolve the mode index by label (first match)
        """
        for i, m in enumerate(self.modes):
            if getattr(m, "label", None) == label:
                return i
        raise ValueError(f"No mode with label {label!r}")

    def _qd(self, name: str) -> jnp.ndarray:
        """Fetch a QD-only operator (already a 4x4 array) from context by name."""
        return self.context[name]([])

    def _k(self, qd: str | jnp.ndarray, fock: str, pos: int) -> tuple:
        """Alias for _kron_pad with shorter name."""
        return self._kron_pad(qd, fock, pos)

    def _near_resonant_indices(
        self, wavelength_nm: float, sigma_time_s: float, k: float = 2.0
    ):
        """
        Determine which quantum dot transitions are near-resonant with a given
        external field mode, based on its center wavelength and temporal width.

        Parameters
        ----------
        wavelength_nm : float
            Center wavelength of the external field in nanometers.
        sigma_time_s : float
            Temporal standard deviation of the field envelope (in seconds).
            Used to estimate the spectral bandwidth as σ_ω ≈ 1 / σ_t.
        k : float, optional
            Multiplicative tolerance factor (default = 2.0). A transition is
            considered "near-resonant" if its detuning from the mode frequency
            is less than k × σ_ω.

        Returns
        -------
        near : list of int
            Indices of transitions considered near-resonant:
                0 → G ↔ X1
                1 → G ↔ X2
                2 → X1 ↔ XX
                3 → X2 ↔ XX
        near_tpe_center : bool
            True if the mode is near-resonant with the effective
            two-photon excitation (TPE) center frequency
            (i.e., (E_XX − E_G)/2).

        Notes
        -----
        - The function converts the given wavelength into angular frequency ω
        and compares it against all candidate transition frequencies.
        - The TPE center is checked separately, since it corresponds to half
        the biexciton energy gap rather than a single-photon line.
        """
        # compute candidate transition wavelengths
        EL = self.energy_levels
        lines_nm = [
            energy_to_wavelength_nm(EL["X1"] - EL["G"]),  # 0: G<->X1
            energy_to_wavelength_nm(EL["X2"] - EL["G"]),  # 1: G<->X2
            energy_to_wavelength_nm(EL["XX"] - EL["X1"]),  # 2: X1<->XX
            energy_to_wavelength_nm(EL["XX"] - EL["X2"]),  # 3: X2<->XX
            energy_to_wavelength_nm((EL["XX"] - EL["G"]) / 2),  # 4: TPE center
        ]

        wm = omega_from_lambda_nm(wavelength_nm)
        w_lines = [
            omega_from_energy_eV(EL["X1"] - EL["G"]),
            omega_from_energy_eV(EL["X2"] - EL["G"]),
            omega_from_energy_eV(EL["XX"] - EL["X1"]),
            omega_from_energy_eV(EL["XX"] - EL["X2"]),
            omega_from_lambda_nm(
                lines_nm[4]
            ),  # same as wm if exactly at half energy
        ]
        sigma_w = 2 * np.pi / sigma_time_s

        near = []
        for i in [0, 1, 2, 3]:
            print(wm, w_lines[i])
            if abs(wm - w_lines[i]) <= k * sigma_w:
                near.append(i)
        near_tpe_center = abs(wm - w_lines[4]) <= k * sigma_w
        return near, near_tpe_center

    def build_context(self):

        # Quantum Dot Operators
        ctx: Dict[str, Callable] = {
            "s_XX_G": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.G)
            ),
            "s_XX_X1": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.X1)
            ),
            "s_XX_X2": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.X2)
            ),
            "s_X1_G": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.G)
            ),
            "s_X2_G": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.G)
            ),
            "s_G_X1": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.X1)
            ),
            "s_G_X2": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.X2)
            ),
            "s_G_XX": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.XX)
            ),
            "s_X1_XX": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.XX)
            ),
            "s_X2_XX": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.XX)
            ),
            "s_X1_X1": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.X1)
            ),
            "s_X1_X2": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.X2)
            ),
            "s_X2_X1": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.X1)
            ),
            "s_X2_X2": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.X2)
            ),
            "s_XX_XX": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.XX)
            ),
            "idq": lambda _: jnp.eye(4),
        }

        for i, _ in enumerate(self.modes):
            f_ctx = self._create_fock_context(i)
            ctx.update(f_ctx)

        self._context = ctx

    def _create_fock_context(self, i: int):
        s = 1 + i * 2
        THETA, PHI = self.THETA, self.PHI
        return {
            f"a{i}+": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.PLUS
            ),
            f"a{i}+_dag": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.PLUS
            ),
            f"a{i}-": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.MINUS
            ),
            f"a{i}-_dag": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.MINUS
            ),
            f"n{i}+": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, pol=Pol.PLUS, operator=Ladder.A_DAG
            )
            @ rotated_ladder_operator(
                d[_s], THETA, PHI, pol=Pol.PLUS, operator=Ladder.A
            ),
            f"n{i}-": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, pol=Pol.MINUS, operator=Ladder.A_DAG
            )
            @ rotated_ladder_operator(
                d[_s], THETA, PHI, pol=Pol.MINUS, operator=Ladder.A
            ),
            f"aa{i}": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.PLUS
            )
            @ rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.MINUS
            ),
            f"aa{i}_dag": lambda d, _s=s: rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.PLUS
            )
            @ rotated_ladder_operator(
                d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.MINUS
            ),
            f"if{i}": lambda d, _s=s: jnp.eye(d[_s] * d[_s + 1]),
            f"vac{i}": lambda d, _s=s: vacuum_projector(d[_s]),
        }

    def _kron_pad(
        self, qd_op: str | jnp.ndarray, fock_op: str, position: int
    ) -> tuple:
        match fock_op:
            case "a+":
                op = f"a{position}+"
            case "a+_dag":
                op = f"a{position}+_dag"
            case "a-":
                op = f"a{position}-"
            case "a-_dag":
                op = f"a{position}-_dag"
            case "aa":
                op = f"aa{position}"
            case "aa_dag":
                op = f"aa{position}_dag"
            case "n+":
                op = f"n{position}+"
            case "n-":
                op = f"n{position}-"
            case "i":
                op = f"if{position}"
            case "vac":
                op = f"vac{position}"
            case _:
                raise Exception(f"Unknown operator {fock_op}")

        op_order = []
        for i, mode in enumerate(self.modes):
            if i == position:
                op_order.append(op)
            else:
                op_order.append(f"if{i}")
        return ("kron", qd_op, *op_order)

    def register_flying_mode(
        self,
        flying_mode: Envelope,
        gaussian: GaussianEnvelope,
        label: str,
        wavelength_tolerance_nm: float = 0.2,
        *,
        k: float = 2.0,  # detuning-vs-bandwidth threshold
        tpe_alpha_X1: float = 0.0,
        tpe_alpha_X2: float = 0.0,
    ):
        """
        Register one physical flying mode and split it into up to TWO LightModes:
        - one 'single' LightMode that drives all near-resonant single-photon transitions;
        - one 'tpe' LightMode (at half energy) whose eliminated arms exclude any explicit ones.
        """
        lam = flying_mode.wavelength
        # Classify near-resonant single-photon transitions and TPE center
        near, near_tpe_center = self._near_resonant_indices(
            lam, gaussian.sigma * self.time_unit_s, k=k
        )

        # (A) single-photon LightMode if any near-resonant single-photon lines
        print(near, near_tpe_center)
        if near:
            self._pb_modes.append(
                LightMode(
                    wavelength_nm=lam,
                    source="external",
                    # e.g., [0,2] if broadband covers G<->X1 and X1<->XX
                    transitions=near,
                    gaussian=gaussian,
                    label=label,
                    role="single",
                )
            )

        # (B) TPE LightMode if near TPE center
        if near_tpe_center:
            # Explicit arms are those we plan to drive explicitly via (A).
            explicit_arms = set()
            if 0 in near or 2 in near:
                explicit_arms.add("X1")
            if 1 in near or 3 in near:
                explicit_arms.add("X2")
            eliminated = {
                "X1",
                "X2",
            } - explicit_arms  # only virtual arms contribute to Λ(t)

            self._pb_modes.append(
                LightMode(
                    wavelength_nm=lam,
                    source="external",
                    transitions=[4],  # TPE “transition index”
                    gaussian=gaussian,
                    label=f"{label}_TPE",
                    role="tpe",
                    tpe_eliminated=eliminated,
                    tpe_alpha_X1=tpe_alpha_X1,
                    tpe_alpha_X2=tpe_alpha_X2,
                )
            )

        if not near and not near_tpe_center:
            # Fallback: your old tolerance list (optional)
            levels = self.energy_levels
            possibles = [
                energy_to_wavelength_nm(levels["X1"] - levels["G"]),
                energy_to_wavelength_nm(levels["X2"] - levels["G"]),
                energy_to_wavelength_nm(levels["XX"] - levels["X1"]),
                energy_to_wavelength_nm(levels["XX"] - levels["X2"]),
                energy_to_wavelength_nm((levels["XX"] - levels["G"]) / 2),
            ]
            candidates = [
                i
                for i, t in enumerate(possibles)
                if abs(t - lam) < wavelength_tolerance_nm
            ]
            if not candidates:
                print("[QD] no coupling candidates found!")
                print("[QD] Possible couplings:", possibles)
            else:
                # treat as single by default
                self._pb_modes.append(
                    LightMode(
                        lam,
                        "external",
                        candidates,
                        gaussian=gaussian,
                        label=label,
                        role="single",
                    )
                )

        self.build_context()

    def build_hamiltonians(
        self,
        dims: List[int],
        classical_2g: Optional[ClassicalTwoPhotonDrive] = None,
        tlist=None,
        wavelength_tolerance_nm: float = 1.0,
        **params,
    ):
        H0 = self.qutip_hamiltonian_fss(dims)  # static base term
        H_terms = [H0]  # first element is a bare Qobj

        # External modes (time-dependent couplings)
        for m in [
            m for m in self.modes if getattr(m, "source", None) == "external"
        ]:
            if m.role == "single":
                Hk = self.qutip_hamiltonian_light_matter_interaction(
                    m.label, dims
                )
                f = m.gaussian
                H_terms.append([Hk, lambda t, args=None, _f=f: float(_f(t))])
            elif m.role == "tpe":
                Hk = self.qutip_hamiltonian_two_photon_excitation(m.label, dims)
                f = m.gaussian
                A = 0.0
                if "X1" in m.tpe_eliminated:
                    A += m.tpe_alpha_X1
                if "X2" in m.tpe_eliminated:
                    A += m.tpe_alpha_X2
                H_terms.append(
                    [
                        Hk,
                        lambda t, args=None, _f=f, _A=A: float(
                            _A * (_f(t) ** 2)
                        ),
                    ]
                )

        # Classical 2-photon drive
        if classical_2g is not None:
            H_flip = self.qutip_hamiltonian_classical_2g_flip(dims)
            H_det = self.qutip_hamiltonian_classical_2g_detuning(dims)

            # Ω(t) piece (time-dependent)
            H_terms.append([H_flip, classical_2g.qutip_coeff()])

            # Δ_L piece: you can add it as a constant coefficient term:
            if classical_2g.detuning != 0.0:
                H_terms.append([H_det, classical_2g.detuning])
                # or, if you prefer, fold it into H0:
                # H0 += classical_2g.detuning * H_det

        return H_terms

    def _intrinsic_index_by_label(self, label: str) -> int:
        for i, m in enumerate(self.modes):
            if m.source == "internal" and m.label == label:
                return i
        raise ValueError(f"No internal mode with label '{label}'")

    # HAMILTONIAN EXPRESSIONS

    def _hamiltonian_fss(self):
        Delta = self._energy_levels.fss
        Delta_p = self._energy_levels.delta_prime
        proj_X1 = self.context["s_X1_X1"]([])
        proj_X2 = self.context["s_X2_X2"]([])
        X1X2 = self.context["s_X1_X2"]([])
        X2X1 = self.context["s_X2_X1"]([])

        H_fss_local = (Delta / 2) * (proj_X1 - proj_X2) + (Delta_p / 2) * (
            X1X2 + X2X1
        )

        return self._kron_pad(H_fss_local, "i", -1)

    def _pm_of_transition(self, idx: int) -> str | None:
        """
        Map transition index -> rotated linear polarization.
        0: X1<->G, 1: X2<->G, 2: XX<->X1, 3: XX<->X2, 4: XX<->G (two-photon)
        Return '+' for H, '-' for V in your rotated basis.
        """
        if idx in (0, 2):  # X1<->G and XX<->X1 share the same (say V)
            return "+"
        elif idx in (1, 3):  # X2<->G and XX<->X2 share the other (say H)
            return "-"

    def _hamiltonian_light_matter_interaction(self, label):
        ab = ["s_G_X1", "s_G_X2", "s_X1_XX", "s_X2_XX"]
        em = ["s_X1_G", "s_X2_G", "s_XX_X1", "s_XX_X2"]
        H_ints = []
        idx, m = [
            (idx, m) for idx, m in enumerate(self.modes) if m.label == label
        ][0]
        for i in m.transitions:
            pm = self._pm_of_transition(i)
            h = (
                "s_mult",
                1,
                (
                    "add",
                    self._kron_pad(em[i], f"a{pm}_dag", idx),
                    self._kron_pad(ab[i], f"a{pm}", idx),
                ),
            )
            H_ints.append(h)
        return ("add", *H_ints)

    def _hamiltonian_two_photon_excitation(self, label: str):
        i = self._by_label(label)
        H_abs = self._k("s_G_XX", "aa", i)
        H_em = self._k("s_XX_G", "aa_dag", i)
        return ("add", H_abs, H_em)

    def _hamiltonian_classical_2g_flip(self) -> tuple:
        """
        H_flip = (|G><XX| + |XX><G|) kron I_fock
        This is the time-dependent part multiplied by Ω(t) in the solver list.
        """
        H_local = self._qd("s_G_XX") + self._qd("s_XX_G")
        return self._k(H_local, "i", -1)

    def _hamiltonian_classical_2g_detuning(self) -> tuple:
        """
        H_det = |XX><XX| kron I_fock
        This is multiplied by the constant two-photon detuning Δ_L (rad/s).
        """
        H_local = self._qd("s_XX_XX")
        return self._k(H_local, "i", -1)

    # QUTIP HAMILTONIANS

    def qutip_hamiltonian_classical_2g_flip(self, dims: List[int]) -> Qobj:
        return self._qobj(self._hamiltonian_classical_2g_flip(), dims)

    def qutip_hamiltonian_classical_2g_detuning(self, dims: List[int]) -> Qobj:
        return self._qobj(self._hamiltonian_classical_2g_detuning(), dims)

    def qutip_hamiltonian_fss(self, dims: List[int]):
        return self._qobj(self._hamiltonian_fss(), dims)

    def qutip_hamiltonian_light_matter_interaction(
        self, label: str, dims: List[int]
    ):
        return self._qobj(
            self._hamiltonian_light_matter_interaction(label), dims
        )

    def qutip_hamiltonian_two_photon_excitation(
        self, label: str, dims: List[int]
    ):
        return self._qobj(self._hamiltonian_two_photon_excitation(label), dims)

    # COLLAPSE OPERATORS

    def _collapse_operators(self):
        collapse_ops = {}
        intrinsic_modes = [m for m in self.modes if m.source == "internal"]
        print(f"[QD] raw ops, mode_len:{len(intrinsic_modes)}")
        if len(intrinsic_modes) == 1:
            i_xxxg = self._intrinsic_index_by_label("XX<->X<->G")
            collapse_ops = {
                "L_XX_X": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X1"]),
                        self._kron_pad("s_XX_X1", "a+_dag", i_xxxg),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X2"]),
                        self._kron_pad("s_XX_X2", "a-_dag", i_xxxg),
                    ),
                ),
                "L_X_G": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X1_G"]),
                        self._kron_pad("s_X1_G", "a-_dag", i_xxxg),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X2_G"]),
                        self._kron_pad("s_X2_G", "a+_dag", i_xxxg),
                    ),
                ),
            }
        if len(intrinsic_modes) == 2:
            try:
                i_xxx = self._intrinsic_index_by_label("XX<->X")
                i_xg = self._intrinsic_index_by_label("X<->G")
            except Exception as _:
                i_xxx = self._intrinsic_index_by_label("X1<->G & X2<->XX")
                i_xg = self._intrinsic_index_by_label("X2<->G & X1<->XX")

            collapse_ops = {
                "L_XX_X": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X1"]),
                        self._kron_pad("s_XX_X1", "a+_dag", i_xxx),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X2"]),
                        self._kron_pad("s_XX_X2", "a-_dag", i_xxx),
                    ),
                ),
                "L_X_G": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X1_G"]),
                        self._kron_pad("s_X1_G", "a-_dag", i_xg),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X2_G"]),
                        self._kron_pad("s_X2_G", "a+_dag", i_xg),
                    ),
                ),
            }
        elif len(intrinsic_modes) == 4:
            i_xxx1 = self._intrinsic_index_by_label("XX<->X1")
            i_xxx2 = self._intrinsic_index_by_label("XX<->X2")
            i_x1g = self._intrinsic_index_by_label("X1<->G")
            i_x2g = self._intrinsic_index_by_label("X2<->G")
            collapse_ops = {
                "L_XX_X": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X1"]),
                        self._kron_pad("s_XX_X1", "a+_dag", i_xxx1),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_XX_X2"]),
                        self._kron_pad("s_XX_X2", "a-_dag", i_xxx2),
                    ),
                ),
                "L_X_G": (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X1_G"]),
                        self._kron_pad("s_X1_G", "a-_dag", i_x1g),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self.gammas["L_X2_G"]),
                        self._kron_pad("s_X2_G", "a+_dag", i_x2g),
                    ),
                ),
            }
        return collapse_ops

    def qutip_collapse_operators(self, dimensions: list):
        print("[QD] Builgind Collapse ops")
        collapse_ops = self._collapse_operators()
        print(f"[QD] Op nr: {len(collapse_ops)}")
        C_ops = []
        for key in collapse_ops:
            op = interpreter(collapse_ops[key], self.context, dimensions)
            C_ops.append(
                Qobj(
                    np.array(op),
                    dims=[dimensions, dimensions],
                ).to("csr")
            )
        return C_ops

    def qutip_projectors(self, dimensions: list) -> dict[str, Qobj]:
        """Return QD projectors as Qobj with correct full dimensions."""
        projectors = {
            "P_G": jnp.array(transition_operator(QDState.G, QDState.G)),
            "P_X1": jnp.array(transition_operator(QDState.X1, QDState.X1)),
            "P_X2": jnp.array(transition_operator(QDState.X2, QDState.X2)),
            "P_XX": jnp.array(transition_operator(QDState.XX, QDState.XX)),
        }

        result = {}
        for name, op in projectors.items():
            kron_tuple = self._kron_pad(op, "i", -1)
            op_full = interpreter(kron_tuple, self.context, dimensions)
            result[name] = Qobj(
                np.array(op_full), dims=[dimensions, dimensions]
            )

        return result

    def qutip_light_mode_projectors(self, dimensions: list) -> dict[str, Qobj]:
        """
        Per light mode (each carries two pol subspaces '+' and '-'):
        - N[label]         : total photon number (N+ + N-)
        - N+[label], N-[label] : pol-resolved numbers
        - Pvac[label]      : projector onto |0,0>
        - P10[label]       : projector onto |1,0>   (one + photon)
        - P01[label]       : projector onto |0,1>   (one - photon)
        - P11[label]       : projector onto |1,1>   (one in each pol)
        - S0[mode], S1[mode]: Stokes intensities (optional)
        """
        ops: dict[str, Qobj] = {}
        for i, m in enumerate(self.modes):
            label = getattr(m, "label", f"mode_{i}")

            # Basic blocks from your DSL
            n_plus = interpreter(
                self._kron_pad("idq", "n+", i), self.context, dimensions
            )
            n_minus = interpreter(
                self._kron_pad("idq", "n-", i), self.context, dimensions
            )
            p_vac = interpreter(
                self._kron_pad("idq", "vac", i), self.context, dimensions
            )
            I_mode = interpreter(
                self._kron_pad("idq", "i", i), self.context, dimensions
            )

            # Wrap to Qobj
            Np = Qobj(np.array(n_plus), dims=[dimensions, dimensions])
            Nm = Qobj(np.array(n_minus), dims=[dimensions, dimensions])
            P0 = Qobj(np.array(p_vac), dims=[dimensions, dimensions])
            I = Qobj(np.array(I_mode), dims=[dimensions, dimensions])

            # With 0/1 truncation per pol: |1><1| = n, |0><0| = I - n.
            P1p = Np
            P0p = I - Np
            P1m = Nm
            P0m = I - Nm

            # Joint occupancy projectors for the two-pol mode
            P10 = P1p @ P0m  # |1,0>
            P01 = P0p @ P1m  # |0,1>
            P11 = P1p @ P1m  # |1,1>

            # Store
            ops[f"N[{label}]"] = Np + Nm
            ops[f"N+[{label}]"] = Np
            ops[f"N-[{label}]"] = Nm
            ops[f"Pvac[{label}]"] = P0
            ops[f"P10[{label}]"] = P10
            ops[f"P01[{label}]"] = P01
            ops[f"P11[{label}]"] = P11
            # Optional Stokes
            ops[f"S0[{label}]"] = Np + Nm
            ops[f"S1[{label}]"] = Np - Nm

        return ops
