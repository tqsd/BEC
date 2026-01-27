import unittest
from typing import Callable, Optional, List, Any

import numpy as np

from bec.params.dipole_params import DipoleParams
from bec.params.energy_levels import EnergyLevels
from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot

from bec.light.envelopes import GaussianEnvelope
from bec.light.classical import ClassicalCoherentDrive, JonesState

from bec.quantum_dot.me.types import HamiltonianTerm, HamiltonianTermKind
from bec.simulation.types import ResolvedDrive

from bec.simulation.stages.hamiltonian_composition import (
    DefaultHamiltonianComposer,
    HamiltonianCompositionPolicy,
)

from bec.quantum_dot.me.coeffs import CoeffExpr


# ----------------------------
# helpers
# ----------------------------


def _const_detuning(delta_solver: float) -> Callable[[float], float]:
    d = float(delta_solver)
    return lambda t: d


def _terms_of_kind(
    terms: List[HamiltonianTerm], kind: HamiltonianTermKind
) -> List[HamiltonianTerm]:
    return [t for t in terms if t.kind == kind]


def _qd_dims_from_registry(
    qd: QuantumDot, *, per_pol_dim: int = 2
) -> List[int]:
    """
    Derive a compatible dims list from the *actual* mode registry.

    Your QDContextBuilder uses indexing:
        s = 1 + 2*i
        and accesses d[s] and d[s+1]
    i.e. 2 tensor factors per mode.

    So the minimal consistent dims layout is:
        [4] + [per_pol_dim] * (2 * num_modes)

    This does NOT assume anything about how you later interpret those factors
    (Fock x pol vs pol+ and pol- etc.) — it just matches your current context contract.
    """
    n_modes = len(qd.modes.modes)
    return [4] + [int(per_pol_dim)] * (2 * n_modes)


def _enum_name(x: Any) -> str:
    """
    Normalize QDState-like things into a stable name:
      QDState.X1 -> "X1"
      "QDState.X1" -> "X1"
      "X1" -> "X1"
    """
    s = str(x)
    # common forms: "QDState.X1" or "QDState.X1" repr-like
    if "." in s:
        s = s.split(".")[-1]
    # strip any stray quotes or angle brackets
    return s.strip().strip("'").strip('"').strip(">").strip("<")


def _find_projector_op(
    qd: QuantumDot, *, dims: List[int], time_unit_s: float, level_name: str
):
    """
    Find a DETUNING catalog projector by meta["level"] and return its operator.

    Matches by enum *name* (e.g. "X1"), not by exact string repr.
    """
    catalog = qd.h_builder.build_catalog(dims=dims, time_unit_s=time_unit_s)
    want = str(level_name)
    for t in catalog:
        if t.kind != HamiltonianTermKind.DETUNING:
            continue
        meta = dict(t.meta or {})
        if meta.get("type") != "projector":
            continue
        lvl = meta.get("level", None)
        if lvl is None:
            continue
        if _enum_name(lvl) == want:
            return t.op
    raise KeyError(f"Projector for level {want!r} not found in catalog.")


def _find_coherence_op(
    qd: QuantumDot,
    *,
    dims: List[int],
    time_unit_s: float,
    bra_name: str,
    ket_name: str,
):
    """
    Find a DRIVE catalog coherence operator by meta["bra"], meta["ket"].

    Matches by enum *name* (e.g. "X1", "G"), not exact repr strings.
    """
    catalog = qd.h_builder.build_catalog(dims=dims, time_unit_s=time_unit_s)
    want_bra = str(bra_name)
    want_ket = str(ket_name)
    for t in catalog:
        if t.kind != HamiltonianTermKind.DRIVE:
            continue
        meta = dict(t.meta or {})
        if meta.get("type") != "coherence":
            continue
        bra = meta.get("bra", None)
        ket = meta.get("ket", None)
        if bra is None or ket is None:
            continue
        if _enum_name(bra) == want_bra and _enum_name(ket) == want_ket:
            return t.op
    raise KeyError(
        f"Coherence op for bra={want_bra!r}, ket={
            want_ket!r} not found in catalog."
    )


def _eval_coeff(coeff: CoeffExpr, t: float) -> complex:
    return complex(coeff(t, {}))


# ----------------------------
# test suite
# ----------------------------


class TestHamiltonianComposerRealQD(unittest.TestCase):
    def setUp(self) -> None:
        # IMPORTANT: enforce_2g_guard must be False for these unit tests
        self.sigma_t = 50e-12
        el = EnergyLevels(
            biexciton=2.6,
            exciton=1.3,
            fss=20e-6,
            pulse_sigma_t_s=self.sigma_t,
            enforce_2g_guard=False,
        )

        self.qd = QuantumDot(
            energy_levels=el,
            dipole_params=DipoleParams(dipole_moment_Cm=1e-29),
        )

        # Solver unit is 1 ps
        self.time_unit_s = 1e-12

        # IMPORTANT: do NOT hardcode dims; derive it from the QD’s actual mode registry.
        self.per_pol_dim = 2

        self.composer = DefaultHamiltonianComposer(
            policy=HamiltonianCompositionPolicy(
                include_detuning_terms=True,
                hermitian_drive=True,
                # These flags assume you implemented them in your composer.
                split_components=True,
                build_drive_coeff_from_physical=True,
            )
        )

    def _dims(self) -> List[int]:
        return _qd_dims_from_registry(self.qd, per_pol_dim=self.per_pol_dim)

    def _make_drive(
        self,
        *,
        label: str,
        omega0: float = 1.0,
        delta_omega: float = 0.0,
        pol: Optional[JonesState] = None,
    ) -> ClassicalCoherentDrive:
        env = GaussianEnvelope(t0=0.0, sigma=self.sigma_t, area=1.0)
        return ClassicalCoherentDrive(
            envelope=env,
            omega0=float(omega0),
            pol_state=pol,
            pol_transform=None,
            laser_omega0=1.0,
            delta_omega=delta_omega,  # may be float or chirp fn
            label=label,
        )

    # ------------------------------------------------------------

    def test_static_terms_present_without_drives(self) -> None:
        out = self.composer.compose(
            self.qd,
            resolved=[],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )
        static_terms = _terms_of_kind(out, HamiltonianTermKind.STATIC)
        self.assertGreater(len(static_terms), 0)

    # ------------------------------------------------------------

    def test_single_transition_emits_drive_and_detuning(self) -> None:
        drv = self._make_drive(
            label="d0", omega0=2.0, delta_omega=0.0, pol=JonesState.H()
        )

        rd = ResolvedDrive(
            drive_id="d0",
            physical=drv,
            components=((Transition.G_X1, 1.0 + 0j),),
            transition=Transition.G_X1,
            detuning=_const_detuning(0.25),
            candidates=(Transition.G_X1,),
            meta={},
        )

        out = self.composer.compose(
            self.qd,
            resolved=[rd],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )
        drive_terms = _terms_of_kind(out, HamiltonianTermKind.DRIVE)
        det_terms = _terms_of_kind(out, HamiltonianTermKind.DETUNING)

        # split_components=True => 1 component => exactly 1 DRIVE term
        self.assertEqual(len(drive_terms), 1)
        self.assertEqual(len(det_terms), 1)

        # DRIVE op must match catalog |X1><G| + h.c.
        A = _find_coherence_op(
            self.qd,
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
            bra_name="X1",
            ket_name="G",
        )
        expected = A + A.dag()
        self.assertAlmostEqual(
            (drive_terms[0].op - expected).norm(), 0.0, places=12
        )

        # DETUNING op must be projector on X1
        PX1 = _find_projector_op(
            self.qd,
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
            level_name="X1",
        )
        self.assertAlmostEqual((det_terms[0].op - PX1).norm(), 0.0, places=12)

        # coeff must be callable
        self.assertIsNotNone(drive_terms[0].coeff)
        self.assertTrue(callable(drive_terms[0].coeff))

        # detuning coeff must be callable and return the constant
        self.assertIsNotNone(det_terms[0].coeff)
        self.assertTrue(callable(det_terms[0].coeff))

        val = complex(det_terms[0].coeff(0.0, {}))
        self.assertAlmostEqual(val.real, 0.25, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)

    # ------------------------------------------------------------

    def test_components_split_into_multiple_drive_terms_and_detunings(
        self,
    ) -> None:
        drv = self._make_drive(
            label="d0", omega0=1.0, delta_omega=0.0, pol=JonesState.D()
        )

        rd = ResolvedDrive(
            drive_id="d0",
            physical=drv,
            components=(
                (Transition.G_X1, 1.0 + 0j),
                (Transition.G_X2, 0.0 + 1.0j),
            ),
            transition=None,
            detuning=_const_detuning(0.0),
            candidates=(Transition.G_X1, Transition.G_X2),
            meta={},
        )

        out = self.composer.compose(
            self.qd,
            resolved=[rd],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )
        drive_terms = _terms_of_kind(out, HamiltonianTermKind.DRIVE)
        det_terms = _terms_of_kind(out, HamiltonianTermKind.DETUNING)

        # Two components -> two DRIVE terms
        self.assertEqual(len(drive_terms), 2)

        # Detuning should exist for both upper levels X1 and X2
        self.assertEqual(len(det_terms), 2)

        PX1 = _find_projector_op(
            self.qd,
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
            level_name="X1",
        )
        PX2 = _find_projector_op(
            self.qd,
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
            level_name="X2",
        )

        det_ops = [t.op for t in det_terms]
        self.assertTrue(any((op - PX1).norm() < 1e-12 for op in det_ops))
        self.assertTrue(any((op - PX2).norm() < 1e-12 for op in det_ops))

    # ------------------------------------------------------------

    def test_component_weight_scales_drive_coeff_pointwise(self) -> None:
        drv = self._make_drive(
            label="d0", omega0=1.0, delta_omega=0.0, pol=JonesState.H()
        )

        rd1 = ResolvedDrive(
            drive_id="d0",
            physical=drv,
            components=((Transition.G_X1, 1.0 + 0j),),
            transition=Transition.G_X1,
            detuning=None,
            candidates=(Transition.G_X1,),
            meta={},
        )
        rd2 = ResolvedDrive(
            drive_id="d0",
            physical=drv,
            components=((Transition.G_X1, 0.25 + 0.5j),),
            transition=Transition.G_X1,
            detuning=None,
            candidates=(Transition.G_X1,),
            meta={},
        )

        out1 = self.composer.compose(
            self.qd,
            resolved=[rd1],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )
        out2 = self.composer.compose(
            self.qd,
            resolved=[rd2],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )

        drive1 = _terms_of_kind(out1, HamiltonianTermKind.DRIVE)[0]
        drive2 = _terms_of_kind(out2, HamiltonianTermKind.DRIVE)[0]

        c1 = drive1.coeff
        c2 = drive2.coeff
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)

        t = 0.0
        v1 = _eval_coeff(c1, t)
        v2 = _eval_coeff(c2, t)

        # Ratio should match weight ratio at any fixed time.
        ratio = v2 / v1
        self.assertAlmostEqual(ratio.real, 0.25, places=12)
        self.assertAlmostEqual(ratio.imag, 0.5, places=12)

    # ------------------------------------------------------------

    def test_delta_omega_or_chirp_makes_coeff_time_dependent(self) -> None:
        # nonzero delta_omega creates time-dependent phase accumulation
        delta_omega = 2.0 * np.pi * 1e9  # rad/s
        drv = self._make_drive(
            label="d0", omega0=1.0, delta_omega=delta_omega, pol=JonesState.H()
        )

        rd = ResolvedDrive(
            drive_id="d0",
            physical=drv,
            components=((Transition.G_X1, 1.0 + 0j),),
            transition=Transition.G_X1,
            detuning=None,
            candidates=(Transition.G_X1,),
            meta={},
        )

        out = self.composer.compose(
            self.qd,
            resolved=[rd],
            dims=self._dims(),
            time_unit_s=self.time_unit_s,
        )
        drive = _terms_of_kind(out, HamiltonianTermKind.DRIVE)[0]

        coeff = drive.coeff
        self.assertIsNotNone(coeff)

        # Compare two solver times (converted internally via time_unit_s)
        t0 = 0.0
        t1 = 10.0  # solver units => 10 ps
        v0 = _eval_coeff(coeff, t0)
        v1 = _eval_coeff(coeff, t1)

        # The envelope might also change, but phase change alone is enough.
        self.assertTrue(abs(v1 - v0) > 1e-15)


if __name__ == "__main__":
    unittest.main()
