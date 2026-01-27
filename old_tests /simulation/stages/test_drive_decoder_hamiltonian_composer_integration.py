import unittest
import numpy as np
from scipy.constants import e as _e, hbar as _hbar

from bec.params.dipole_params import DipoleParams
from bec.params.energy_levels import EnergyLevels
from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot

from bec.light.envelopes import GaussianEnvelope
from bec.light.classical import ClassicalCoherentDrive, JonesState

from bec.simulation.stages.drive_decoder import (
    DefaultDriveDecoder,
    DecodePolicy,
)
from bec.simulation.stages.hamiltonian_composition import (
    DefaultHamiltonianComposer,
    HamiltonianCompositionPolicy,
)
from bec.quantum_dot.me.types import HamiltonianTermKind


def _ev_to_omega(e_ev: float) -> float:
    return float(e_ev) * _e / _hbar


def _make_real_qd(*, sigma_t: float, fss_ev: float = 20e-6) -> QuantumDot:
    el = EnergyLevels(
        biexciton=2.6,
        exciton=1.3,
        fss=fss_ev,
        pulse_sigma_t_s=sigma_t,
        enforce_2g_guard=False,
    )
    return QuantumDot(
        energy_levels=el,
        dipole_params=DipoleParams(dipole_moment_Cm=1e-29),
    )


def _make_drive(
    *, w0: float, sigma_t: float, label: str, pol: JonesState
) -> ClassicalCoherentDrive:
    env = GaussianEnvelope(t0=0.0, sigma=sigma_t, area=1.0)
    return ClassicalCoherentDrive(
        envelope=env,
        omega0=1.0,
        laser_omega0=float(w0),
        delta_omega=0.0,
        label=label,
        pol_state=pol,
        pol_transform=None,
    )


def _dims_for_qd(qd: QuantumDot, *, fock_dim: int = 2) -> list[int]:
    # QD (4) + 2 polarization dims per intrinsic mode
    n_modes = len(qd.modes.modes)
    return [4] + [fock_dim, fock_dim] * n_modes


class TestDecodeComposeContract(unittest.TestCase):
    def test_decode_then_compose_emits_consistent_terms(self):
        sigma_t = 100e-12
        qd = _make_real_qd(sigma_t=sigma_t)
        el = qd.energy_levels

        w_gx1 = _ev_to_omega(el.X1)
        drv = _make_drive(
            w0=w_gx1, sigma_t=sigma_t, label="gx1", pol=JonesState.H()
        )

        tlist = np.linspace(-4.0, 4.0, 401)
        time_unit_s = 1e-12

        # ---- decode ----
        decoder = DefaultDriveDecoder(
            policy=DecodePolicy(allow_multi=False, k_bandwidth=3.0)
        )
        resolved = decoder.decode(
            qd, [drv], time_unit_s=time_unit_s, tlist=tlist
        )

        self.assertIsInstance(resolved, tuple)
        self.assertGreaterEqual(len(resolved), 1)
        self.assertEqual(resolved[0].transition, Transition.G_X1)

        # ---- compose ----
        composer = DefaultHamiltonianComposer(
            policy=HamiltonianCompositionPolicy(
                include_detuning_terms=True, hermitian_drive=True
            )
        )
        dims = _dims_for_qd(qd, fock_dim=2)
        terms = composer.compose(
            qd, resolved, dims=dims, time_unit_s=time_unit_s
        )

        self.assertGreater(len(terms), 0)

        # basic kind accounting
        static_terms = [
            t for t in terms if t.kind == HamiltonianTermKind.STATIC
        ]
        drive_terms = [t for t in terms if t.kind == HamiltonianTermKind.DRIVE]
        det_terms = [t for t in terms if t.kind ==
                     HamiltonianTermKind.DETUNING]

        self.assertGreaterEqual(len(static_terms), 1)
        self.assertEqual(len(drive_terms), len(resolved))
        self.assertGreaterEqual(len(det_terms), 1)

        # operator dims sanity
        for t in terms:
            # if your HamiltonianTerm.op is always Qobj, this is enough:
            self.assertTrue(hasattr(t.op, "dims"))
            self.assertEqual(t.op.dims[0], dims)
            self.assertEqual(t.op.dims[1], dims)

        # meta sanity for the emitted drive term
        d0 = drive_terms[0]
        self.assertEqual(d0.meta.get("drive_id"), "gx1")
        self.assertEqual(d0.meta.get("transition"), str(Transition.G_X1))
        self.assertIn("components", d0.meta)
        self.assertIn("candidates", d0.meta)

    def test_dims_contract_matches_context_builder_assumptions(self):
        sigma_t = 50e-12
        qd = _make_real_qd(sigma_t=sigma_t)
        dims = _dims_for_qd(qd, fock_dim=2)

        # QDContextBuilder uses d[1], d[2] for mode0; so need >= 3 dims if any mode exists
        if len(qd.modes.modes) > 0:
            self.assertGreaterEqual(len(dims), 3)


if __name__ == "__main__":
    unittest.main()
