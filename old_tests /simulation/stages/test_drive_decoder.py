import unittest
from typing import Optional

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


def _ev_to_omega(e_ev: float) -> float:
    return float(e_ev) * _e / _hbar


def _make_qd(*, fss_ev: float, sigma_t: float) -> QuantumDot:
    el = EnergyLevels(
        biexciton=2.0,
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
    *,
    w0: Optional[float],
    sigma_t: float,
    label: str,
    delta_omega=0.0,
    pol_state: Optional[JonesState] = None,
) -> ClassicalCoherentDrive:
    """
    IMPORTANT: pol_state must be a JonesState (or None), NOT a numpy array.
    ClassicalCoherentDrive.effective_polarization() calls pol_state.as_array().
    """
    env = GaussianEnvelope(t0=0.0, sigma=sigma_t, area=1.0)
    return ClassicalCoherentDrive(
        envelope=env,
        omega0=1.0,
        laser_omega0=w0,
        delta_omega=delta_omega,
        label=label,
        pol_state=pol_state,
        pol_transform=None,
    )


def _decode(
    qd: QuantumDot,
    drv: ClassicalCoherentDrive,
    *,
    allow_multi: bool = True,
    k_bandwidth: float = 3.0,
    time_unit_s: float = 1e-12,
    tlist: Optional[np.ndarray] = None,
):
    if tlist is None:
        tlist = np.linspace(-5.0, 5.0, 501)

    dec = DefaultDriveDecoder(
        policy=DecodePolicy(
            allow_multi=allow_multi,
            k_bandwidth=k_bandwidth,
            default_kind="1ph",
        )
    )
    # DriveDecoder.decode returns Tuple[ResolvedDrive, ...]
    return dec.decode(qd, [drv], time_unit_s=time_unit_s, tlist=tlist)


class TestDefaultDriveDecoderWithPolarization(unittest.TestCase):
    def test_H_polarization_prefers_G_X1(self):
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)
        el = qd.energy_levels

        w_gx1 = _ev_to_omega(el.X1)
        drv = _make_drive(
            w0=w_gx1,
            sigma_t=sigma_t,
            label="H_on_gx1",
            pol_state=JonesState.H(),  # <-- FIX: JonesState, not ndarray
        )

        out = _decode(qd, drv, allow_multi=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].transition, Transition.G_X1)

    def test_V_polarization_prefers_G_X2(self):
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)
        el = qd.energy_levels

        w_gx2 = _ev_to_omega(el.X2)
        drv = _make_drive(
            w0=w_gx2,
            sigma_t=sigma_t,
            label="V_on_gx2",
            pol_state=JonesState.V(),  # <-- FIX
        )

        out = _decode(qd, drv, allow_multi=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].transition, Transition.G_X2)

    def test_polarization_gate_suppresses_dark_transitions_when_allow_multi(
        self,
    ):
        """
        With huge threshold we'd normally emit many transitions.
        With H polarization, V-only 1ph transitions (G_X2, X2_XX) should be gated away.
        Candidates should still include all *scored* transitions (your contract).
        """
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)
        el = qd.energy_levels

        w_gx1 = _ev_to_omega(el.X1)
        drv = _make_drive(
            w0=w_gx1,
            sigma_t=sigma_t,
            label="H_multi",
            pol_state=JonesState.H(),  # <-- FIX
        )

        out = _decode(qd, drv, allow_multi=True, k_bandwidth=1e9)
        self.assertGreaterEqual(len(out), 1)

        transitions = {rd.transition for rd in out}
        self.assertIn(Transition.G_X1, transitions)
        self.assertNotIn(Transition.G_X2, transitions)
        self.assertNotIn(Transition.X2_XX, transitions)

        # candidates should include all 1ph transitions that were scored
        # (exact set may vary if G_XX is present in your Transition enum)
        cand = set(out[0].candidates)
        self.assertIn(Transition.G_X1, cand)
        self.assertIn(Transition.G_X2, cand)
        self.assertIn(Transition.X1_XX, cand)
        self.assertIn(Transition.X2_XX, cand)

    def test_diagonal_linear_polarization_produces_coherent_merge(self):
        """
        Diagonal (D) polarization couples equally to X1 and X2.
        Expect: a 1ph_coherent ResolvedDrive with transition=None and two components.
        """
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)
        el = qd.energy_levels

        w_mid = 0.5 * (_ev_to_omega(el.X1) + _ev_to_omega(el.X2))

        drv = _make_drive(
            w0=w_mid,
            sigma_t=sigma_t,
            label="diag",
            pol_state=JonesState.D(),  # <-- FIX: use JonesState.D()
        )

        out = _decode(qd, drv, allow_multi=True, k_bandwidth=1e9)

        coh = next(
            (rd for rd in out if rd.meta.get("kind") == "1ph_coherent"), None
        )
        self.assertIsNotNone(coh, "Expected a coherent merged drive entry.")
        assert coh is not None

        self.assertIsNone(coh.transition)
        self.assertTrue(coh.meta.get("detuning_is_reference_only", False))

        comp = dict(coh.components)
        self.assertCountEqual(comp.keys(), [Transition.G_X1, Transition.G_X2])

        self.assertAlmostEqual(
            abs(comp[Transition.G_X1]), 1 / np.sqrt(2.0), places=12
        )
        self.assertAlmostEqual(
            abs(comp[Transition.G_X2]), 1 / np.sqrt(2.0), places=12
        )

        self.assertIn("omega_components_phys_rad_s", coh.meta)

    def test_circular_sigma_plus_coherent_merge_and_phase(self):
        """
        σ+ in HV Jones convention is (1, +i)/sqrt(2).
        Expect coherent merge with relative phase +pi/2 between X2 and X1 components.
        """
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)
        el = qd.energy_levels

        w_mid = 0.5 * (_ev_to_omega(el.X1) + _ev_to_omega(el.X2))

        drv = _make_drive(
            w0=w_mid,
            sigma_t=sigma_t,
            label="sigma+",
            pol_state=JonesState.L(),  # <-- In your JonesState, L() is (1, +i)/sqrt(2)
        )

        out = _decode(qd, drv, allow_multi=True, k_bandwidth=1e9)
        coh = next(
            (rd for rd in out if rd.meta.get("kind") == "1ph_coherent"), None
        )
        self.assertIsNotNone(coh, "Expected a coherent merged drive entry.")
        assert coh is not None

        comp = dict(coh.components)
        c1 = comp[Transition.G_X1]
        c2 = comp[Transition.G_X2]

        self.assertAlmostEqual(abs(c2 / c1), 1.0, places=12)

        phase = np.angle(c2 / c1)  # should be +pi/2
        # wrap to (-pi, pi] already, so just compare
        self.assertAlmostEqual(phase, np.pi / 2.0, places=12)

    def test_missing_carrier_frequency_raises(self):
        sigma_t = 50e-12
        qd = _make_qd(fss_ev=20e-6, sigma_t=sigma_t)

        drv = _make_drive(
            w0=None,
            sigma_t=sigma_t,
            label="no_carrier",
            pol_state=JonesState.H(),  # <-- FIX
        )

        with self.assertRaisesRegex(ValueError, "laser_omega0 is None"):
            _decode(qd, drv, allow_multi=False)


if __name__ == "__main__":
    unittest.main()
