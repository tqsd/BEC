import unittest
import numpy as np

from bec.light.classical import ClassicalDrive


import bec.light.classical as classical_mod
from bec.light.envelopes import SerializableEnvelope


class TestClassicalDrive(unittest.TestCase):
    def test_qutip_coeff_scaling(self):
        # envelope f(t) = t  (physical seconds)
        def env(t):
            return t

        drive = ClassicalDrive(envelope=env, omega0=2.0, laser_omega0=10.0)

        # time_unit_s = seconds per solver unit
        s = 0.5  # 1 solver unit = 0.5 s
        coeff = drive.qutip_coeff(time_unit_s=s)

        # Omega_solver(t') = s * omega0 * env(s * t')
        # For t' = 3.0, env(s * t') = 0.5 * 3.0 = 1.5
        # Omega = 0.5 * 2.0 * 1.5 = 1.5
        self.assertAlmostEqual(coeff(3.0, {}), 1.5, places=12)

    def test_sample_solver_matches_pointwise_coeff(self):
        def env(t):
            return 1.0 + 0.1 * t  # linear in physical time

        drive = ClassicalDrive(envelope=env, omega0=3.0)

        tlist = np.array([0.0, 1.0, 2.0, 5.0], dtype=float)  # solver times
        s = 2.0  # seconds per solver unit
        samples = drive.sample_solver(tlist, time_unit_s=s)

        # Expected: s * omega0 * (1 + 0.1 * (s * t'))
        expected = s * 3.0 * (1.0 + 0.1 * (s * tlist))
        np.testing.assert_allclose(samples, expected, rtol=0, atol=1e-12)

    def test_from_callable_wraps_signature(self):
        # Define a callable that accepts (t, args)
        def omega_fn(t, args=None):
            return 2.0 * t

        drive = ClassicalDrive.from_callable(
            omega_fn, delta_omega=1.0, laser_omega0=5.0, label="raw"
        )

        self.assertEqual(drive.label, "raw")
        self.assertEqual(drive.omega0, 1.0)  # fixed by from_callable
        self.assertIsNotNone(drive._raw_callable)

        coeff = drive.qutip_coeff(time_unit_s=1.0)
        # For t' = 4.0, s=1 -> Omega = 1 * 1.0 * (2.0 * 4.0) = 8.0
        self.assertAlmostEqual(coeff(4.0, {}), 8.0, places=12)

    def test_to_dict_raises_for_non_serializable_envelope(self):
        drive = ClassicalDrive.from_callable(lambda t, args=None: 0.0)
        with self.assertRaises(TypeError):
            _ = drive.to_dict()

    def test_to_dict_raises_for_callable_chirp(self):
        # if envelope is serializable but chirp is callable, JSON should fail
        # We monkeypatch "SerializableEnvelope" check by patching envelope_to_json/envelope_from_json
        # only if needed; otherwise, use a simple monkeypatched SerializableEnvelope-ish object.
        # type: ignore[name-defined]
        class FakeSerializableEnvelope(SerializableEnvelope):
            def __call__(self, t: float) -> float:
                return 1.0

            def to_json(self):
                return {"type": "fake"}

        def chirp(t):
            return 0.1 * t

        drive = ClassicalDrive(
            envelope=FakeSerializableEnvelope(),
            omega0=1.0,
            delta_omega=chirp,
        )

        with self.assertRaises(TypeError):
            _ = drive.to_dict()

    def test_delta_omega_coeff_none_if_zero(self):
        drive = ClassicalDrive(envelope=lambda t: 1.0, delta_omega=0.0)
        self.assertIsNone(drive.delta_omega_coeff(time_unit_s=0.25))

    def test_delta_omega_coeff_constant_float(self):
        drive = ClassicalDrive(envelope=lambda t: 1.0, delta_omega=10.0)
        s = 0.2
        dw = drive.delta_omega_coeff(time_unit_s=s)
        self.assertIsNotNone(dw)
        # returns D * s in solver units
        self.assertAlmostEqual(dw(123.0, {}), 10.0 * s, places=12)

    def test_delta_omega_coeff_callable(self):
        def chirp(t_phys_s: float) -> float:
            return 0.1 * t_phys_s  # rad/s

        drive = ClassicalDrive(envelope=lambda t: 1.0, delta_omega=chirp)
        s = 2.0
        dw = drive.delta_omega_coeff(time_unit_s=s)
        self.assertIsNotNone(dw)

        # t_solver=3 -> t_phys=6 -> chirp=0.6 rad/s -> solver units multiply by s => 1.2
        self.assertAlmostEqual(dw(3.0, {}), 1.2, places=12)

    def test_omega_instantaneous_coeff_none_when_no_carrier_and_no_chirp(self):
        drive = ClassicalDrive(
            envelope=lambda t: 1.0, laser_omega0=None, delta_omega=0.0
        )
        self.assertIsNone(drive.omega_instantaneous_coeff(time_unit_s=0.5))

    def test_omega_instantaneous_coeff_constant_when_only_carrier(self):
        drive = ClassicalDrive(
            envelope=lambda t: 1.0, laser_omega0=10.0, delta_omega=0.0
        )
        s = 0.25
        wL = drive.omega_instantaneous_coeff(time_unit_s=s)
        self.assertIsNotNone(wL)
        # ω_L_solver = ω0 * s
        self.assertAlmostEqual(wL(999.0, {}), 10.0 * s, places=12)

    def test_omega_instantaneous_coeff_adds_carrier_and_chirp(self):
        def chirp(t_phys_s: float) -> float:
            return 0.1 * t_phys_s  # rad/s

        drive = ClassicalDrive(
            envelope=lambda t: 1.0, laser_omega0=10.0, delta_omega=chirp
        )
        s = 2.0
        wL = drive.omega_instantaneous_coeff(time_unit_s=s)
        self.assertIsNotNone(wL)

        # ω0 contribution: 10 * s = 20
        # chirp at t_solver=3 -> t_phys=6 -> 0.6 rad/s -> solver units: *s => 1.2
        self.assertAlmostEqual(wL(3.0, {}), 21.2, places=12)

    def test_from_dict_uses_envelope_from_json(self):
        # Monkeypatch envelope_from_json in the module to return a simple callable
        original = classical_mod.envelope_from_json
        try:

            def fake_env_from_json(env_json):
                scale = float(env_json.get("scale", 1.0))
                return lambda t: scale * t

            classical_mod.envelope_from_json = fake_env_from_json  # patch

            data = {
                "type": "classical_drive",
                "label": "patched",
                "omega0": 5.0,
                "delta_omega": 0.5,
                "envelope": {"scale": 2.0},
                "laser_omega0": 7.5,
            }
            drive = ClassicalDrive.from_dict(data)

            self.assertEqual(drive.label, "patched")
            self.assertEqual(drive.omega0, 5.0)
            self.assertEqual(drive.delta_omega, 0.5)
            self.assertEqual(drive.laser_omega0, 7.5)

            # Check the envelope behavior via qutip_coeff
            coeff = drive.qutip_coeff(time_unit_s=1.0)
            # env(t) = 2.0 * t; Omega = 1.0 * 5.0 * env(t) = 10 * t
            self.assertAlmostEqual(coeff(3.0, {}), 30.0, places=12)
        finally:
            classical_mod.envelope_from_json = original

    def test_with_cached_tlist_and_with_chirp_return_new_instances(self):
        drive = ClassicalDrive(envelope=lambda t: 1.0, delta_omega=1.0)

        tlist = np.linspace(0.0, 1.0, 5)
        d2 = drive.with_cached_tlist(tlist)
        self.assertIsNot(drive, d2)
        self.assertTrue(np.array_equal(d2._cached_tlist, tlist))
        self.assertTrue(drive._cached_tlist.size == 0)  # original unchanged

        # Replace chirp with a callable
        def chirp_fn(t_sec: float) -> float:
            return 0.1 * t_sec

        d3 = drive.with_chirp(chirp_fn)
        self.assertIsNot(drive, d3)
        self.assertTrue(callable(d3.delta_omega))
        self.assertEqual(drive.delta_omega, 1.0)  # original unchanged

    def test_coeff_ignores_args(self):
        drive = ClassicalDrive(envelope=lambda t: 2.0, omega0=4.0)
        coeff = drive.qutip_coeff(time_unit_s=3.0)
        self.assertAlmostEqual(coeff(1.23), 3.0 * 4.0 * 2.0, places=12)
        self.assertAlmostEqual(
            coeff(1.23, {"anything": 1}), 3.0 * 4.0 * 2.0, places=12
        )
