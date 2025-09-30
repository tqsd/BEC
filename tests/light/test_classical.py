import unittest
import numpy as np

from bec.light.classical import ClassicalTwoPhotonDrive
import bec.light.classical as classical_mod


class TestClassicalTwoPhotonDrive(unittest.TestCase):
    def test_qutip_coeff_scaling(self):
        # envelope f(t) = t  (in seconds)
        def env(t):
            return t

        drive = ClassicalTwoPhotonDrive(envelope=env, omega0=2.0)

        # time_unit_s = s_per_solver_unit
        s = 0.5  # 1 solver unit = 0.5 s
        coeff = drive.qutip_coeff(time_unit_s=s)

        # Omega_solver(t') = s * omega0 * env(s * t')
        # For t' = 3.0, env(s * t') = 0.5 * 3.0 = 1.5
        # Omega = 0.5 * 2.0 * 1.5 = 1.5
        val = coeff(3.0, {})
        self.assertAlmostEqual(val, 1.5, places=12)

    def test_detuning_solver_float(self):
        drive = ClassicalTwoPhotonDrive(envelope=lambda t: 1.0, detuning=10.0)
        self.assertAlmostEqual(
            drive.detuning_solver(time_unit_s=0.2), 2.0, places=12
        )

    def test_sample_solver_matches_pointwise_coeff(self):
        def env(t):
            return 1.0 + 0.1 * t  # linear in physical time

        drive = ClassicalTwoPhotonDrive(envelope=env, omega0=3.0)

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

        drive = ClassicalTwoPhotonDrive.from_callable(
            omega_fn, detuning=1.0, label="raw"
        )
        coeff = drive.qutip_coeff(time_unit_s=1.0)

        self.assertEqual(drive.label, "raw")
        self.assertEqual(drive.omega0, 1.0)  # fixed by from_callable
        self.assertIsNotNone(drive._raw_callable)

        # For t' = 4.0, s=1 -> Omega = 1 * 1.0 * (2.0 * 4.0) = 8.0
        self.assertAlmostEqual(coeff(4.0, {}), 8.0, places=12)

    def test_to_dict_raises_for_non_serializable_envelope(self):
        # from_callable envelopes are not SerializableEnvelope
        drive = ClassicalTwoPhotonDrive.from_callable(lambda t, args=None: 0.0)
        with self.assertRaises(TypeError):
            _ = drive.to_dict()

    def test_from_dict_uses_envelope_from_json(self):
        # Monkeypatch envelope_from_json in the module to return a simple callable
        original = classical_mod.envelope_from_json
        try:

            def fake_env_from_json(env_json):
                scale = float(env_json.get("scale", 1.0))
                return lambda t: scale * t

            classical_mod.envelope_from_json = fake_env_from_json  # patch

            data = {
                "type": "classical_2photon",
                "label": "patched",
                "omega0": 5.0,
                "detuning": 0.5,
                "envelope": {"scale": 2.0},
                "laser_omega": 7.5,
            }
            drive = ClassicalTwoPhotonDrive.from_dict(data)

            self.assertEqual(drive.label, "patched")
            self.assertEqual(drive.omega0, 5.0)
            self.assertEqual(drive.detuning, 0.5)
            self.assertEqual(drive.laser_omega, 7.5)

            # Check the envelope behavior via qutip_coeff
            coeff = drive.qutip_coeff(time_unit_s=1.0)
            # env(t) = 2.0 * t; Omega = 1.0 * 5.0 * env(t) = 10 * t
            self.assertAlmostEqual(coeff(3.0, {}), 30.0, places=12)
        finally:
            classical_mod.envelope_from_json = original

    def test_from_envelope_json_path(self):
        # Monkeypatch to control envelope reconstruction
        original = classical_mod.envelope_from_json
        try:
            classical_mod.envelope_from_json = lambda j: (lambda t: 10.0)
            drive = ClassicalTwoPhotonDrive.from_envelope_json(
                {"dummy": True},
                omega0=2.0,
                detuning=0.0,
                laser_omega=1.0,
                label="ejson",
            )
            self.assertEqual(drive.label, "ejson")
            self.assertEqual(drive.omega0, 2.0)
            self.assertEqual(drive.laser_omega, 1.0)
            coeff = drive.qutip_coeff(time_unit_s=0.25)
            # env = 10.0, Omega = s * omega0 * env = 0.25 * 2.0 * 10.0 = 5.0
            self.assertAlmostEqual(coeff(123.0, {}), 5.0, places=12)
        finally:
            classical_mod.envelope_from_json = original

    def test_with_cached_tlist_and_with_detuning_return_new_instances(self):
        def env(t):
            return 1.0

        drive = ClassicalTwoPhotonDrive(envelope=env, detuning=1.0)

        tlist = np.linspace(0.0, 1.0, 5)
        d2 = drive.with_cached_tlist(tlist)
        self.assertIsNot(drive, d2)
        self.assertTrue(np.array_equal(d2._cached_tlist, tlist))
        self.assertTrue(drive._cached_tlist.size == 0)  # original unchanged

        # Replace detuning with a callable
        def det_fn(t_sec: float) -> float:
            return 0.1 * t_sec

        d3 = drive.with_detuning(det_fn)
        self.assertIsNot(drive, d3)
        self.assertTrue(callable(d3.detuning))
        self.assertEqual(drive.detuning, 1.0)  # original unchanged

    def test_coeff_ignores_args(self):
        def env(t):
            return 2.0

        drive = ClassicalTwoPhotonDrive(envelope=env, omega0=4.0)
        coeff = drive.qutip_coeff(time_unit_s=3.0)
        # Ensure extra args do not matter
        self.assertAlmostEqual(coeff(1.23), 3.0 * 4.0 * 2.0, places=12)
        self.assertAlmostEqual(
            coeff(1.23, {"anything": 1}), 3.0 * 4.0 * 2.0, places=12
        )


if __name__ == "__main__":
    unittest.main()
