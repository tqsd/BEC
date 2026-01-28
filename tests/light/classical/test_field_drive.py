import unittest

import numpy as np

from smef.core.units import Q, magnitude

from bec.light.classical.amplitude import FieldAmplitude
from bec.light.classical.carrier import Carrier
from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.light.core.polarization import JonesMatrix, JonesState
from bec.light.envelopes.gaussian import GaussianEnvelopeU


class TestClassicalFieldDriveU(unittest.TestCase):
    def assertComplexAllClose(
        self, a: np.ndarray, b: np.ndarray, tol: float = 1e-12
    ) -> None:
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(float(np.max(np.abs(a - b))), float(tol))

    def test_E_env_phys_units(self) -> None:
        env = GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(1.0, "s"))
        amp = FieldAmplitude(E0=Q(2.0, "V/m"))
        d = ClassicalFieldDriveU(envelope=env, amplitude=amp)

        E0 = d.E_env_phys(Q(0.0, "s"))
        self.assertAlmostEqual(float(magnitude(E0, "V/m")), 2.0, places=12)

        # float time interpreted as seconds
        E1 = d.E_env_phys(0.0)
        self.assertAlmostEqual(float(magnitude(E1, "V/m")), 2.0, places=12)

    def test_omega_L_phys_optional(self) -> None:
        env = GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(1.0, "s"))
        amp = FieldAmplitude(E0=Q(1.0, "V/m"))

        d0 = ClassicalFieldDriveU(envelope=env, amplitude=amp)
        self.assertIsNone(d0.omega_L_phys(Q(0.0, "s")))

        c = Carrier(omega0=Q(5.0, "rad/s"))
        d1 = ClassicalFieldDriveU(envelope=env, amplitude=amp, carrier=c)
        w = d1.omega_L_phys(Q(0.0, "s"))
        self.assertIsNotNone(w)
        self.assertAlmostEqual(float(magnitude(w, "rad/s")), 5.0, places=12)

    def test_effective_pol(self) -> None:
        env = GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(1.0, "s"))
        amp = FieldAmplitude(E0=Q(1.0, "V/m"))

        d0 = ClassicalFieldDriveU(envelope=env, amplitude=amp)
        self.assertIsNone(d0.effective_pol())

        # H rotated by 90 deg -> V
        s = JonesState.H()
        R = JonesMatrix.rotation(np.pi / 2.0)
        d1 = ClassicalFieldDriveU(
            envelope=env, amplitude=amp, pol_state=s, pol_transform=R
        )
        E = d1.effective_pol()
        self.assertIsNotNone(E)
        self.assertComplexAllClose(E, np.array([0.0 + 0j, 1.0 + 0j]))

    def test_serialization_roundtrip(self) -> None:
        env = GaussianEnvelopeU(t0=Q(3.0, "ps"), sigma=Q(2.0, "ps"))
        amp = FieldAmplitude(E0=Q(7.0, "V/m"), label="amp")
        c = Carrier(omega0=Q(9.0, "rad/s"), phi0=0.25, label="car")
        s = JonesState.D()
        M = JonesMatrix.qwp(theta_rad=0.1)

        d = ClassicalFieldDriveU(
            envelope=env,
            amplitude=amp,
            carrier=c,
            pol_state=s,
            pol_transform=M,
            preferred_kind="1ph",
            label="drive",
        )

        blob = d.to_dict()
        d2 = ClassicalFieldDriveU.from_dict(blob)

        self.assertEqual(d2.label, "drive")
        self.assertEqual(d2.preferred_kind, "1ph")
        self.assertAlmostEqual(d2.amplitude.E0_V_m(), 7.0, places=12)
        self.assertIsNotNone(d2.carrier)
        self.assertAlmostEqual(
            float(magnitude(d2.carrier.omega0, "rad/s")), 9.0, places=12
        )

        # Compare effective pol (should match)
        E1 = d.effective_pol()
        E2 = d2.effective_pol()
        self.assertIsNotNone(E1)
        self.assertIsNotNone(E2)
        self.assertComplexAllClose(E1, E2)


if __name__ == "__main__":
    unittest.main()
