import unittest

import numpy as np

from smef.core.units import Q, magnitude

from bec.light.classical.factories import gaussian_field_drive
from bec.light.core.polarization import JonesMatrix, JonesState


class TestClassicalFactories(unittest.TestCase):
    def test_gaussian_requires_exactly_one_width(self) -> None:
        with self.assertRaises(ValueError):
            _ = gaussian_field_drive(t0=Q(0.0, "s"), E0=Q(1.0, "V/m"))

        with self.assertRaises(ValueError):
            _ = gaussian_field_drive(
                t0=Q(0.0, "s"),
                sigma=Q(1.0, "ps"),
                fwhm=Q(2.0, "ps"),
                E0=Q(1.0, "V/m"),
            )

    def test_gaussian_peak_amplitude(self) -> None:
        d = gaussian_field_drive(
            t0=Q(0.0, "s"),
            sigma=Q(2.0, "s"),
            E0=Q(3.0, "V/m"),
        )
        E = d.E_env_phys(Q(0.0, "s"))
        self.assertAlmostEqual(float(magnitude(E, "V/m")), 3.0, places=12)

    def test_gaussian_with_omega0(self) -> None:
        d = gaussian_field_drive(
            t0=Q(0.0, "s"),
            sigma=Q(1.0, "ps"),
            E0=Q(1.0, "V/m"),
            omega0=Q(5.0, "rad/s"),
            delta_omega=Q(2.0, "rad/s"),
        )
        w = d.omega_L_phys(Q(0.0, "s"))
        self.assertIsNotNone(w)
        self.assertAlmostEqual(float(magnitude(w, "rad/s")), 7.0, places=12)

    def test_gaussian_with_wavelength(self) -> None:
        d = gaussian_field_drive(
            t0=Q(0.0, "s"),
            sigma=Q(1.0, "ps"),
            E0=Q(1.0, "V/m"),
            wavelength=Q(1550.0, "nm"),
        )
        w = d.omega_L_phys(Q(0.0, "s"))
        self.assertIsNotNone(w)
        self.assertGreater(float(magnitude(w, "rad/s")), 0.0)

    def test_gaussian_with_energy(self) -> None:
        d = gaussian_field_drive(
            t0=Q(0.0, "s"),
            sigma=Q(1.0, "ps"),
            E0=Q(1.0, "V/m"),
            energy=Q(1.0, "eV"),
        )
        w = d.omega_L_phys(Q(0.0, "s"))
        self.assertIsNotNone(w)
        self.assertGreater(float(magnitude(w, "rad/s")), 0.0)

    def test_pol_passthrough(self) -> None:
        s = JonesState.H()
        R = JonesMatrix.rotation(np.pi / 2.0)

        d = gaussian_field_drive(
            t0=Q(0.0, "s"),
            sigma=Q(1.0, "ps"),
            E0=Q(1.0, "V/m"),
            pol_state=s,
            pol_transform=R,
        )
        E = d.effective_pol()
        self.assertIsNotNone(E)
        self.assertLessEqual(
            float(np.max(np.abs(E - np.array([0.0 + 0j, 1.0 + 0j])))), 1e-12
        )


if __name__ == "__main__":
    unittest.main()
