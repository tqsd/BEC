from __future__ import annotations

import unittest
import numpy as np

from bec.units import Q, magnitude
from bec.light.classical import gaussian_field_drive
from bec.light.classical.compile import compile_drive


class TestClassicalGaussianFieldDrive(unittest.TestCase):
    def test_gaussian_drive_peak_and_carrier_units(self) -> None:
        # Arrange
        t0 = Q(50.0, "ps")
        fwhm = Q(20.0, "ps")
        E0 = Q(5e4, "V/m")
        lam = Q(930.0, "nm")
        time_unit = Q(1.0, "ps")
        time_unit_s = float(time_unit.to("s").magnitude)

        drv = gaussian_field_drive(
            t0=t0,
            fwhm=fwhm,
            E0=E0,
            wavelength=lam,
            delta_omega=Q(0.0, "rad/s"),
            label="pump",
        )

        # Basic unit sanity: these should be quantities and convertible
        self.assertAlmostEqual(magnitude(E0, "V/m"), 5e4, places=12)
        self.assertAlmostEqual(magnitude(lam, "nm"), 930.0, places=12)
        self.assertAlmostEqual(magnitude(t0, "ps"), 50.0, places=12)
        self.assertAlmostEqual(magnitude(fwhm, "ps"), 20.0, places=12)

        # Act
        compiled = compile_drive(drv, time_unit_s=time_unit_s)

        # 1) Peak-normalized envelope should produce E(t0) == E0 (in solver units)
        # Here: t_solver=50 corresponds to 50 ps because time_unit = 1 ps.
        E_peak = compiled.E_env_V_m(50.0)
        self.assertAlmostEqual(E_peak, float(magnitude(E0, "V/m")), places=9)

        # 2) Field far away from the pulse should be small
        E_far = compiled.E_env_V_m(0.0)
        self.assertLess(E_far, 1e-6 * float(magnitude(E0, "V/m")))

        # 3) Carrier: omega_L_solver should exist and be consistent with wavelength
        self.assertIsNotNone(compiled.omega_L_solver)
        w_solver = float(compiled.omega_L_solver(50.0))  # dimensionless
        self.assertGreater(w_solver, 1.0)

        # Convert back to physical omega (rad/s)
        w_phys = w_solver / time_unit_s  # rad/s

        # Infer wavelength from omega: omega = 2*pi*c/lambda
        c = 299_792_458.0  # m/s
        f = w_phys / (2.0 * np.pi)  # Hz
        lam_inferred_m = c / f
        lam_inferred_nm = lam_inferred_m * 1e9

        self.assertAlmostEqual(
            lam_inferred_nm, float(magnitude(lam, "nm")), places=3
        )

        # 4) Envelope symmetry check around center for a Gaussian:
        # Pick +/- 10 ps around t0, in solver units (ps units)
        v_plus = compiled.E_env_V_m(60.0)
        v_minus = compiled.E_env_V_m(40.0)
        self.assertAlmostEqual(v_plus, v_minus, places=9)


if __name__ == "__main__":
    unittest.main()
