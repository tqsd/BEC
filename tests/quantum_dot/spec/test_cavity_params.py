import unittest

from smef.core.units import Q

from bec.quantum_dot.spec.cavity_params import CavityParams


class TestCavityParams(unittest.TestCase):
    def test_from_values_accepts_bare_numbers(self) -> None:
        cp = CavityParams.from_values(
            Q=10000.0, Veff_um3=0.5, lambda_nm=930.0, n=3.5
        )
        d = cp.as_floats()
        self.assertAlmostEqual(d["Q"], 10000.0, places=12)
        self.assertAlmostEqual(d["n"], 3.5, places=12)
        self.assertGreater(d["Veff_m3"], 0.0)
        self.assertGreater(d["lambda_m"], 0.0)

    def test_from_values_accepts_quantities(self) -> None:
        cp = CavityParams.from_values(
            Q=20000.0,
            Veff_um3=Q(1.2, "um**3"),
            lambda_nm=Q(950.0, "nm"),
            n=Q(3.4, ""),  # dimensionless quantity is fine if pint supports it
        )
        self.assertAlmostEqual(
            float(cp.lambda_cav.to("nm").magnitude), 950.0, places=12
        )
        self.assertAlmostEqual(
            float(cp.Veff.to("um**3").magnitude), 1.2, places=12
        )
        self.assertAlmostEqual(float(cp.n), 3.4, places=12)

    def test_validate_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            CavityParams.from_values(
                Q=0.0, Veff_um3=1.0, lambda_nm=930.0, n=3.5
            )

        with self.assertRaises(ValueError):
            CavityParams.from_values(
                Q=1000.0, Veff_um3=1.0, lambda_nm=930.0, n=0.0
            )

        with self.assertRaises(ValueError):
            CavityParams.from_values(
                Q=1000.0, Veff_um3=0.0, lambda_nm=930.0, n=3.5
            )

        with self.assertRaises(ValueError):
            CavityParams.from_values(
                Q=1000.0, Veff_um3=1.0, lambda_nm=0.0, n=3.5
            )


if __name__ == "__main__":
    unittest.main()
