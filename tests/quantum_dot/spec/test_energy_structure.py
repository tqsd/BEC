import unittest

from smef.core.units import Q

from bec.quantum_dot.spec.energy_structure import EnergyStructure


class TestEnergyStructure(unittest.TestCase):
    def test_from_levels_valid(self) -> None:
        es = EnergyStructure.from_levels(
            X1=Q(1.30, "eV"),
            X2=Q(1.29, "eV"),
            XX=Q(2.60, "eV"),
        )
        d = es.energies_eV()
        self.assertAlmostEqual(d["X1"], 1.30, places=12)
        self.assertAlmostEqual(d["X2"], 1.29, places=12)
        self.assertAlmostEqual(d["XX"], 2.60, places=12)
        self.assertAlmostEqual(d["G"], 0.0, places=12)

    def test_from_params_consistency(self) -> None:
        exciton = Q(1.30, "eV")
        fss = Q(0.01, "eV")
        binding = Q(0.02, "eV")

        es = EnergyStructure.from_params(
            exciton=exciton, fss=fss, binding=binding
        )

        self.assertAlmostEqual(
            float(es.exciton_center.to("eV").magnitude), 1.30, places=12
        )
        self.assertAlmostEqual(
            float(es.fss.to("eV").magnitude), 0.01, places=12
        )
        self.assertAlmostEqual(
            float(es.binding.to("eV").magnitude), 0.02, places=12
        )

        # Check the construction formulas
        self.assertAlmostEqual(
            float(es.X1.to("eV").magnitude), 1.305, places=12
        )
        self.assertAlmostEqual(
            float(es.X2.to("eV").magnitude), 1.295, places=12
        )
        self.assertAlmostEqual(
            float(es.XX.to("eV").magnitude), 2.0 * 1.30 - 0.02, places=12
        )

    def test_validate_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            EnergyStructure.from_levels(
                X1=Q(0.0, "eV"), X2=Q(1.0, "eV"), XX=Q(2.0, "eV")
            )

        with self.assertRaises(ValueError):
            EnergyStructure.from_levels(
                X1=Q(1.0, "eV"), X2=Q(-1.0, "eV"), XX=Q(2.0, "eV")
            )

        with self.assertRaises(ValueError):
            EnergyStructure.from_levels(
                X1=Q(1.0, "eV"), X2=Q(1.0, "eV"), XX=Q(0.0, "eV")
            )

    def test_validate_rejects_xx_below_exciton(self) -> None:
        with self.assertRaises(ValueError):
            EnergyStructure.from_levels(
                X1=Q(1.2, "eV"), X2=Q(1.1, "eV"), XX=Q(1.2, "eV")
            )

        with self.assertRaises(ValueError):
            EnergyStructure.from_levels(
                X1=Q(1.2, "eV"), X2=Q(1.1, "eV"), XX=Q(1.19, "eV")
            )


if __name__ == "__main__":
    unittest.main()
