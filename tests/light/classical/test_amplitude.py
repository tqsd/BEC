import unittest

from smef.core.units import Q

from bec.light.classical.amplitude import FieldAmplitude


class TestFieldAmplitude(unittest.TestCase):
    def test_converts_and_caches(self) -> None:
        amp = FieldAmplitude(E0=Q(2.0, "V/m"))
        self.assertAlmostEqual(
            float(amp.E0.to("V/m").magnitude), 2.0, places=12
        )
        self.assertAlmostEqual(amp.E0_V_m(), 2.0, places=12)

    def test_accepts_convertible_units(self) -> None:
        # 1 kV/m = 1000 V/m
        amp = FieldAmplitude(E0=Q(1.0, "kV/m"))
        self.assertAlmostEqual(amp.E0_V_m(), 1000.0, places=12)

    def test_rejects_incompatible_units(self) -> None:
        with self.assertRaises(TypeError):
            _ = FieldAmplitude(E0=Q(1.0, "s"))

    def test_to_dict_from_dict_roundtrip(self) -> None:
        amp = FieldAmplitude(E0=Q(3.5, "V/m"), label="drive_amp")
        d = amp.to_dict()
        amp2 = FieldAmplitude.from_dict(d)

        self.assertEqual(amp2.label, "drive_amp")
        self.assertAlmostEqual(amp2.E0_V_m(), 3.5, places=12)


if __name__ == "__main__":
    unittest.main()
