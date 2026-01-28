import unittest

from smef.core.units import Q

from bec.light.envelopes.base import TimeBasisU


class TestTimeBasisU(unittest.TestCase):
    def test_unit_must_be_quantity(self) -> None:
        with self.assertRaises(TypeError):
            TimeBasisU(unit=1.0)  # not QuantityLike

    def test_unit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TimeBasisU(unit=Q(0.0, "s"))
        with self.assertRaises(ValueError):
            TimeBasisU(unit=Q(-1.0, "ps"))

    def test_conversions_roundtrip(self) -> None:
        basis = TimeBasisU(Q(1.0, "ps"))

        x = basis.to_dimless(Q(3.0, "ps"))
        self.assertAlmostEqual(x, 3.0, places=12)

        t = basis.to_phys(3.0)
        self.assertAlmostEqual(float(t.to("ps").magnitude), 3.0, places=12)

    def test_to_dimless_requires_time(self) -> None:
        basis = TimeBasisU(Q(1.0, "ps"))
        # Pint will raise when trying to convert incompatible dims
        with self.assertRaises(Exception):
            basis.to_dimless(Q(1.0, "m"))


if __name__ == "__main__":
    unittest.main()
