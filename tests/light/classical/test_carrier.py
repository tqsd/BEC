import unittest

from smef.core.units import Q

from bec.light.classical.carrier import Carrier


class TestCarrier(unittest.TestCase):
    def test_constant_delta_omega(self) -> None:
        c = Carrier(omega0=Q(10.0, "rad/s"), delta_omega=Q(2.5, "rad/s"))
        w = c.omega_phys(Q(0.0, "s"))
        self.assertAlmostEqual(float(w.to("rad/s").magnitude), 12.5, places=12)
        self.assertAlmostEqual(c.omega_rad_s(Q(0.0, "s")), 12.5, places=12)

    def test_delta_omega_callable(self) -> None:
        def dw(t):
            # t is QuantityLike seconds
            return Q(2.0, "rad/s") * 0.0 + Q(
                float(t.to("s").magnitude), "rad/s"
            )

        c = Carrier(omega0=Q(1.0, "rad/s"), delta_omega=dw)
        w = c.omega_phys(Q(3.0, "s"))
        self.assertAlmostEqual(float(w.to("rad/s").magnitude), 4.0, places=12)

    def test_to_dict_rejects_callable(self) -> None:
        c = Carrier(
            omega0=Q(1.0, "rad/s"), delta_omega=lambda t: Q(0.0, "rad/s")
        )
        with self.assertRaises(TypeError):
            _ = c.to_dict()

    def test_to_dict_from_dict_roundtrip(self) -> None:
        c = Carrier(
            omega0=Q(10.0, "rad/s"),
            delta_omega=Q(2.0, "rad/s"),
            phi0=1.25,
            label="car",
        )
        d = c.to_dict()
        c2 = Carrier.from_dict(d)

        self.assertEqual(c2.label, "car")
        self.assertAlmostEqual(c2.phi0, 1.25, places=12)
        self.assertAlmostEqual(c2.omega_rad_s(Q(0.0, "s")), 12.0, places=12)

    def test_incompatible_units_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = Carrier(omega0=Q(1.0, "s"))


if __name__ == "__main__":
    unittest.main()
