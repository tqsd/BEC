import unittest
import math

from bec.quantum_dot.metrics.overlap import OverlapCalculator


class TestOverlapCalculator(unittest.TestCase):
    def setUp(self):
        # Choose simple numbers to make expectations clear
        # early linewidths: 6 and 10   -> gamma_early = 0.5*(6+10)=8
        # late  linewidths: 2 and  4   -> gamma_late  = 0.5*(2+4)=3
        # avg: g1 = 0.5*(X1_G + XX_X1) = 0.5*(2+6)=4
        #      g2 = 0.5*(X2_G + XX_X2) = 0.5*(4+10)=7
        #      gamma_avg = 0.5*(4+7)=5.5
        self.oc = OverlapCalculator(
            gamma_XX_X1=6.0,
            gamma_XX_X2=10.0,
            gamma_X1_G=2.0,
            gamma_X2_G=4.0,
            delta_rad_s=5.0,
        )

    def test_pair_linewidths(self):
        self.assertAlmostEqual(self.oc._pair_linewidth("early"), 8.0, places=12)
        self.assertAlmostEqual(self.oc._pair_linewidth("late"), 3.0, places=12)
        self.assertAlmostEqual(self.oc._pair_linewidth("avg"), 5.5, places=12)

    def test_overlap_formula_matches_definition(self):
        # late branch: gamma=3, delta=5 -> 3 / sqrt(3^2 + 5^2)
        expected_late = 3.0 / math.hypot(3.0, 5.0)
        self.assertAlmostEqual(
            self.oc.overlap("late"), expected_late, places=12
        )

        # early branch: gamma=8, delta=5
        expected_early = 8.0 / math.hypot(8.0, 5.0)
        self.assertAlmostEqual(
            self.oc.overlap("early"), expected_early, places=12
        )

        # avg branch: gamma=5.5, delta=5
        expected_avg = 5.5 / math.hypot(5.5, 5.0)
        self.assertAlmostEqual(self.oc.overlap("avg"), expected_avg, places=12)

    def test_overlap_edge_cases(self):
        # zero gamma -> 0 overlap regardless of delta
        oc_zero = OverlapCalculator(0.0, 0.0, 0.0, 0.0, 7.0)
        self.assertEqual(oc_zero.overlap("late"), 0.0)
        self.assertEqual(oc_zero.overlap("early"), 0.0)
        self.assertEqual(oc_zero.overlap("avg"), 0.0)

        # zero delta -> overlap = 1 for any positive gamma
        oc_delta0 = OverlapCalculator(1.0, 1.0, 1.0, 1.0, 0.0)
        self.assertEqual(oc_delta0.overlap("late"), 1.0)
        self.assertEqual(oc_delta0.overlap("early"), 1.0)
        self.assertEqual(oc_delta0.overlap("avg"), 1.0)

    def test_hom_is_overlap_squared(self):
        val = self.oc.overlap("late")
        self.assertAlmostEqual(self.oc.hom("late"), val * val, places=12)
        val_e = self.oc.overlap("early")
        self.assertAlmostEqual(self.oc.hom("early"), val_e * val_e, places=12)
        val_a = self.oc.overlap("avg")
        self.assertAlmostEqual(self.oc.hom("avg"), val_a * val_a, places=12)


if __name__ == "__main__":
    unittest.main()
