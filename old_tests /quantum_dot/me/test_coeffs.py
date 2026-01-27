import unittest

from bec.quantum_dot.me.coeffs import (
    ConstCoeff,
    FuncCoeff,
    as_coeff,
    scale,
    add,
)


class TestCoeffs(unittest.TestCase):
    def test_const_coeff(self):
        c = ConstCoeff(3.5)
        self.assertEqual(c(0.0), 3.5)
        self.assertEqual(c(123.0, args={"x": 1}), 3.5)

    def test_func_coeff_accepts_t_only(self):
        def f(t):
            return 2.0 * t

        c = FuncCoeff(f)
        self.assertEqual(c(2.0), 4.0)
        self.assertEqual(c(2.0, args={"ignored": True}), 4.0)

    def test_func_coeff_accepts_t_args(self):
        def f(t, args):
            return t + float(args["a"])

        c = FuncCoeff(f)
        self.assertEqual(c(2.0, args={"a": 3}), 5.0)

    def test_as_coeff_float(self):
        c = as_coeff(1.25)
        self.assertIsInstance(c, ConstCoeff)
        self.assertEqual(c(0.0), 1.25)

    def test_as_coeff_callable(self):
        c = as_coeff(lambda t: 7.0)
        self.assertIsInstance(c, FuncCoeff)
        self.assertEqual(c(0.0), 7.0)

    def test_scale(self):
        base = as_coeff(lambda t: t)
        c = scale(base, 3.0)
        self.assertAlmostEqual(c(2.0), 6.0)

    def test_add(self):
        c1 = as_coeff(lambda t: 2.0 * t)
        c2 = as_coeff(1.0)
        s = add(c1, c2)
        self.assertAlmostEqual(s(3.0), 7.0)  # 2*3 + 1


if __name__ == "__main__":
    unittest.main()
