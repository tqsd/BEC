import sys
import types
import unittest

import bec.simulation.solvers as solvers
from bec.simulation.solvers import QutipMesolveBackend, MesolveOptions


class TestMesolveOptions(unittest.TestCase):
    def test_as_dict_defaults(self):
        backend = QutipMesolveBackend()
        opts = backend._as_dict()
        self.assertEqual(opts["nsteps"], 10000)
        self.assertAlmostEqual(opts["rtol"], 1e-9)
        self.assertAlmostEqual(opts["atol"], 1e-11)
        self.assertEqual(opts["progress_bar"], "tqdm")
        self.assertFalse(opts["store_final_state"])
        self.assertAlmostEqual(opts["max_step"], 1e-2)

    def test_as_dict_custom(self):
        custom = MesolveOptions(
            nsteps=123,
            rtol=1e-7,
            atol=1e-9,
            progress_bar="enhanced",
            store_final_state=True,
            max_step=0.5,
        )
        backend = QutipMesolveBackend(custom)
        opts = backend._as_dict()
        self.assertEqual(opts["nsteps"], 123)
        self.assertAlmostEqual(opts["rtol"], 1e-7)
        self.assertAlmostEqual(opts["atol"], 1e-9)
        self.assertEqual(opts["progress_bar"], "enhanced")
        self.assertTrue(opts["store_final_state"])
        self.assertAlmostEqual(opts["max_step"], 0.5)


class TestQutipMesolveBackend(unittest.TestCase):
    def setUp(self):
        # Keep a handle to the symbol 'mesolve' inside the module under test.
        self._orig_mesolve = solvers.mesolve

    def tearDown(self):
        # Restore original mesolve symbol.
        solvers.mesolve = self._orig_mesolve

    def test_solve_passes_dict_options_when_supported(self):
        calls = {}

        def fake_mesolve(H, rho0, tlist, c_ops=None, e_ops=None, options=None):
            # Assert that options is a plain dict and contains expected keys.
            self.assertIsInstance(options, dict)
            for k in (
                "nsteps",
                "rtol",
                "atol",
                "progress_bar",
                "store_final_state",
                "max_step",
            ):
                self.assertIn(k, options)
            calls["seen"] = True
            # Return a simple sentinel object
            return {"ok": True, "options": options, "len_t": len(tlist)}

        solvers.mesolve = fake_mesolve

        backend = QutipMesolveBackend(
            MesolveOptions(nsteps=50, store_final_state=True)
        )
        res = backend.solve(H=0, rho0=0, tlist=[0, 1, 2], c_ops=[], e_ops=[])

        self.assertTrue(calls.get("seen", False))
        self.assertEqual(res["len_t"], 3)
        self.assertTrue(res["options"]["store_final_state"])
        self.assertEqual(res["options"]["nsteps"], 50)

    def test_solve_falls_back_to_Options_on_typeerror(self):
        # Install a dummy qutip.Options into sys.modules so the fallback import works.
        dummy_qutip = types.ModuleType("qutip")

        class DummyOptions:
            def __init__(self, **kwargs):
                # keep kwargs for assertions
                self.kwargs = kwargs

        # mesolve here will be overridden in the solvers module below,
        # but the fallback import pulls Options from sys.modules["qutip"].
        dummy_qutip.Options = DummyOptions
        sys.modules["qutip"] = dummy_qutip

        options_seen = {}

        def fake_mesolve_raise(
            H, rho0, tlist, c_ops=None, e_ops=None, options=None
        ):
            # First call (with dict) should raise to trigger fallback.
            if isinstance(options, dict):
                raise TypeError("expects Options")
            # Second call should receive an Options instance
            self.assertIsInstance(options, DummyOptions)
            options_seen["opts"] = options.kwargs
            return {"ok": True}

        # Monkeypatch the symbol used by the backend module
        solvers.mesolve = fake_mesolve_raise

        try:
            backend = QutipMesolveBackend(
                MesolveOptions(nsteps=77, max_step=0.25)
            )
            res = backend.solve(
                H=0, rho0=0, tlist=[0.0, 0.5], c_ops=None, e_ops=None
            )
            self.assertEqual(res["ok"], True)
            self.assertIn("nsteps", options_seen["opts"])
            self.assertEqual(options_seen["opts"]["nsteps"], 77)
            self.assertAlmostEqual(options_seen["opts"]["max_step"], 0.25)
        finally:
            # Clean up the dummy qutip module to avoid side effects
            sys.modules.pop("qutip", None)


if __name__ == "__main__":
    unittest.main()
