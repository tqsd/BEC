from bec.params.transitions import TransitionType as _tt
import unittest
import numpy as np
from qutip import Qobj
import importlib

from bec.simulation.engine import SimulationEngine, SimulationConfig


class FakeScenario:
    def __init__(self, drive=None):
        self._drive = drive
        self.prepared = False

    def prepare(self, qd):
        self.prepared = True

    def classical_drive(self):
        return self._drive


class FakeDrive:
    """Provides the minimal API used by SimulationEngine."""

    def __init__(self, omega0=1.0, env=None):
        self.omega0 = float(omega0)
        self._tlist = None
        self._det = None
        self.envelope = env or (lambda t: 1.0)

    def qutip_coeff(self, *, time_unit_s: float = 1.0):
        # Omega_solver(t') = s * omega0 * env(s*t')
        s = float(time_unit_s)
        om0 = self.omega0
        env = self.envelope

        def coeff(t, args=None):
            return s * om0 * float(env(s * float(t)))

        return coeff

    def with_cached_tlist(self, tlist):
        self._tlist = np.array(tlist, copy=True)
        return self

    def with_detuning(self, det_fn):
        # store callables for inspection if desired
        self._det = det_fn
        return self


class FakeModes:
    def __init__(self):
        # minimal mode objects with label, label_tex, and source
        class M:
            pass

        m_ext = M()
        m_ext.label = "EXT"
        m_ext.label_tex = "EXT"
        m_ext.source = _tt.EXTERNAL
        m_int = M()
        m_int.label = "G_X"
        m_int.label_tex = "G\\to X"
        m_int.source = _tt.INTERNAL
        self.modes = [m_ext, m_int]


class FakeQD:
    """Bare-minimum facade with modes and a dot placeholder."""

    def __init__(self):
        self.modes = FakeModes()
        self.dot = object()  # not used by tests
        self._EL = object()  # energy levels placeholder
        self.N_cut = None


class FakeSpaceBuilder:
    def build_space(self, qd, trunc_per_pol):
        # envs, focks, cstate: only cstate is used by build_qutip_space
        return ["envs"], ["focks"], "cstate"

    def build_qutip_space(self, cstate, dot, focks):
        # Build a 2x3 bipartite space so ptrace works (dims [2,3])
        dim_qd, dim_phot = 2, 3
        dim_total = dim_qd * dim_phot
        rho0 = Qobj(
            np.eye(dim_total) / dim_total,
            dims=[[dim_qd, dim_phot], [dim_qd, dim_phot]],
        )
        dims = [dim_qd, dim_phot]
        dims2 = dims
        return dims, dims2, rho0


class FakeHamiltonianComposer:
    def compose(self, qd, dims, drive, time_unit_s=1.0):
        # Return a simple H (identity)
        dim_total = np.prod(dims)
        return Qobj(np.eye(dim_total))


class FakeCollapseComposer:
    def compose(self, qd, dims, time_unit_s=1.0):
        # Return an empty list (no collapses)
        return []


class FakeObservableComposer:
    def compose_qd(self, qd, dims, time_unit_s=1.0):
        # Provide projectors P_G, P_X1, P_X2, P_XX (labels used by layout)
        # Use 1x1 identities; actual shapes are irrelevant for this test
        return {
            k: Qobj(np.array([[1.0]])) for k in ("P_G", "P_X1", "P_X2", "P_XX")
        }

    def compose_lm(self, qd, dims, time_unit_s=1.0):
        # Layout will look for keys N[EXT], N-[EXT], N+[EXT], N[G_X], ...
        keys = ["N[EXT]", "N-[EXT]", "N+[EXT]", "N[G_X]", "N-[G_X]", "N+[G_X]"]
        return {k: Qobj(np.array([[1.0]])) for k in keys}


class FakeExpectationLayout:
    def select(self, qd_proj, lm_proj, qd):
        # Mimic DefaultExpectationLayout: return ordered list and slice map
        qd_eops = [qd_proj[k] for k in ("P_G", "P_X1", "P_X2", "P_XX")]
        fly_T = [lm_proj["N[EXT]"]]
        fly_H = [lm_proj["N-[EXT]"]]
        fly_V = [lm_proj["N+[EXT]"]]
        out_T = [lm_proj["N[G_X]"]]
        out_H = [lm_proj["N-[G_X]"]]
        out_V = [lm_proj["N+[G_X]"]]
        e_ops = [*qd_eops, *fly_T, *fly_H, *fly_V, *out_T, *out_H, *out_V]
        # Build index slices
        i = 0
        idx = {}
        idx["qd"] = slice(i, i + len(qd_eops))
        i += len(qd_eops)
        idx["fly_T"] = slice(i, i + len(fly_T))
        i += len(fly_T)
        idx["fly_H"] = slice(i, i + len(fly_H))
        i += len(fly_H)
        idx["fly_V"] = slice(i, i + len(fly_V))
        i += len(fly_V)
        idx["out_T"] = slice(i, i + len(out_T))
        i += len(out_T)
        idx["out_H"] = slice(i, i + len(out_H))
        i += len(out_H)
        idx["out_V"] = slice(i, i + len(out_V))
        i += len(out_V)
        return e_ops, idx


class FakeSolveResult:
    def __init__(self, expect, final_state=None):
        self.expect = expect
        self.final_state = final_state


class FakeSolver:
    def __init__(self):
        self.last_call = None

    def solve(self, H, rho0, tlist, c_ops, e_ops):
        self.last_call = (H, rho0, tlist, c_ops, e_ops)
        # Build deterministic expectation outputs: one array per e_op
        # length = len(tlist)
        n_ops = len(e_ops)
        T = len(tlist)
        expect = [np.full(T, float(k)) for k in range(n_ops)]
        # Return rho0 as "final_state" with dims intact so ptrace works
        return FakeSolveResult(expect=expect, final_state=rho0)


# Import enums after we load tests to avoid circulars


class SimulationEngineTests(unittest.TestCase):
    def setUp(self):
        # Monkeypatch the detuning function in the engine module to a flat zero
        import bec.simulation.engine as eng_mod

        self._eng_mod = eng_mod
        self._orig_det = eng_mod.two_photon_detuning_profile

        def fake_detuning_profile(EL, drive, time_unit_s):
            # Return the cached tlist and zero detuning in solver units
            t = getattr(drive, "_tlist", None)
            if t is None:
                return None, None
            return t, np.zeros_like(t, dtype=float)

        eng_mod.two_photon_detuning_profile = fake_detuning_profile

    def tearDown(self):
        # Restore original detuning function
        self._eng_mod.two_photon_detuning_profile = self._orig_det

    def test_run_with_state_orchestrates_and_packs(self):
        qd = FakeQD()
        drive = FakeDrive(omega0=2.0, env=lambda t: 1.0 + 0.0 * t)
        scenario = FakeScenario(drive)
        space = FakeSpaceBuilder()
        hams = FakeHamiltonianComposer()
        collapses = FakeCollapseComposer()
        observables = FakeObservableComposer()
        layout = FakeExpectationLayout()
        solver = FakeSolver()

        eng = SimulationEngine(
            space=space,
            hams=hams,
            collapses=collapses,
            observables=observables,
            layout=layout,
            solver=solver,
        )

        tlist = np.linspace(0.0, 1.0, 5)
        cfg = SimulationConfig(tlist=tlist, trunc_per_pol=3, time_unit_s=2.0)

        traces, rho_final, rho_phot = eng.run_with_state(
            qd, scenario, cfg, reduce_photonic=True
        )

        # Scenario was prepared and solver invoked
        self.assertTrue(scenario.prepared)
        self.assertIsNotNone(solver.last_call)

        # Traces: shapes and labels
        self.assertTrue(np.array_equal(traces.t, tlist))
        self.assertEqual(traces.time_unit_s, 2.0)
        self.assertTrue(traces.classical)
        self.assertEqual(traces.flying_labels, ["EXT"])
        self.assertEqual(traces.intrinsic_labels, ["G_X"])
        self.assertEqual(len(traces.qd), 4)  # P_G, P_X1, P_X2, P_XX
        self.assertEqual(len(traces.fly_H), 1)
        self.assertEqual(len(traces.fly_V), 1)
        self.assertEqual(len(traces.out_H), 1)
        self.assertEqual(len(traces.out_V), 1)

        # Drive diagnostics present and sized to tlist
        self.assertEqual(traces.omega.shape, tlist.shape)
        self.assertEqual(traces.area.shape, tlist.shape)

        # Final states are returned; photonic reduction exists
        self.assertIsInstance(rho_final, Qobj)
        self.assertIsInstance(rho_phot, Qobj)
        # The kept subsystem is the photonic one with dim 3
        self.assertEqual(rho_phot.dims[0], [3])
        self.assertEqual(rho_phot.dims[1], [3])

    def test_run_without_drive_skips_omega_area(self):
        qd = FakeQD()
        scenario = FakeScenario(drive=None)
        eng = SimulationEngine(
            space=FakeSpaceBuilder(),
            hams=FakeHamiltonianComposer(),
            collapses=FakeCollapseComposer(),
            observables=FakeObservableComposer(),
            layout=FakeExpectationLayout(),
            solver=FakeSolver(),
        )
        tlist = np.linspace(0.0, 1.0, 3)
        cfg = SimulationConfig(tlist=tlist, time_unit_s=1.0)

        traces, rho_final, rho_phot = eng.run_with_state(
            qd, scenario, cfg, reduce_photonic=False
        )

        self.assertFalse(traces.classical)
        self.assertIsNone(traces.omega)
        self.assertIsNone(traces.area)
        self.assertIsNotNone(rho_final)
        self.assertIsNone(rho_phot)  # reduction disabled


if __name__ == "__main__":
    unittest.main()
