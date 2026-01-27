from bec.simulation.compiler import MECompiler
from bec.quantum_dot.adapters.simulation_provider import QDDecodeContextProvider


def compile_qd(*, qd, drives, tlist, time_unit_s, **kw):
    return MECompiler().compile(
        model=qd,
        provider=QDDecodeContextProvider(),
        drives=drives,
        tlist=tlist,
        time_unit_s=time_unit_s,
        **kw,
    )
