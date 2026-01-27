from bec.quantum_dot.adapters.polarization_adapter import QDPolarizationAdapter
from bec.quantum_dot.adapters.transition_registry_adapter import (
    QDTransitionRegistryAdapter,
)
from bec.simulation.drive_decode import DriveDecodeContext
from bec.simulation.drive_decode.decoder import DecodePolicy
from bec.simulation.drive_decode import DefaultDriveDecoder
from bec.simulation.compiler import MECompiler
from bec.quantum_dot.adapters.simulation_provider import QDDecodeContextProvider




def compile_qd(
    *, qd, drives, tlist, time_unit_s, report=False, sample_points=21
):
    decode_ctx = DriveDecodeContext(
        transitions=QDTransitionRegistryAdapter(qd),
        pol=QDPolarizationAdapter(qd),
        phonons=None,
        bandwidth=None,
    )

    decoder = DefaultDriveDecoder(
        policy=DecodePolicy(sample_points=sample_points)
    )

    compiler = MECompiler(decoder=decoder)  # keep compiler QD-free if you want
    return compiler.compile(
        model=qd,  # opaque handle ok
        provider=QDDecodeContextProvider(),
        drives=drives,  # <-- your chirped drive goes here
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0="G",
        report=report,
    )
