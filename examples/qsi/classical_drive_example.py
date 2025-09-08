"""
Example 01
-------------------
Example with:
"""

import numpy as np
import uuid
from qsi.coordinator import Coordinator
from qsi.state import State, StateProp

from bec.light.envelopes import gaussian_to_tabulated


# Initiate the Coordinator
coordinator = Coordinator()

# Start the module processes, before running the coordinator process
qd_source = coordinator.register_component(
    module="../../qsi/qd_source.py", runtime="python"
)

# Run the coordinator, when the coordinator starts, it automatically queries
# all modules for the possible parameters to set. ('param_query')
coordinator.run()

# Set the parameters of the devices one by one
# This is an example, this value is not actually used in the simulation
qd_source.set_param("exciton_eV", 1.35)
qd_source.set_param("biexciton_eV", 2.7)
qd_source.set_param("fss_eV", 0.1)
qd_source.set_param("dipole_moment_Cm", 3.0e-29)
qd_source.set_param("cavity_Q", 1e4)
qd_source.set_param("cavity_V_eff_um3", 1.0)
qd_source.set_param("cavity_lambda_nm", 920.0)
qd_source.set_param("cavity_n", 3.5)

# Set all configured parameters to the device ('param_set')
qd_source.send_params()

# Initialize internal states if device has any ('state_init')


state_one = State(
    StateProp(
        state_type="light",
        truncation=1,
        wavelength=1550,
        polarization="R",
        bandwidth=1,
    )
)
env_json = gaussian_to_tabulated(
    t0=2e-9, sigma=0.5e-9, area=np.pi, nsigma=6.0, num=401, to_json=True
)
drive = {
    "omega0": 2e9,  # rad/s
    "detuning": 0.0,  # rad/s
    "label": "tabulated π-pulse",
    "envelope": env_json,
}
response, operators = qd_source.channel_query(
    state_one,
    {"input": state_one.state_props[0].uuid},
    signals=[drive],
)
print(response)


coordinator.terminate()
