# `simulation/` Simulation Core

This module implements the simulation with the qutip `mesolve` and 
it conveniently wraps the functionalities together so the 
Quantum Dot model fits in. The engine is a facade which 
brings together the different modules.

## What is here?
- `collapse_composer.py` - Implementation for calling qd collapse operator construction
- `engine.py` - Engine Facade
- `expectation_layouts.py` - Expectation Layout for plotting
- `hamiltonian_composer.py` - Implementation for calling qd Hamiltonian terms
- `observable_composer.py` - Implements the observable operators, for plotting
- `protocols.py` - Enforces the implementations for the facade
- `qd_traces.py` - Constructs a dataclass, for uniform handling of the results (for plotting)
- `scenarios.py` - Constructs different simulation scenarios
- `solvers.py` - Implements the solvers (only `mesolve` available)
- `space_builder.py` - Builds the space for the simulation (Full Hilbert space in initial state)

