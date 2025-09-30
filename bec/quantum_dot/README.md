# `quantum_dot/` - QD module

This module contains a small, testable toolkit for simulating a 4-level quantum dot coupled to light modes.
It is organized around a facade (`dot.py`) that wires together builder/registry classes with single responsibilities.

## What is here
- `dot.py` - Facade `QuantumDot` class. This class orchestrates everything:
  - builds mode registry, symbolic context,
  - builds Hamiltonians, decay rates, and collapse ops
  - exposes quantum dot and light-mode observables
  - provides diagnostic helpers
- `mode_registry.py` - Keeps intrinsic and external (not used in this model) `LightMode`s exposing lookup by transition and source
- `context_builder.py` - Builds the symbolic operator context used by the `photon_weave.extras.interpreter`
- `kron_pad_utility.py` - Pads local ops into the full Hilbert space constructing an operator for the whole Hilbert space
- `hamiltonian_builder.py` - Constructs Hamiltonian terms (FSS, classical two-photon flip/detuning)
- `decay_model.py` - Computes the radiative rates (free space, Purcel) from energies, cavity and dipole.
- `collapse_builder.py` - Turns rate dict and registry into a QuTiP collapse operators
- `observables_builder.py` - QD projectors and photonic mode projectors
- `diagnostics.py` - Read-only metrics: overlaps, HOM visibility, central frequencise, bandwidths, simple entanglement and purity summaries
- `helpers.py` - Small utilities
- `metrics/` - Diagnostic/Analysis helpers
- `protocils.py` - Minimal Interfaces used for type-safe composition
