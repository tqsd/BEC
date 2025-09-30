# `light/` - Light modes and drives

This modulle contains building block for representing optical modes.

## What is here?
- `light_mode.py` - Single optical mode associated with a quantum dot transition
- `envelopes.py` - Time-domain envelopes for light pulses
- `classical.py` - `ClassicalTwoPhotonDrive`, which wraps an envelope with scaling and detuning
- `detuning.py` - Utilities for computing effective two-photon detuning profiles
