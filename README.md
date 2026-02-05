# BEC: Biexciton-Exciton Cascade Simulation framework

## Features
- Light modes are represented with wavelength/enegy (convertible) and time envelopes (Gaussian, tabulated, symbolic expression)
- Classical two-photon drive with QuTiP-compatible coefficients
- Quantum Dot energy levels, transition registry, and mode construction
- Composers for Hamiltonians, Collapse operators, Observables and expectation operators
- A minimal engine with `QuTiP`s `mesolve` returning structured traces
- Focus on unit tests with small reusable components

## Install
Install the dependencies with:
```bash
python -m pip install -e .
```

## Tests
Run the tests with:
```bash
pytest -q
```

## QSI implementation
This library provides a wrapper for the `QSI` library in `qsi/qd_source.py` together with helpers to decode the application of the 
channel into a mode representation with the correct properties (wavelength, polarization).

An example of how to use the wrapper is implemented in the `examples/qsi/simple_example.py`.

To run the example:
```bash
python classical_drive_example.py <port>
```
where the port can be any currently unused port.


## Paper results
The results used in the paper 'Entangled Photon Pair Generator via Biexciton-Exciton Cascade in Semiconductor Quantum Dots and its Simulation' are stored in `example/paper` directory.

