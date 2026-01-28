# bec.light.envelopes

Unitful temporal envelopes for classical optical fields.

This module defines **unitful envelope objects** that describe the *shape* of
classical optical pulses in time, independently of any quantum system or solver.

The envelopes are designed to integrate cleanly with the SMEF (Standardized
Master Equation Framework) pipeline while keeping all physical units explicit
until explaining/compilation time.

---

## Design principles

- **Unitful by default**
  - All envelopes accept and store time parameters as `QuantityLike`
    (Pint quantities from `smef.core.units`).
  - Calling an envelope requires a unitful time input.
  - No implicit "seconds" assumptions at the API boundary.

- **Dimensionless output**
  - Envelopes return dimensionless floats.
  - Physical scaling (field amplitudes, dipole moments, phonons, etc.) happens
    outside the envelope layer.

- **Separation of concerns**
  - Envelopes describe *temporal shape only*.
  - They do not know about transitions, detunings, Hamiltonians, or solvers.
  - Conversion to solver-compatible (unitless) callables happens in a dedicated
    compilation step.

- **Serializable**
  - Envelopes implement `to_dict()` / `from_dict()` and are registered via a
    central registry for JSON roundtrips.

Serialization is handled via:

```python
from bec.light.envelopes import envelope_to_json, envelope_from_json
```
---

## Envelope contract

All envelopes implement the following contract:

- Input: `t : QuantityLike` (unitful time)
- Output: `float` (dimensionless)
- Deterministic and side-effect free

## Envelope types

### GaussianEnvelopeU

Peak-normalized Gaussian envelope:

$$ g(t) = \exp\!\left(-\frac{(t - t_0)^2}{2\sigma^2}\right) $$


- Parameters: `t0`, `sigma` (unitful)
- Peak value is exactly 1
- Provides `from_fwhm(...)` and `area_seconds()`

### SymbolicEnvelopeU

User-defined symbolic envelope evaluated via restricted `eval`.

- Expression is evaluated with:
  - `t` as a float in a declared time unit
  - `np` (NumPy) and `math`
  - user-supplied dimensionless parameters
- Explicit `t_unit` declaration avoids hidden assumptions
- Evaluation is sandboxed (`__builtins__` removed)

Example:

```python
env = SymbolicEnvelopeU(
    expr="np.exp(-(t - t0)**2 / (2*sigma**2))",
    params={"t0": 10.0, "sigma": 2.0},
    t_unit="ps",
)
```
### TabulatedEnvelopeU
Piecewise-linear envelope defined by sampled values.
- Sample times are unitful at the API level
- Internally stored as seconds floats for fast interpolation
- Supports clamping outside the sampled range
- Explicit `t_unit` for serialization/readability

