# Biexciton–Exciton Cascade: Plots & CSV Exports

This directory contains simulation outputs (plots and CSV exports) for the biexciton–exciton cascade in a semiconductor quantum dot.

## Contents

- `fss.py`  
  The driver script. Configures the quantum dot, sets cavity and dipole parameters, runs three drive scenarios (π-pulse, 5π-pulse, detuned π-pulse), generates plots, and exports CSVs.

- `fss_0.pdf`, `fss_5.pdf`  
  Plot PDFs for two values of the fine-structure splitting (FSS):  
  - **fss_0.pdf** → FSS = 0 µeV (degenerate exciton doublet)  
  - **fss_5.pdf** → FSS = 5 µeV (finite splitting)

- `fss_0_pi_pulse.csv`, `fss_0_5pi_pulse.csv`, `fss_0_detuned_pi_pulse.csv`  
  CSV exports for the three drive scenarios at FSS = 0 µeV. (Missing files will appear once the corresponding runs are generated.)

> Filenames are constructed from the FSS value (in integer µeV):  
> ```python
> FILE_NAME = f"fss_{int(round(FSS * 1e6))}"
> ```

## Simulation setup

- **Energy parameters**  
  - Exciton center: `EXCITON = 1.300 eV`  
  - Binding energy: `BINDING = 3 meV`  
  - Fine-structure splitting: `FSS` (varied, e.g. 0 µeV or 5 µeV)  
  - Biexciton level: derived from the above

- **Cavity parameters**  
  - Quality factor: `Q = 5 × 10^4`  
  - Mode volume: `V_eff = 0.5 µm³`  
  - Resonant wavelength: `λ = 930 nm`  
  - Refractive index: `n = 3.4`

- **Dipole moment**  
  - `10 Debye` (converted to SI)

- **Detuning**  
  - Resonant case: Δ = 0 GHz  
  - Detuned π-pulse: Δ = 2 × 10^10 rad/s (~3.18 GHz)  

## Scenarios

Each FSS setting is simulated for three classical two-photon drive scenarios:

1. **π-pulse**  
   On resonance, pulse area ≈ π.

2. **5π-pulse**  
   On resonance, longer pulse (re-excitation effects visible).

3. **Detuned π-pulse**  
   Same pulse area as π, but with finite two-photon detuning.

The plot titles indicate the detuning (\(\Delta_{2\gamma}\)) in GHz.  
Top panels show the drive:  
- Solid curve → \(\Omega(t)\) (instantaneous Rabi frequency)  
- Dashed curve → \(\int^t \Omega(t')\,dt'\) (cumulative pulse area, in radians)

## CSV format

Each CSV contains **all** time series in one fil
