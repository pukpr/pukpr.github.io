# PySINDy Latent Layer Example

## Overview

This example demonstrates a PySINDy (Sparse Identification of Nonlinear Dynamics) model that uses:
1. **An analytic input** generating a hidden latent layer
2. **Fitting to a one-dimensional time series**
3. **Latent state variables** that are slow and take the place of time: `t → s(t)`

## Concept

Traditional dynamical systems are modeled as:
```
dx/dt = f(x, t)
```

In this example, we model dynamics driven by a **slow oscillating latent variable**:
```
dx/dt = f(x, s(t))
```

where `s(t)` is a slow oscillating latent variable that evolves in time. This is useful when:
- The underlying dynamics are driven by slow periodic processes
- Environmental or forcing variables oscillate (e.g., seasonal, tidal, lunar cycles)
- The system responds to cyclic latent states rather than explicit time
- Hidden periodic processes influence the observable dynamics

## Mathematical Framework

### Latent Variable Generation
The slow oscillating latent variable `s(t)` is generated as:
```
s(t) = 1 + 0.5 × sin(ω₁t) + 0.15 × cos(ω₂t)
```

where ω₁ = 0.2 and ω₂ = 0.15 are slow frequencies. This creates an oscillating "environmental" or "forcing" variable that represents cyclic processes like:
- Seasonal variations
- Tidal cycles
- Lunar nodal cycles (18.6 years)
- Multi-decadal climate oscillations

### Hidden Dynamics
The observable `x` is generated through a nonlinear transformation of the latent variable:
```
x = a × s² + b × sin(2πs) + c × s
```

This represents the hidden latent layer connecting `s` to the observed signal `x`.

### SINDy Model
PySINDy discovers the governing equation `dx/dt = f(x, s(t))` from data by:
1. Computing numerical derivatives `dx/dt`
2. Building a library of candidate functions (polynomials in x and s)
3. Using sparse regression (STLSQ) to identify active terms

The key insight is that even though `s` oscillates, the model captures how `x` responds to the periodic forcing from `s`.

## Running the Example

### Prerequisites
```bash
pip install pysindy numpy matplotlib scipy
```

### Execution
```bash
python latent_layer_example.py
```

### Output
The script produces:
- Console output showing the analysis steps
- A visualization saved as `latent_layer_example.png` with four subplots:
  1. **Oscillating Latent Variable s(t)**: Shows the slow periodic forcing variable
  2. **Time Series**: Observed data vs model predictions
  3. **Phase Space**: x vs s (colored by time) - shows limit cycle-like behavior
  4. **Derivatives**: Comparison of numerical and model dx/dt

## Results Interpretation

The discovered equations show how `x` changes in response to the oscillating latent variable `s`:
- Terms involving `s` show explicit dependence on the periodic forcing
- Terms involving `x` show self-interaction (feedback)
- Terms with `x × s` show coupling between state and forcing
- The model score (R²) indicates quality of fit

### Key Insights

1. **Oscillating Forcing**: The latent variable `s(t)` provides periodic forcing that drives the observable `x`
2. **Time-Dependent Parametrization**: By using `s(t)` as an explicit variable, we capture how slow oscillations influence fast dynamics
3. **Phase Space Structure**: The x-s phase plot reveals limit cycle-like behavior driven by the periodic latent variable
4. **Applications**: Useful for systems with:
   - Seasonal or periodic environmental forcing
   - Tidal influences
   - Climate cycles (ENSO, PDO, lunar nodal cycle)
   - Biological rhythms

## Applications

This approach is particularly useful for:
- **Climate systems**: Where oscillations occur on slow manifolds (e.g., ENSO, lunar cycles)
- **Biological systems**: With slow metabolic processes
- **Economic systems**: With long-term trend dynamics
- **Geophysical systems**: With tidal or orbital forcing

## Extensions

Possible extensions of this example:
1. Multi-dimensional latent spaces: `s(t) = [s₁(t), s₂(t), ...]`
2. Multiple observables: `x(t) = [x₁(t), x₂(t), ...]`
3. Control inputs: `dx/ds = f(x, s, u)`
4. Time-delay embeddings combined with latent variables

## References

- [PySINDy Documentation](https://pysindy.readthedocs.io/)
- Brunton, S. L., et al. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." PNAS (2016)
- Champion, K., et al. "Data-driven discovery of coordinates and governing equations." PNAS (2019)

## Code Structure

```
latent_layer_example.py
├── generate_latent_variable()     # Creates slow s(t)
├── generate_hidden_dynamics()      # Maps s → x through hidden layer
├── compute_derivative_wrt_latent() # Computes dx/ds numerically
├── main()                         # Orchestrates the analysis
└── create_visualization()          # Generates comprehensive plots
```

## License

This example is part of the pukpr.github.io repository and follows the same license.

---

# PySINDy with model_step_algorithm Integration

## New Examples (Integration with GIST Template)

Two additional examples have been added that integrate the `model_step_algorithm` from the GIST template with PySINDy:

### 1. `pysindy_with_model_step.py`

**Purpose**: Demonstrates integration of the `model_step_algorithm` for generating latent hidden layers.

**Key Features**:
- Loads parameters from `.p` JSON files (Aliased, AliasedAmp, AliasedPhase)
- Uses `model_step_algorithm` to generate latent layer via harmonic sum
- Applies PySINDy with Fourier (sinusoidal) feature library
- Discovers dynamics: dx/dt = f(x, s) where s is the latent layer

**Algorithm: model_step_algorithm**

The core algorithm computes model values by summing harmonic components:

```
model_val = Σ(A_i * sin(ω_i * t + φ_i))

where:
  ω_i = 2π / Period_i  (angular frequency)
  A_i = Amplitude_i
  φ_i = Phase_i
```

**Usage**:
```bash
python pysindy_with_model_step.py
```

### 2. `run_pysindy_on_data.py`

**Purpose**: Command-line tool for running PySINDy analysis on real or synthetic data.

**Usage**:

Demo mode (synthetic data):
```bash
python run_pysindy_on_data.py --demo
```

With CSV data:
```bash
python run_pysindy_on_data.py path/to/data.csv
```

With custom parameter file:
```bash
python run_pysindy_on_data.py data.csv --p-file path/to/params.p
```

**Arguments**:
- `csv`: Input CSV file (two-column format: time, value)
- `--p-file`: Parameter `.p` JSON file
- `--output`: Output plot filename (default: `pysindy_result.png`)
- `--demo`: Force demo mode with synthetic data

**CSV Format**:
```
time1,value1
time2,value2
...
```

**Parameter File Format (.p JSON)**:
```json
{
  "Aliased": [18.6, 9.3, 6.2, ...],
  "AliasedAmp": [0.5, 0.3, 0.2, ...],
  "AliasedPhase": [0.0, 1.5, 3.0, ...]
}
```

**Output**:
- Console: Discovered equations, model score, summary
- Image: 4-panel visualization showing:
  - Panel 1: Observed time series
  - Panel 2: Latent hidden layer (harmonic components)
  - Panel 3: Model fit comparison
  - Panel 4: Phase space (x vs s)

## Method

The new examples implement a novel approach combining:

1. **Parameter Loading**: Read harmonic component parameters from `.p` JSON files
2. **Latent Layer Generation**: Use `model_step_algorithm` to compute hidden features
3. **Sinusoidal Regression**: Apply PySINDy with Fourier library
4. **Dynamics Discovery**: Find sparse representation of time evolution

## Data Files

Parameter files (`.p` JSON) are located in:
```
../../results/python_1930_1960/p/
```

These contain fitted harmonic parameters from tidal gauge data and climate records.

## GIST Reference

- **GIST Template**: model_step_algorithm implementation
  - URL: https://gist.github.com/pukpr/0b7ac85fad1ea36f65a9b50d6c30958b
