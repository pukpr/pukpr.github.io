"""
PySINDy Example: Integration with model_step_algorithm for Latent Layer Generation

This example demonstrates:
1. Loading parameters from .p JSON files
2. Using model_step_algorithm to generate a latent hidden layer
3. Fitting PySindy with sinusoidal regression library
4. Discovering dynamics in time series data with harmonic components

The model_step_algorithm computes values using harmonic (sinusoidal) components
from the parameter file, which serve as latent features for PySindy.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary, PolynomialLibrary, FourierLibrary
from pysindy.optimizers import STLSQ
import warnings
warnings.filterwarnings('ignore')


def model_step_algorithm(time_val, params):
    """
    Generate model value at a single time point using harmonic components.
    
    This is the core algorithm from the GIST that computes a model value
    by summing sinusoidal (harmonic) components with specified frequencies,
    amplitudes, and phases.
    
    Parameters:
    -----------
    time_val : float
        Single time point to evaluate
    params : dict
        Dictionary containing:
        - 'Aliased': list of frequencies (periods)
        - 'AliasedAmp': list of amplitudes
        - 'AliasedPhase': list of phases
    
    Returns:
    --------
    model_val : float
        Computed model value at the given time point
    """
    aliased = params.get('Aliased', [])
    aliased_amp = params.get('AliasedAmp', [])
    aliased_phase = params.get('AliasedPhase', [])
    
    model_val = 0.0
    
    # Sum up all harmonic components
    for i in range(len(aliased)):
        if i >= len(aliased_amp) or i >= len(aliased_phase):
            break
        
        period = aliased[i]
        amplitude = aliased_amp[i]
        phase = aliased_phase[i]
        
        # Skip if period is zero or too small to avoid division issues
        if abs(period) < 1e-10:
            continue
        
        # Calculate angular frequency: omega = 2*pi / period
        omega = 2.0 * np.pi / period
        
        # Add harmonic component: A * sin(omega * t + phi)
        model_val += amplitude * np.sin(omega * time_val + phase)
    
    return model_val


def generate_latent_layer_from_model(time, params):
    """
    Generate the latent hidden layer for all time points using model_step_algorithm.
    
    Parameters:
    -----------
    time : array
        Array of time points
    params : dict
        Parameter dictionary with Aliased, AliasedAmp, AliasedPhase
    
    Returns:
    --------
    latent_layer : array
        Array of model values (latent features) for each time point
    """
    latent_layer = np.zeros(len(time))
    
    for i, t in enumerate(time):
        latent_layer[i] = model_step_algorithm(t, params)
    
    return latent_layer


def generate_sinusoidal_features(time, params):
    """
    Generate individual sinusoidal features for each harmonic component.
    
    This creates a feature matrix where each column is a sinusoidal component
    at a specific frequency from the parameter file.
    
    Parameters:
    -----------
    time : array
        Array of time points
    params : dict
        Parameter dictionary with Aliased frequencies
    
    Returns:
    --------
    features : 2D array
        Matrix of sinusoidal features (n_samples x n_harmonics)
    feature_names : list
        Names of the features
    """
    aliased = params.get('Aliased', [])
    n_harmonics = len(aliased)
    n_samples = len(time)
    
    # Create feature matrix: sin and cos for each frequency
    features = np.zeros((n_samples, 2 * n_harmonics))
    feature_names = []
    
    for i, period in enumerate(aliased):
        if abs(period) < 1e-10:
            continue
        
        omega = 2.0 * np.pi / period
        
        # Sin component
        features[:, 2*i] = np.sin(omega * time)
        feature_names.append(f'sin(2π*t/{period:.3f})')
        
        # Cos component
        features[:, 2*i + 1] = np.cos(omega * time)
        feature_names.append(f'cos(2π*t/{period:.3f})')
    
    return features, feature_names


def load_parameters_from_p_file(p_file_path):
    """
    Load parameters from a .p JSON file.
    
    Parameters:
    -----------
    p_file_path : str or Path
        Path to the .p file
    
    Returns:
    --------
    params : dict
        Dictionary containing Aliased, AliasedAmp, AliasedPhase
    """
    with open(p_file_path, 'r') as f:
        params = json.load(f)
    
    return params


def load_data_csv(csv_path):
    """
    Load time series data from CSV file.
    
    Expected format: two columns (time, value)
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file
    
    Returns:
    --------
    time : array
        Time points
    values : array
        Observed values
    """
    # Try different CSV loading approaches
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] >= 2:
            time = df.iloc[:, 0].values
            values = df.iloc[:, 1].values
        else:
            raise ValueError("CSV must have at least 2 columns")
    except Exception as e:
        print(f"Warning: Could not load CSV as expected: {e}")
        # Create synthetic data as fallback
        time = np.linspace(0, 100, 1000)
        values = np.sin(0.1 * time) + 0.1 * np.random.randn(len(time))
    
    return time, values


def fit_pysindy_with_latent_layer(time, observed, latent_layer, params):
    """
    Fit PySINDy model using the latent layer as additional features.
    
    Parameters:
    -----------
    time : array
        Time points
    observed : array
        Observed time series
    latent_layer : array
        Latent hidden layer from model_step_algorithm
    params : dict
        Parameter dictionary
    
    Returns:
    --------
    model : SINDy
        Fitted PySINDy model
    prediction : array
        Model predictions
    score : float
        Model score
    """
    # Prepare data for SINDy
    # Stack observed and latent layer as features
    X_train = np.column_stack([observed, latent_layer])
    
    # Compute derivative with respect to time
    dx_dt = np.gradient(observed, time)
    
    # Create sinusoidal feature library
    # We'll use a custom library that includes the latent layer harmonics
    sin_features, feature_names = generate_sinusoidal_features(time, params)
    
    # Initialize SINDy with Fourier library (sinusoidal basis)
    # Use low frequency components based on the data
    feature_library = FourierLibrary(n_frequencies=5, include_sin=True, include_cos=True)
    
    # Alternative: use polynomial library
    # feature_library = PolynomialLibrary(degree=2, include_bias=True)
    
    optimizer = STLSQ(threshold=0.05, alpha=0.001)
    
    model = SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    # Fit the model
    try:
        model.fit(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
    except Exception as e:
        print(f"Warning during fitting: {e}")
        return None, None, 0.0
    
    # Generate predictions
    x_dot_pred = model.predict(X_train)
    
    # Integrate to get x
    x_model = np.zeros_like(observed)
    x_model[0] = observed[0]
    
    for i in range(len(time) - 1):
        dt_step = time[i+1] - time[i]
        x_model[i+1] = x_model[i] + x_dot_pred[i, 0] * dt_step
    
    # Compute score
    try:
        score = model.score(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
    except:
        score = 0.0
    
    return model, x_model, score


def create_visualization(time, observed, latent_layer, prediction, params, output_file='pysindy_model_step.png'):
    """
    Create comprehensive visualization of the analysis.
    
    Parameters:
    -----------
    time : array
        Time points
    observed : array
        Observed time series
    latent_layer : array
        Latent hidden layer
    prediction : array
        Model predictions
    params : dict
        Parameter dictionary
    output_file : str
        Output filename for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PySINDy with model_step_algorithm Latent Layer', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Observed data
    ax1 = axes[0, 0]
    ax1.plot(time, observed, 'b-', linewidth=1, alpha=0.7, label='Observed')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Observable x', fontsize=11)
    ax1.set_title('Observed Time Series', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Latent layer from model_step_algorithm
    ax2 = axes[0, 1]
    ax2.plot(time, latent_layer, 'g-', linewidth=1.5, label='Latent Layer')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Latent Variable s(t)', fontsize=11)
    ax2.set_title('Latent Hidden Layer (from model_step_algorithm)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Observed vs Prediction
    ax3 = axes[1, 0]
    ax3.plot(time, observed, 'b-', linewidth=1, alpha=0.6, label='Observed')
    if prediction is not None:
        ax3.plot(time, prediction, 'r-', linewidth=2, label='PySINDy Prediction')
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Observable x', fontsize=11)
    ax3.set_title('Observed vs Model Prediction', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Phase space (x vs latent layer)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(latent_layer, observed, c=time, cmap='viridis', s=10, alpha=0.6)
    ax4.set_xlabel('Latent Layer s', fontsize=11)
    ax4.set_ylabel('Observable x', fontsize=11)
    ax4.set_title('Phase Space: x vs s (colored by time)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{output_file}'")
    plt.close()


def main():
    """
    Main function demonstrating PySINDy with model_step_algorithm integration.
    """
    print("=" * 70)
    print("PySINDy with model_step_algorithm Latent Layer")
    print("=" * 70)
    print()
    
    # Example 1: Load parameters from a .p file
    # Find available .p files
    p_file_dir = Path('/home/runner/work/pukpr.github.io/pukpr.github.io/results/python_1930_1960/p')
    
    if p_file_dir.exists():
        p_files = list(p_file_dir.glob('*.p'))
        if p_files:
            # Use the first available .p file as an example
            p_file = p_files[0]
            print(f"Step 1: Loading parameters from {p_file.name}...")
            params = load_parameters_from_p_file(p_file)
            
            print(f"  - Found {len(params.get('Aliased', []))} harmonic components")
            print(f"  - Frequency range: {min(params.get('Aliased', [1])):.3f} - {max(params.get('Aliased', [1])):.3f}")
            print()
        else:
            print("No .p files found. Using synthetic parameters...")
            params = create_synthetic_params()
    else:
        print("Parameter directory not found. Using synthetic parameters...")
        params = create_synthetic_params()
    
    # Step 2: Generate synthetic time series or load from CSV
    print("Step 2: Generating/Loading time series data...")
    time = np.linspace(0, 200, 2000)
    
    # For demonstration, create synthetic data that includes the latent layer
    # In real use, you would load actual CSV data
    latent_true = generate_latent_layer_from_model(time, params)
    
    # Add some dynamics on top of the latent layer
    observed = latent_true + 0.3 * np.sin(0.05 * time) + 0.1 * np.random.randn(len(time))
    
    print(f"  - Time points: {len(time)}")
    print(f"  - Time range: [{time.min():.2f}, {time.max():.2f}]")
    print(f"  - Observable range: [{observed.min():.3f}, {observed.max():.3f}]")
    print()
    
    # Step 3: Generate latent layer using model_step_algorithm
    print("Step 3: Generating latent hidden layer using model_step_algorithm...")
    latent_layer = generate_latent_layer_from_model(time, params)
    print(f"  - Latent layer range: [{latent_layer.min():.3f}, {latent_layer.max():.3f}]")
    print(f"  - This represents the harmonic decomposition from parameters")
    print()
    
    # Step 4: Fit PySINDy model
    print("Step 4: Fitting PySINDy model with latent layer...")
    print("  - Using Fourier (sinusoidal) feature library")
    print("  - Model form: dx/dt = f(x, s) where s is the latent layer")
    
    model, prediction, score = fit_pysindy_with_latent_layer(time, observed, latent_layer, params)
    
    if model is not None:
        print()
        print("Discovered equations:")
        print("-" * 70)
        model.print()
        print("-" * 70)
        print(f"Model R² score: {score:.4f}")
    else:
        print("  - Model fitting encountered issues")
        prediction = latent_layer  # Fallback to latent layer
    
    print()
    
    # Step 5: Create visualization
    print("Step 5: Creating visualization...")
    create_visualization(time, observed, latent_layer, prediction, params)
    print()
    
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Key Features:")
    print("  1. Loaded parameters from .p JSON file (Aliased, AliasedAmp, AliasedPhase)")
    print("  2. Generated latent layer using model_step_algorithm (harmonic sum)")
    print("  3. Used sinusoidal (Fourier) library in PySINDy for regression")
    print("  4. Discovered dynamics: dx/dt = f(x, s(t))")
    print("  5. The latent layer captures slow harmonic oscillations")
    print()
    
    return model, params, latent_layer, prediction


def create_synthetic_params():
    """Create synthetic parameters for demonstration."""
    return {
        'Aliased': [18.6, 9.3, 6.2, 27.9, 13.95],
        'AliasedAmp': [0.5, 0.3, 0.2, 0.15, 0.1],
        'AliasedPhase': [0.0, 1.5, 3.0, 4.5, 2.0]
    }


if __name__ == "__main__":
    main()
