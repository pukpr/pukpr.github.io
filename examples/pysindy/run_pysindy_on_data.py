#!/usr/bin/env python3
"""
PySINDy Runner for Time Series Data with model_step_algorithm

This script extends the GIST template to work with real CSV data files.
It uses the model_step_algorithm to generate latent features and then
applies PySINDy with sinusoidal regression to discover dynamics.

Usage:
    python run_pysindy_on_data.py <csv_file> [--p-file <params.p>]
    
Example:
    python run_pysindy_on_data.py results/150sites.csv --p-file results/python_1930_1960/p/23d.dat.p
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
from pathlib import Path
from pysindy import SINDy
from pysindy.feature_library import FourierLibrary, PolynomialLibrary
from pysindy.optimizers import STLSQ
import warnings
warnings.filterwarnings('ignore')


def model_step_algorithm(time_val, params):
    """
    Core algorithm from GIST: compute model value using harmonic components.
    
    Parameters:
    -----------
    time_val : float
        Time point to evaluate
    params : dict
        Dictionary with 'Aliased', 'AliasedAmp', 'AliasedPhase'
    
    Returns:
    --------
    model_val : float
        Computed value from summing all harmonic components
    """
    aliased = params.get('Aliased', [])
    aliased_amp = params.get('AliasedAmp', [])
    aliased_phase = params.get('AliasedPhase', [])
    
    model_val = 0.0
    
    for i in range(min(len(aliased), len(aliased_amp), len(aliased_phase))):
        period = aliased[i]
        amplitude = aliased_amp[i]
        phase = aliased_phase[i]
        
        if abs(period) < 1e-10:
            continue
        
        omega = 2.0 * np.pi / period
        model_val += amplitude * np.sin(omega * time_val + phase)
    
    return model_val


def run_loop_time_series(time, observed, params):
    """
    Run the model_step_algorithm for all time points (explicit loop).
    
    This generates the latent hidden layer that will be used with PySINDy.
    
    Parameters:
    -----------
    time : array
        Time points
    observed : array
        Observed values (not used in latent generation, but kept for compatibility)
    params : dict
        Parameter dictionary
    
    Returns:
    --------
    model_vals : array
        Model values (latent layer) for all time points
    """
    model_vals = np.zeros(len(time))
    
    for i, t in enumerate(time):
        model_vals[i] = model_step_algorithm(t, params)
    
    return model_vals


def read_json_p(json_path):
    """Load parameters from .p JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def read_two_column_csv(csv_path):
    """
    Read CSV with two columns: time and value.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    
    Returns:
    --------
    time : array
        Time points
    values : array
        Observed values
    """
    df = pd.read_csv(csv_path, header=None)
    
    if df.shape[1] < 2:
        raise ValueError(f"CSV file must have at least 2 columns, got {df.shape[1]}")
    
    time = df.iloc[:, 0].values
    values = df.iloc[:, 1].values
    
    # Remove NaN values
    mask = ~(np.isnan(time) | np.isnan(values))
    time = time[mask]
    values = values[mask]
    
    return time, values


def normalize_rms_y(y):
    """Normalize array by RMS value."""
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        return y / rms
    return y


def fit_pysindy_with_sinusoidal_library(time, observed, latent_layer):
    """
    Fit PySINDy model using sinusoidal (Fourier) library.
    
    The latent layer serves as additional input features alongside the observed data.
    
    Parameters:
    -----------
    time : array
        Time points
    observed : array
        Observed time series
    latent_layer : array
        Latent features from model_step_algorithm
    
    Returns:
    --------
    model : SINDy
        Fitted model
    prediction : array
        Integrated predictions
    score : float
        Model R² score
    """
    # Stack observed and latent as features
    X_train = np.column_stack([observed, latent_layer])
    
    # Compute time derivative
    dx_dt = np.gradient(observed, time)
    
    # Use Fourier (sinusoidal) library as specified
    feature_library = FourierLibrary(
        n_frequencies=6,
        include_sin=True,
        include_cos=True
    )
    
    optimizer = STLSQ(threshold=0.05, alpha=0.001, max_iter=100)
    
    model = SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    try:
        model.fit(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
        
        # Generate predictions
        x_dot_pred = model.predict(X_train)
        
        # Integrate predictions
        x_model = np.zeros_like(observed)
        x_model[0] = observed[0]
        
        for i in range(len(time) - 1):
            dt = time[i+1] - time[i]
            x_model[i+1] = x_model[i] + x_dot_pred[i, 0] * dt
        
        # Compute score
        score = model.score(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
        
        return model, x_model, score
    
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None, None, 0.0


def create_comprehensive_plot(time, observed, latent_layer, prediction, 
                              output_file, title="PySINDy Analysis"):
    """
    Create a comprehensive 4-panel visualization.
    
    Parameters:
    -----------
    time : array
        Time points
    observed : array
        Observed data
    latent_layer : array
        Latent features from model_step_algorithm
    prediction : array
        PySINDy model predictions
    output_file : str
        Output filename
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Panel 1: Observed time series
    ax1 = axes[0, 0]
    ax1.plot(time, observed, 'b-', linewidth=0.8, alpha=0.7, label='Observed')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Value (normalized)', fontsize=11)
    ax1.set_title('Observed Data', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Latent layer from model_step_algorithm
    ax2 = axes[0, 1]
    ax2.plot(time, latent_layer, 'g-', linewidth=1.2, label='Latent Layer (model_step)')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Latent Value', fontsize=11)
    ax2.set_title('Latent Hidden Layer (Harmonic Components)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Observed vs PySINDy prediction
    ax3 = axes[1, 0]
    ax3.plot(time, observed, 'b-', linewidth=0.8, alpha=0.6, label='Observed')
    if prediction is not None:
        ax3.plot(time, prediction, 'r-', linewidth=1.5, alpha=0.8, label='PySINDy Model')
        residual = observed - prediction
        ax3.fill_between(time, observed, prediction, alpha=0.2, color='gray', label='Residual')
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Model Fit: Observed vs PySINDy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Phase space (observed vs latent)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(latent_layer, observed, c=time, cmap='viridis', 
                         s=15, alpha=0.6, edgecolors='none')
    ax4.set_xlabel('Latent Layer s(t)', fontsize=11)
    ax4.set_ylabel('Observable x(t)', fontsize=11)
    ax4.set_title('Phase Space: x vs s (colored by time)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run PySINDy with model_step_algorithm on time series data'
    )
    parser.add_argument('csv', nargs='?', default=None,
                       help='Input CSV file (two columns: time, value)')
    parser.add_argument('--p-file', '--p', dest='p_file', default=None,
                       help='Parameter .p JSON file')
    parser.add_argument('--output', '-o', default='pysindy_result.png',
                       help='Output plot filename')
    parser.add_argument('--demo', action='store_true',
                       help='Run with synthetic demo data')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PySINDy with model_step_algorithm - Time Series Analysis")
    print("=" * 80)
    print()
    
    # Load parameters
    if args.p_file:
        print(f"Loading parameters from: {args.p_file}")
        params = read_json_p(args.p_file)
    else:
        # Use first available .p file or create synthetic
        p_dir = Path('results/python_1930_1960/p')
        if p_dir.exists():
            p_files = list(p_dir.glob('*.p'))
            if p_files:
                args.p_file = str(p_files[0])
                print(f"Using parameter file: {args.p_file}")
                params = read_json_p(args.p_file)
            else:
                print("No .p files found, using synthetic parameters")
                params = {
                    'Aliased': [18.6, 9.3, 6.2],
                    'AliasedAmp': [0.5, 0.3, 0.2],
                    'AliasedPhase': [0.0, 1.5, 3.0]
                }
        else:
            print("Using synthetic parameters")
            params = {
                'Aliased': [18.6, 9.3, 6.2],
                'AliasedAmp': [0.5, 0.3, 0.2],
                'AliasedPhase': [0.0, 1.5, 3.0]
            }
    
    n_harmonics = len(params.get('Aliased', []))
    print(f"  - Number of harmonic components: {n_harmonics}")
    print()
    
    # Load or generate data
    if args.csv:
        print(f"Loading data from: {args.csv}")
        try:
            time, values = read_two_column_csv(args.csv)
            values = normalize_rms_y(values)
            print(f"  - Loaded {len(time)} data points")
        except Exception as e:
            print(f"  - Error loading CSV: {e}")
            print("  - Using synthetic data instead")
            args.demo = True
    
    if args.demo or not args.csv:
        print("Generating synthetic demo data...")
        time = np.linspace(0, 200, 2000)
        # Create observed data with latent layer plus additional dynamics
        latent_true = run_loop_time_series(time, None, params)
        values = latent_true + 0.2 * np.sin(0.05 * time) + 0.05 * np.random.randn(len(time))
        values = normalize_rms_y(values)
        print(f"  - Generated {len(time)} data points")
    
    print(f"  - Time range: [{time.min():.2f}, {time.max():.2f}]")
    print(f"  - Value range: [{values.min():.3f}, {values.max():.3f}]")
    print()
    
    # Step 1: Generate latent hidden layer using model_step_algorithm
    print("Step 1: Generating latent hidden layer...")
    print("  - Using model_step_algorithm with harmonic components from .p file")
    latent_layer = run_loop_time_series(time, values, params)
    print(f"  - Latent layer range: [{latent_layer.min():.3f}, {latent_layer.max():.3f}]")
    print(f"  - This is the sum of {n_harmonics} sinusoidal components")
    print()
    
    # Step 2: Fit PySINDy with sinusoidal library
    print("Step 2: Fitting PySINDy model...")
    print("  - Using Fourier (sinusoidal) feature library")
    print("  - Discovering dynamics: dx/dt = f(x, s)")
    print("    where x = observable, s = latent layer")
    
    model, prediction, score = fit_pysindy_with_sinusoidal_library(
        time, values, latent_layer
    )
    
    if model is not None:
        print()
        print("  Discovered PySINDy equations:")
        print("  " + "-" * 76)
        model.print()
        print("  " + "-" * 76)
        print(f"  Model R² score: {score:.4f}")
    else:
        print("  - Model fitting failed")
        prediction = latent_layer
    
    print()
    
    # Step 3: Create visualization
    print("Step 3: Creating visualization...")
    create_comprehensive_plot(
        time, values, latent_layer, prediction,
        output_file=args.output,
        title=f"PySINDy Analysis: {Path(args.csv).name if args.csv else 'Demo Data'}"
    )
    print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Input: {args.csv if args.csv else 'Synthetic demo data'}")
    print(f"  - Parameters: {args.p_file if args.p_file else 'Synthetic'}")
    print(f"  - Harmonics: {n_harmonics} components")
    print(f"  - Model score: {score:.4f}")
    print(f"  - Output: {args.output}")
    print()
    print("Method:")
    print("  1. Loaded parameters (Aliased, AliasedAmp, AliasedPhase) from .p file")
    print("  2. Applied model_step_algorithm to generate latent hidden layer")
    print("  3. Used PySINDy with sinusoidal regression library")
    print("  4. Discovered dynamics in terms of observable and latent features")
    print()


if __name__ == "__main__":
    main()
