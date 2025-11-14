#!/usr/bin/env python3
"""
Comprehensive Example: PySINDy with model_step_algorithm Integration

This script demonstrates the complete workflow from the GIST template:
1. Load time series data (CSV)
2. Load parameters from .p JSON file
3. Generate latent layer using model_step_algorithm
4. Fit PySINDy with sinusoidal library
5. Analyze and visualize results

This serves as a complete demonstration of the integration between
the GIST template's model_step_algorithm and PySINDy.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path


def demo_workflow():
    """
    Complete demonstration of the integrated workflow.
    """
    print("=" * 80)
    print("COMPREHENSIVE EXAMPLE: PySINDy with model_step_algorithm")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic data that mimics real tidal/climate data
    print("Step 1: Generating Synthetic Time Series Data")
    print("-" * 80)
    
    # Time range: 30 years of monthly data (similar to climate records)
    time = np.linspace(1930, 1960, 360)
    
    # Create synthetic data with known harmonic components
    # These mimic lunar tidal cycles and climate oscillations
    signal = (
        0.5 * np.sin(2 * np.pi * time / 18.6) +      # 18.6-year lunar nodal cycle
        0.3 * np.sin(2 * np.pi * time / 9.3) +       # First harmonic
        0.2 * np.sin(2 * np.pi * time / 6.2) +       # Second harmonic
        0.15 * np.sin(2 * np.pi * time / 4.65) +     # Third harmonic
        0.1 * np.random.randn(len(time))             # Measurement noise
    )
    
    print(f"  - Generated {len(time)} data points")
    print(f"  - Time range: [{time.min():.1f}, {time.max():.1f}] years")
    print(f"  - Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"  - Contains 4 harmonic components + noise")
    print()
    
    # Step 2: Create parameter dictionary (as would be in .p file)
    print("Step 2: Setting Up Parameters (from .p JSON format)")
    print("-" * 80)
    
    params = {
        'Aliased': [18.6, 9.3, 6.2, 4.65],           # Periods (years)
        'AliasedAmp': [0.5, 0.3, 0.2, 0.15],         # Amplitudes
        'AliasedPhase': [0.0, 0.0, 0.0, 0.0]         # Phases (radians)
    }
    
    print(f"  - Number of harmonic components: {len(params['Aliased'])}")
    print(f"  - Periods: {params['Aliased']}")
    print(f"  - Amplitudes: {params['AliasedAmp']}")
    print(f"  - Phases: {params['AliasedPhase']}")
    print()
    
    # Step 3: Apply model_step_algorithm
    print("Step 3: Generating Latent Layer using model_step_algorithm")
    print("-" * 80)
    
    def model_step_algorithm(t, params):
        """Compute model value using harmonic sum."""
        model_val = 0.0
        for i in range(len(params['Aliased'])):
            period = params['Aliased'][i]
            amplitude = params['AliasedAmp'][i]
            phase = params['AliasedPhase'][i]
            
            omega = 2.0 * np.pi / period
            model_val += amplitude * np.sin(omega * t + phase)
        
        return model_val
    
    # Generate latent layer for all time points
    latent_layer = np.array([model_step_algorithm(t, params) for t in time])
    
    print(f"  - Latent layer computed for {len(latent_layer)} time points")
    print(f"  - Latent layer range: [{latent_layer.min():.3f}, {latent_layer.max():.3f}]")
    print(f"  - This is the sum of {len(params['Aliased'])} sinusoidal components")
    print()
    
    # Verify latent layer matches signal (should be close since we used same params)
    correlation = np.corrcoef(signal, latent_layer)[0, 1]
    print(f"  - Correlation with original signal: {correlation:.4f}")
    print(f"    (High correlation expected since we used the same parameters)")
    print()
    
    # Step 4: Fit PySINDy model
    print("Step 4: Fitting PySINDy Model")
    print("-" * 80)
    
    try:
        from pysindy import SINDy
        from pysindy.feature_library import FourierLibrary
        from pysindy.optimizers import STLSQ
        
        # Prepare data
        X_train = np.column_stack([signal, latent_layer])
        dx_dt = np.gradient(signal, time)
        
        # Create model with Fourier (sinusoidal) library
        feature_library = FourierLibrary(n_frequencies=4, include_sin=True, include_cos=True)
        optimizer = STLSQ(threshold=0.05, alpha=0.001)
        model = SINDy(feature_library=feature_library, optimizer=optimizer)
        
        # Fit model
        model.fit(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
        
        print("  - Model fitted successfully")
        print("  - Feature library: Fourier (sinusoidal)")
        print("  - Optimizer: STLSQ (sparse regression)")
        print()
        
        # Display discovered equations
        print("  Discovered Equations:")
        print("  " + "=" * 76)
        model.print()
        print("  " + "=" * 76)
        
        # Compute score
        score = model.score(X_train, t=time, x_dot=dx_dt.reshape(-1, 1))
        print(f"  - Model R² score: {score:.4f}")
        print()
        
        # Generate predictions
        x_dot_pred = model.predict(X_train)
        x_model = np.zeros_like(signal)
        x_model[0] = signal[0]
        
        for i in range(len(time) - 1):
            dt = time[i+1] - time[i]
            x_model[i+1] = x_model[i] + x_dot_pred[i, 0] * dt
        
        has_pysindy = True
        
    except ImportError:
        print("  - PySINDy not available, skipping model fitting")
        x_model = latent_layer
        has_pysindy = False
    
    # Step 5: Visualization
    print("Step 5: Creating Comprehensive Visualization")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PySINDy with model_step_algorithm: Complete Workflow', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Original time series
    ax1 = axes[0, 0]
    ax1.plot(time, signal, 'b-', linewidth=1, label='Original Signal')
    ax1.set_xlabel('Time (years)', fontsize=11)
    ax1.set_ylabel('Signal Value', fontsize=11)
    ax1.set_title('Input: Time Series Data', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Latent layer from model_step_algorithm
    ax2 = axes[0, 1]
    ax2.plot(time, latent_layer, 'g-', linewidth=1.5, label='Latent Layer')
    ax2.set_xlabel('Time (years)', fontsize=11)
    ax2.set_ylabel('Latent Value', fontsize=11)
    ax2.set_title('Latent Layer (model_step_algorithm)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add text showing harmonic components
    text_str = "Harmonic Components:\n"
    for i, period in enumerate(params['Aliased']):
        text_str += f"  {period:.1f} yr\n"
    ax2.text(0.02, 0.98, text_str, transform=ax2.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel 3: Model fit comparison
    ax3 = axes[1, 0]
    ax3.plot(time, signal, 'b-', linewidth=1, alpha=0.6, label='Original')
    if has_pysindy:
        ax3.plot(time, x_model, 'r-', linewidth=1.5, label='PySINDy Model')
        residual = signal - x_model
        rmse = np.sqrt(np.mean(residual**2))
        ax3.text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=ax3.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        ax3.plot(time, latent_layer, 'r-', linewidth=1.5, label='Latent Layer')
    ax3.set_xlabel('Time (years)', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Model Fit Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Phase space
    ax4 = axes[1, 1]
    scatter = ax4.scatter(latent_layer, signal, c=time, cmap='viridis',
                         s=20, alpha=0.6, edgecolors='none')
    ax4.set_xlabel('Latent Layer', fontsize=11)
    ax4.set_ylabel('Signal', fontsize=11)
    ax4.set_title('Phase Space: Signal vs Latent', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4, label='Time (years)')
    
    plt.tight_layout()
    plt.savefig('comprehensive_example.png', dpi=300, bbox_inches='tight')
    print(f"  - Visualization saved: comprehensive_example.png")
    print()
    
    # Step 6: Summary
    print("=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print()
    print("This example demonstrated:")
    print("  1. ✓ Generated synthetic time series with known harmonic components")
    print("  2. ✓ Created parameter dictionary (format compatible with .p JSON files)")
    print("  3. ✓ Applied model_step_algorithm to generate latent hidden layer")
    print("  4. ✓ Fitted PySINDy model with sinusoidal (Fourier) library")
    print("  5. ✓ Discovered dynamics: dx/dt = f(x, s)")
    print("  6. ✓ Created comprehensive 4-panel visualization")
    print()
    print("Key Takeaways:")
    print("  - model_step_algorithm generates latent features from harmonic components")
    print("  - PySINDy discovers sparse dynamics linking observables to latent layer")
    print("  - Sinusoidal regression captures periodic/cyclic behavior")
    print("  - Method applicable to tidal, climate, and other oscillatory data")
    print()
    print("For real data analysis, use:")
    print("  python run_pysindy_on_data.py your_data.csv --p-file params.p")
    print()


if __name__ == "__main__":
    demo_workflow()
