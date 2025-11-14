"""
PySINDy Example: Latent Layer Model with Oscillating Slow Dynamics

This example demonstrates:
1. An analytic input generating a hidden latent layer
2. Fitting to a one-dimensional time series
3. Latent state variables that are slow and oscillate, taking the place of time (t -> s(t))

The concept is that instead of modeling dynamics as dx/dt = f(x, t), we model
them as dx/dt = f(x, s(t)), where s(t) is a slow oscillating latent variable.
This is useful when the underlying dynamics are driven by slow periodic processes
such as seasonal cycles, tidal forces, or other environmental oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from scipy.integrate import odeint


def generate_latent_variable(t, omega1=0.2, omega2=0.15, amplitude=0.5):
    """
    Generate a slow oscillating latent variable s(t) that will replace time.
    This represents a slow manifold along which the dynamics evolve.
    The latent variable oscillates, representing cyclic or periodic slow dynamics.
    
    Parameters:
    -----------
    t : array
        Time points
    omega1 : float
        Primary frequency of oscillation (kept small for slow dynamics)
    omega2 : float
        Secondary frequency of oscillation
    amplitude : float
        Amplitude of oscillations
    
    Returns:
    --------
    s : array
        Slow oscillating latent variable
    """
    # Create a slow oscillating latent variable with multiple frequencies
    # This represents a slow cyclic process (e.g., seasonal, tidal, climate cycles)
    s = 1.0 + amplitude * np.sin(omega1 * t) + 0.3 * amplitude * np.cos(omega2 * t)
    return s


def generate_hidden_dynamics(s, params):
    """
    Hidden dynamics that depend on the latent variable s rather than time t.
    This is the true underlying system: dx/ds = f(x, s)
    
    Parameters:
    -----------
    s : array
        Latent variable
    params : dict
        Parameters for the dynamics
    
    Returns:
    --------
    x : array
        Generated time series
    """
    # Create a nonlinear function of the latent variable
    # This represents the hidden latent layer
    a, b, c = params['a'], params['b'], params['c']
    
    # Generate the observable through a nonlinear transformation
    # x = a*s^2 + b*sin(s) + c*s
    x = a * s**2 + b * np.sin(2 * np.pi * s) + c * s
    
    return x


def add_noise(signal, noise_level=0.05):
    """Add Gaussian noise to the signal"""
    noise = noise_level * np.std(signal) * np.random.randn(len(signal))
    return signal + noise


def compute_derivative_wrt_latent(x, s, window=5):
    """
    Compute dx/ds using smoothed numerical differentiation.
    
    Parameters:
    -----------
    x : array
        Observable time series
    s : array
        Latent variable
    window : int
        Window size for smoothing
    
    Returns:
    --------
    dx_ds : array
        Derivative of x with respect to s
    """
    from scipy.ndimage import uniform_filter1d
    
    # Smooth the signals to reduce noise in derivatives
    x_smooth = uniform_filter1d(x, size=window, mode='nearest')
    s_smooth = uniform_filter1d(s, size=window, mode='nearest')
    
    # Use centered differences where possible
    dx_ds = np.zeros_like(x)
    
    # Centered differences for all interior points
    for i in range(1, len(x) - 1):
        ds_diff = s_smooth[i+1] - s_smooth[i-1]
        if abs(ds_diff) > 1e-10:  # Avoid division by very small numbers
            dx_ds[i] = (x_smooth[i+1] - x_smooth[i-1]) / ds_diff
        else:
            dx_ds[i] = 0
    
    # Handle boundaries
    ds_0 = s_smooth[1] - s_smooth[0]
    if abs(ds_0) > 1e-10:
        dx_ds[0] = (x_smooth[1] - x_smooth[0]) / ds_0
    
    ds_n = s_smooth[-1] - s_smooth[-2]
    if abs(ds_n) > 1e-10:
        dx_ds[-1] = (x_smooth[-1] - x_smooth[-2]) / ds_n
    
    return dx_ds


def main():
    """Main function to run the PySINDy latent layer example"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate time points
    t = np.linspace(0, 50, 500)
    dt = t[1] - t[0]
    
    print("=" * 70)
    print("PySINDy Latent Layer Example")
    print("=" * 70)
    print()
    
    # Step 1: Generate the slow latent variable s(t)
    print("Step 1: Generating slow oscillating latent variable s(t)...")
    s = generate_latent_variable(t, omega1=0.2, omega2=0.15, amplitude=0.5)
    print(f"  - Latent variable range: [{s.min():.3f}, {s.max():.3f}]")
    print(f"  - This represents the slow manifold along which dynamics evolve")
    print(f"  - s(t) oscillates, representing cyclic/periodic slow dynamics")
    print()
    
    # Step 2: Generate the observable through hidden dynamics
    print("Step 2: Generating observable x through hidden latent dynamics...")
    params = {'a': 0.5, 'b': 1.0, 'c': 0.3}
    x_true = generate_hidden_dynamics(s, params)
    x_observed = add_noise(x_true, noise_level=0.05)
    print(f"  - Observable x range: [{x_observed.min():.3f}, {x_observed.max():.3f}]")
    print(f"  - Added 5% noise to simulate real measurements")
    print()
    
    # Step 3: Compute derivatives with respect to latent variable
    print("Step 3: Computing dx/ds (derivative w.r.t. latent variable)...")
    dx_ds = compute_derivative_wrt_latent(x_observed, s)
    print(f"  - Derivative range: [{dx_ds.min():.3f}, {dx_ds.max():.3f}]")
    print()
    
    # Step 4: Fit SINDy model
    print("Step 4: Fitting PySINDy model to discover dynamics...")
    print("  - Model form: dx/dt = f(x, s) where s(t) is the oscillating latent variable")
    print("  - Using polynomial feature library (degree 2)")
    
    # Prepare data for SINDy
    # Since s(t) oscillates and PySINDy needs monotonic time,
    # we fit dx/dt but include s as an additional feature
    X_train = np.column_stack([x_observed, s])  # Stack x and s as features
    
    # Compute dx/dt (derivative with respect to real time)
    dx_dt = np.gradient(x_observed, t)
    
    # Initialize SINDy with polynomial features
    feature_library = PolynomialLibrary(degree=2, include_bias=True)
    optimizer = STLSQ(threshold=0.1, alpha=0.001)
    model = SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    # Fit the model: we're discovering dx/dt = f(x, s)
    model.fit(X_train, t=t, x_dot=dx_dt.reshape(-1, 1))
    
    # Print the discovered equations
    print()
    print("Discovered equations:")
    print("-" * 70)
    print("dx/dt = f(x, s) where:")
    print("  x = observable state")
    print("  s = slow oscillating latent variable")
    model.print()
    print("-" * 70)
    print()
    
    # Step 5: Generate predictions
    print("Step 5: Generating model predictions...")
    x_dot_pred = model.predict(X_train)
    
    # Convert AxesArray to numpy array if needed
    if hasattr(x_dot_pred, 'flatten'):
        x_dot_pred_array = np.array(x_dot_pred).flatten()
    else:
        x_dot_pred_array = x_dot_pred.flatten()
    
    # Integrate the predictions to get x (using regular time)
    x_model = np.zeros_like(x_observed)
    x_model[0] = x_observed[0]  # Initial condition
    
    for i in range(len(t) - 1):
        dt_step = t[i+1] - t[i]
        x_model[i+1] = x_model[i] + x_dot_pred_array[i] * dt_step
    
    # Compute model score
    dx_dt_array = dx_dt.reshape(-1, 1)
    score = model.score(X_train, t=t, x_dot=dx_dt_array)
    print(f"  - Model R² score: {score:.4f}")
    print()
    
    # Step 6: Visualization
    print("Step 6: Creating visualizations...")
    create_visualization(t, s, x_observed, x_true, x_model, dx_dt, x_dot_pred_array)
    print("  - Saved plot as 'latent_layer_example.png'")
    print()
    
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Key Insights:")
    print("  1. The latent variable s(t) oscillates slowly, representing cyclic dynamics")
    print("  2. Dynamics are expressed as dx/dt = f(x, s(t))")
    print("  3. PySINDy discovers how x depends on both itself and the latent variable")
    print("  4. This approach captures how slow oscillations in s drive dynamics in x")
    print("  5. Useful for systems with periodic forcing or cyclic environmental variables")
    print()


def create_visualization(t, s, x_observed, x_true, x_model, dx_dt_true, dx_dt_pred):
    """Create comprehensive visualization of the results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PySINDy Latent Layer Model: Oscillating Latent Variable Dynamics', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Latent variable s(t)
    ax1 = axes[0, 0]
    ax1.plot(t, s, 'b-', linewidth=2, label='s(t) - Latent Variable')
    ax1.set_xlabel('Time t', fontsize=11)
    ax1.set_ylabel('Latent Variable s(t)', fontsize=11)
    ax1.set_title('Oscillating Slow Latent Variable', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axhline(y=np.mean(s), color='gray', linestyle='--', alpha=0.5, label='Mean')
    
    # Plot 2: Observable time series
    ax2 = axes[0, 1]
    ax2.plot(t, x_true, 'g-', linewidth=2, alpha=0.5, label='True x(t)')
    ax2.plot(t, x_observed, 'k.', markersize=2, alpha=0.5, label='Observed x(t)')
    ax2.plot(t, x_model, 'r-', linewidth=2, label='Model Prediction')
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.set_ylabel('Observable x', fontsize=11)
    ax2.set_title('Time Series: Observed vs Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Phase space (x vs s)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(s, x_observed, c=t, cmap='viridis', s=10, alpha=0.6, label='Observed')
    ax3.plot(s, x_model, 'r-', linewidth=2, alpha=0.8, label='Model')
    ax3.set_xlabel('Latent Variable s', fontsize=11)
    ax3.set_ylabel('Observable x', fontsize=11)
    ax3.set_title('Phase Space: x vs s (colored by time)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax3, label='Time')
    
    # Plot 4: Derivatives
    ax4 = axes[1, 1]
    ax4.plot(t, dx_dt_true, 'b-', linewidth=1, alpha=0.6, label='Numerical dx/dt')
    ax4.plot(t, dx_dt_pred, 'r--', linewidth=2, label='Model dx/dt')
    ax4.set_xlabel('Time t', fontsize=11)
    ax4.set_ylabel('dx/dt', fontsize=11)
    ax4.set_title('Derivative w.r.t. Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('latent_layer_example.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'latent_layer_example.png'")


if __name__ == "__main__":
    main()
