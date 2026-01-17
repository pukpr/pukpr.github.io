"""
Sinusoidal multiple linear regression with optional single linear X column

This is a simplified version that removes the previous `linear_scales` feature
and instead provides a boolean flag `add_linear_x`. When True, a single column
equal to X is appended to the design matrix (after sin/cos columns). The model
fits terms:

    Y ≈ intercept + sum_i [a_i * sin(k_i N_i X) + b_i * cos(k_i N_i X)] + c * X

Usage:
    - build_design_matrix(X, N_list, k=1.0, add_intercept=True, add_linear_x=False)
    - fit_sinusoidal_regression(..., add_linear_x=False)
    - predict_from_coefs(..., add_linear_x=False)
"""
from typing import Sequence, Union, Tuple, Dict, Any, Optional
import numpy as np


def build_design_matrix(
    X: np.ndarray,
    N_list: Sequence[int],
    k: Union[float, Sequence[float]] = 1.0,
    add_intercept: bool = True,
    add_linear_x: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the design matrix with columns:
      [1 (optional), sin(k_i * N_i * X), cos(k_i * N_i * X) for each N_i,
                       X (optional single linear column)]

    Returns:
        - A: design matrix (n_samples, n_columns)
        - k_arr: numpy array of k values aligned with N_list
    """
    X = np.asarray(X)
    if X.ndim != 1:
        raise ValueError("X must be a 1-D array of sample positions (e.g., time or angle).")

    N_arr = np.asarray(N_list, dtype=int)
    if N_arr.ndim != 1:
        raise ValueError("N_list must be a 1-D sequence of integers.")

    # Normalize k to an array of same length as N_list
    if np.isscalar(k):
        k_arr = np.full_like(N_arr, float(k), dtype=float)
    else:
        k_arr = np.asarray(k, dtype=float)
        if k_arr.shape != N_arr.shape:
            raise ValueError("k must be a scalar or have the same length as N_list.")

    n = X.shape[0]
    cols = []
    # sinusoidal terms
    for ki, Ni in zip(k_arr, N_arr):
        arg = ki * Ni * X
        cols.append(np.sin(arg))
        cols.append(np.cos(arg))

    # single linear X column if requested
    if add_linear_x:
        cols.append(X)

    A = np.column_stack(cols) if cols else np.empty((n, 0))
    if add_intercept:
        A = np.column_stack([np.ones(n), A])
    return A, k_arr


def fit_sinusoidal_regression(
    X: np.ndarray,
    Y: np.ndarray,
    N_list: Sequence[int],
    k: Union[float, Sequence[float]] = 1.0,
    intercept: bool = True,
    ridge: Optional[float] = None,
    rcond: Optional[float] = None,
    add_linear_x: bool = False,
) -> Dict[str, Any]:
    """
    Fit linear regression:
      Y ≈ intercept + sum_i [a_i * sin(k_i N_i X) + b_i * cos(k_i N_i X)] + c * X (optional)

    Parameters:
        add_linear_x: when True, include one column equal to X (coefficient returned as 'coef_x').

    Returns:
        result dict with keys including:
          - 'coefs', 'intercept', 'coefs_by_N', 'coef_x' (if add_linear_x True),
            'predict', 'R2', 'mse', 'A', 'k_arr', ...
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 1:
        raise ValueError("X must be 1-D array of positions.")
    if Y.shape[0] != X.shape[0]:
        raise ValueError("X and Y must have the same number of samples in axis 0.")

    A, k_arr = build_design_matrix(X, N_list, k=k, add_intercept=intercept, add_linear_x=add_linear_x)
    n_samples, n_cols = A.shape

    y_was_1d = (Y.ndim == 1)
    if y_was_1d:
        Y = Y.reshape(-1, 1)

    if ridge is None:
        coefs, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=rcond)
    else:
        ATA = A.T @ A
        n_cols = ATA.shape[0]
        reg = np.eye(n_cols) * float(ridge)
        if intercept and n_cols > 0:
            reg[0, 0] = 0.0
        ATA_reg = ATA + reg
        ATy = A.T @ Y
        coefs = np.linalg.solve(ATA_reg, ATy)
        residuals = None
        rank = None
        s = None

    # Extract intercept and coefficient body
    if intercept:
        intercept_val = coefs[0, :]
        coef_body = coefs[1:, :]
    else:
        intercept_val = np.zeros((1, Y.shape[1]))[0]
        coef_body = coefs

    # Map sin/cos coefficients to N -> (sin_coef, cos_coef)
    N_arr = np.asarray(list(N_list), dtype=int)
    coefs_by_N = {}
    m = N_arr.size
    for i, Ni in enumerate(N_arr):
        sin_idx = 2 * i
        cos_idx = 2 * i + 1
        sin_coef = coef_body[sin_idx, :] if coef_body.shape[0] > sin_idx else np.zeros((Y.shape[1],))
        cos_coef = coef_body[cos_idx, :] if coef_body.shape[0] > cos_idx else np.zeros((Y.shape[1],))
        if y_was_1d:
            coefs_by_N[int(Ni)] = (float(sin_coef[0]), float(cos_coef[0]))
        else:
            coefs_by_N[int(Ni)] = (sin_coef.copy(), cos_coef.copy())

    # Extract the single linear X coefficient if present
    coef_x = None
    if add_linear_x:
        idx = 2 * m  # position after all sin/cos columns in coef_body
        if coef_body.shape[0] > idx:
            linear_coef = coef_body[idx, :]
        else:
            linear_coef = np.zeros((Y.shape[1],))
        if y_was_1d:
            coef_x = float(linear_coef[0])
        else:
            coef_x = linear_coef.copy()

    def predict_fn(X_new: np.ndarray) -> np.ndarray:
        A_new, _ = build_design_matrix(X_new, N_list, k=k_arr, add_intercept=intercept, add_linear_x=add_linear_x)
        preds = A_new @ coefs
        if y_was_1d:
            return preds.ravel()
        return preds

    # Compute predictions/residuals using the same shapes to avoid broadcasting issues
    preds = A @ coefs            # shape (n_samples, n_targets)
    residuals_arr = Y - preds

    ss_res = np.sum(residuals_arr ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        R2 = 1.0 - ss_res / ss_tot
    mse = np.mean(residuals_arr ** 2, axis=0)

    if y_was_1d:
        coefs_out = coefs.ravel()
        intercept_out = float(intercept_val[0]) if intercept else 0.0
        R2 = float(R2[0])
        mse = float(mse[0])

    result = {
        "coefs": coefs_out if y_was_1d else coefs,
        "intercept": intercept_out if y_was_1d else intercept_val,
        "coefs_by_N": coefs_by_N,
        "coef_x": coef_x,
        "predict": predict_fn,
        "R2": R2,
        "mse": mse,
        "A": A,
        "k_arr": k_arr,
        "residuals": residuals,
        "lstsq_rank": rank,
        "lstsq_singular_values": s,
    }
    return result


def predict_from_coefs(
    X: np.ndarray,
    coefs: np.ndarray,
    N_list: Sequence[int],
    k: Union[float, Sequence[float]] = 1.0,
    intercept: bool = True,
    add_linear_x: bool = False,
) -> np.ndarray:
    A, _ = build_design_matrix(X, N_list, k=k, add_intercept=intercept, add_linear_x=add_linear_x)
    return A @ coefs


if __name__ == "__main__":
    # Demo: include a linear c * X term in addition to sin/cos terms
    import pprint
    rng = np.random.default_rng(1)

    n = 500
    X = np.linspace(0, 2 * np.pi, n)
    N_list = [1, 3, 7]
    k_true = 0.9
    true_coefs = {
        1: (1.5, -0.7),
        3: (0.3, 0.8),
        7: (-0.9, 0.1),
    }
    intercept_true = 0.25
    c_true = 2.3  # true linear coefficient for X

    y_true = np.full_like(X, fill_value=intercept_true, dtype=float)
    for N in N_list:
        a, b = true_coefs[N]
        y_true += a * np.sin(k_true * N * X) + b * np.cos(k_true * N * X)
    y_true += c_true * X  # add linear trend

    # NO NOISE so a perfect fit should be possible when we include the linear term
    noise_sigma = 0.0
    Y = y_true + rng.normal(scale=noise_sigma, size=X.shape)

    # Fit with add_linear_x=True to add the X column (so fitted coefficient corresponds to c)
    model = fit_sinusoidal_regression(X, Y, N_list=N_list, k=k_true, intercept=True, add_linear_x=True)

    print("mse:", model["mse"])
    print("R2:", model["R2"])
    print("intercept:", model["intercept"])
    print("coefs_by_N:", model["coefs_by_N"])
    print("coef_x:", model["coef_x"])
    print("norm residuals:", np.linalg.norm(Y - model["predict"](X)))
