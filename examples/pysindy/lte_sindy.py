#!/usr/bin/env python3
"""
timeseries_modeler.py

Read a two-column CSV (time, value). Read a JSON file with the same filename
appended by ".p" (e.g. data.csv.p) into a data dictionary. Create a clone of the
time-series and run a loop that steps through timestamps, calling a user-
customizable model-step function to produce a modeled series. At the end the
script computes Pearson's correlation coefficient and the variance of the
squared errors (and also MSE), using library routines.

Usage:
  python timeseries_modeler.py data.csv [--out fitted.csv] [--plot]
  python3 timeseries_modeler.py 11a.csv --plot 

Notes:
- The CSV must have at least two columns (time, value). Header is tolerated.
- Model is:
    - impulse every Imp_Stride rows
    - C_t = Imp_Amp * impulse_mask * sum_k PeriodsAmp[k] * sin(2*pi*time/Period_k + PeriodsPhase_k)
    - D_t = (1-Hold)*D_{t-1} + Hold*C_t
    - E_t = LTE_Amp * sin(D_t*LTE_Freq + LTE_Phase)
  Provide model parameters via --params as a JSON string (see defaults below).
- The implementation loops over timestamps (explicit for-loop) so the model can
  be stateful (sample-and-hold needs previous D).
- Pearson correlation computed using scipy.stats.pearsonr if available,
  otherwise a fallback via numpy.corrcoef is used.
- Outputs:
    - CSV with columns: time, observed, model, residual
    - JSON .p file updated with metadata (metrics, parameters used)

Dependencies:
  numpy, pandas, scipy (optional but recommended for Pearson), matplotlib (optional for plotting)
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple, Callable, Optional
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
# from sinusoidal_regression import fit_sinusoidal_regression  # use the provided module

from pysindy import SINDy
from pysindy.feature_library import FourierLibrary, PolynomialLibrary
from pysindy.optimizers import STLSQ
import warnings
warnings.filterwarnings('ignore')


TWOPI = 2.0 * math.pi
use_pearson = False
use_random = False
time_series_name = "none"

def read_two_column_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV-like file with two columns: time, value. Tolerant to header rows."""
    df = pd.read_csv(path, sep=r'[,\s]+', engine='python', header=0)
    if df.shape[1] < 2:
        raise ValueError("Input CSV must have at least two columns (time, value).")
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    mask = (~t.isna()) & (~y.isna())
    t = t[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)
    return t, y


def read_json_p(path: str) -> Dict[str, Any]:
    """Read JSON file at path (if exists), else return empty dict."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def write_json_p(path: str, data: Dict[str, Any]) -> None:
    """Write JSON dict to file path."""
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def clone_series(series: np.ndarray) -> np.ndarray:
    """Return a deep copy (clone) of the series."""
    return np.array(series, copy=True)

def pearsonr_truncate(a, b, align='start'):
    """
    Compute Pearson r between a and b when lengths differ by truncating the longer array.

    Parameters
    - a, b: array-like (will be converted to 1-D numpy arrays)
    - align: 'start' (truncate ends) or 'end' (truncate front) — how to align when truncating
    - drop_nan: if True, remove any sample pairs where either value is NaN before truncating

    Returns
    - (r, p): Pearson correlation coefficient and two-tailed p-value
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    # truncate longer to the shorter length according to alignment
    n = min(a.size, b.size)
    if n < 2:
        raise ValueError("Not enough samples after truncation to compute Pearson correlation (need >=2).")

    if align == 'start':
        a_tr = a[:n]
        b_tr = b[:n]
    elif align == 'end':
        a_tr = a[-n:]
        b_tr = b[-n:]
    else:
        raise ValueError("align must be 'start' or 'end'")

    # check for constant arrays (pearsonr requires non-constant inputs)
    if np.std(a_tr) == 0 or np.std(b_tr) == 0:
        # pearsonr will warn/return nan; here we explicitly handle it
        return np.nan, np.nan

    return pearsonr(a_tr, b_tr)

def normalize_rms_y(ts: np.ndarray) -> np.ndarray:
    """
    Return a copy of ts where the y-values (column 1) are scaled so their RMS == 1.0.
    No error checking (assumes ts is a 2D numpy array with at least 2 columns).
    """
    y = ts.copy()
    rms = np.sqrt(np.mean(y * y))
    return y / rms

# ---- Model-step interface ---------------------------------------------------
# A model-step function implements:
#   model_value, state = model_step(i, t_i, clone_i, state, params)
# Where:
#   i: integer time index (0..N-1)
#   t_i: timestamp value (float)
#   clone_i: cloned original series value at index i (float)
#   state: dict carrying model internal state (mutable or replaced)
#   params: dict of user parameters
#
# The runner will set up initial state = {} and call model_step for each i.
# ---------------------------------------------------------------------------


def model_step_algorithm(i: int, t_i: float, clone_i: float, state: Dict[str, Any], 
    Initial: float,
    Year: float,
    Imp_Stride: int,
    Imp_Amp: float,
    Periods: list,
    PeriodsAmp: list,
    PeriodsPhase: list,
    Aliased: list,
    AliasedAmp: list,
    AliasedPhase: list,
    Hold: float,
    Delta_Y: float,
    Imp_Amp2: float,
    Damp: Float):
    """
    Implements the algorithm-style model described:
      - Impulse mask every Imp_Stride rows (row numbering starts at 1)
      - C_t = (Imp_Amp if (row_idx % Imp_Stride == 1) else 0) * SUM_k PeriodsAmp[k]*sin(2*pi * t / Period_k + PeriodsPhase[k])
      - D_t = (1-Hold)*D_{t-1} + Hold*C_t    (initialize D_0 = Hold*C_0)
      - E_t = LTE_Amp * sin(D_t * LTE_Freq + LTE_Phase)

    Required params (defaults provided):
      - Imp_Stride: int (default 1)
      - Imp_Amp: float (default 1.0)
      - Periods: list of floats (default [1.0])
      - PeriodsAmp: list of floats (len match Periods, default [1.0])
      - PeriodsPhase: list of floats (len match Periods, default [0.0])
      - Hold: float in [0,1] (default 0.5)
      - LTE_Amp: float (default 1.0)
      - LTE_Freq: float (default 1.0)
      - LTE_Phase: float (default 0.0)
    """

    # compute sum of sinusoids
    ssum = 0.0
    for amp, per, ph in zip(PeriodsAmp, Periods, PeriodsPhase):
        per = float(per)
        if per == 0.0:
            continue
        ssum += float(amp) * math.sin(TWOPI * float(t_i) * (Year+Delta_Y) / per + float(ph))

#    if os.getenv('FORCING', '') == '1':
#        state['D_prev'] = 0.0
#        return ssum, state, 0.0

    # row index as 1..N (i is 0..N-1)
    row_idx = i + 2
    impulse = 1.0 if (row_idx % 12 == Imp_Stride) else 0.0
    if os.getenv('SEMI', '') == '1':
        impulse2 = 1.0 if (row_idx % 12 == (Imp_Stride+6) % 12) else 0.0
    else:
        impulse2 = 1.0 if (row_idx % 24 == Imp_Stride) else 0.0
    C_t = (impulse * Imp_Amp + impulse2 * Imp_Amp2) * ssum * math.exp(-Damp * abs(ssum))

    # stateful D
    if 'D_prev' not in state:
        # initialize as Hold * C_0 (algorithm initialization)
        state['D_prev'] = Initial

    D_prev = float(state['D_prev'])
    D_t = (1.0 - Hold) * D_prev + Hold * C_t
    state['D_prev'] = D_t

    # compute sum of aliases
    ssum = 0.0
    for amp, freq, ph in zip(AliasedAmp, Aliased, AliasedPhase):
        freq = float(freq)
        ssum += float(amp) * math.sin(TWOPI * float(t_i) * freq + float(ph))

    E_t = D_t

    return float(E_t), state, ssum


def fit_pysindy_with_sinusoidal_library(observed: np.ndarray, latent_layer: np.ndarray) -> np.ndarray:
    """
    Fit a PySINDy model using a Fourier (sinusoidal) library on the latent variable
    and return predictions with the same shape as `observed`.

    Parameters
    ----------
    observed : np.ndarray
        Observed time series (1-D or 2-D). The returned prediction will match this shape.
    latent_layer : np.ndarray
        Latent feature array (1-D or (n,1)). This is the independent variable s(t).

    Returns
    -------
    prediction : np.ndarray
        Predictions from the fitted model; same shape as `observed`.
    """
    # Coerce to numpy and remember original observed shape
    y_orig = np.asarray(observed)
    s_orig = np.asarray(latent_layer)
    orig_shape = y_orig.shape

    # Make column arrays for fitting: shapes (n, 1) and (n, m)
    if y_orig.ndim == 1:
        y = y_orig.reshape(-1, 1)
    elif y_orig.ndim == 2:
        y = y_orig.copy()
    else:
        raise ValueError("observed must be 1-D or 2-D array")

    if s_orig.ndim == 1:
        s = s_orig.reshape(-1, 1)
    elif s_orig.ndim == 2 and s_orig.shape[1] == 1:
        s = s_orig.copy()
    else:
        # If latent has multiple columns, use it as-is; typically expected to be 1-column
        s = s_orig.copy()

    # Basic sanity check
    n_samples = y.shape[0]
    if s.shape[0] != n_samples:
        raise ValueError(f"observed and latent_layer must have same length; "
                         f"observed has {n_samples}, latent has {s.shape[0]}")

    # Build model (Fourier library on latent only)
    feature_library = FourierLibrary(n_frequencies=16, include_sin=True, include_cos=True)
    optimizer = STLSQ(threshold=0.05, alpha=0.001, max_iter=1000)
    model = SINDy(feature_library=feature_library, optimizer=optimizer)

    # PySINDy versions sometimes require a t argument; provide a simple dummy integer sequence
    t_dummy = np.arange(n_samples)

    # Fit: using SINDy as a regression: Theta(s) @ Xi = observed
    model.fit(s, t=t_dummy, x_dot=y)

    # Predict on the same inputs
    x_model = model.predict(s)

    # Normalize prediction into a numeric 2-D array with shape (n_pred, n_states)
    x_model = np.asarray(x_model)
    if x_model.ndim == 1:
        x_model = x_model.reshape(-1, 1)
    elif x_model.ndim == 2:
        pass
    else:
        # If predict returned a ragged / object array, try to stack/pad rows
        try:
            x_model = np.vstack([np.asarray(row, dtype=float).reshape(1, -1) for row in x_model])
        except Exception:
            raise ValueError("model.predict returned an irregular/ragged result that can't be converted to a numeric array")

    # If the row count differs, resample/interpolate in index-space (fallback only)
    n_pred = x_model.shape[0]
    if n_pred != n_samples:
        # interpolate each column from index space [0..n_pred-1] to [0..n_samples-1]
        orig_idx = np.linspace(0.0, 1.0, n_pred)
        target_idx = np.linspace(0.0, 1.0, n_samples)
        x_resampled = np.zeros((n_samples, x_model.shape[1]), dtype=float)
        for col in range(x_model.shape[1]):
            sort_idx = np.argsort(orig_idx)
            ox = orig_idx[sort_idx]
            oy = x_model[sort_idx, col]
            x_resampled[:, col] = np.interp(target_idx, ox, oy)
        x_model = x_resampled

    # Return prediction with same shape as input observed
    if y_orig.ndim == 1:
        return x_model.ravel()
    else:
        return x_model

# Runner that loops through timestamps and produces modeled series ----------------
def run_loop_time_series(time: np.ndarray,
                         observed: np.ndarray,
                         model_step_fn: Callable,
                         params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loop over timestamps and compute model values using model_step_fn.

    Returns (model_array, final_state).
    """
    # Prepare and normalize parameters
    DeltaTime = float(params.get('DeltaTime', 0.0))
    Initial = float(params.get('Initial', 0.04294))
    Year = float(params.get('Year', 365.242))
    Imp_Stride = int(params.get('Imp_Stride', 12))
    Imp_Amp = float(params.get('Imp_Amp', 1.0))
    Periods = list(params.get('Periods', [1.0]))
    PeriodsAmp = list(params.get('PeriodsAmp', [1.0] * len(Periods)))
    PeriodsPhase = list(params.get('PeriodsPhase', [0.0] * len(Periods)))
    Aliased = list(params.get('Aliased', [1.0]))
    AliasedAmp = list(params.get('AliasedAmp', [1.0] * len(Aliased)))
    AliasedPhase = list(params.get('AliasedPhase', [0.0] * len(Aliased)))
    Hold = float(params.get('Hold', 0.5))
    LTE_Freq = float(params.get('LTE_Freq', 1.0))
    Delta_Y = float(params.get('Delta_Y', 0.0))
    Imp_Amp2 = float(params.get('Imp_Amp2', 0.0))
    Damp = float(params.get('Damp', 0.0))
    Harmonics = list(params.get('Harmonics', [1]))
    DC = float(params.get('DC', 0.0))
    Scale = float(params.get('Scale', 1.0))

    N = time.size
    clone = clone_series(observed)
    state: Dict[str, Any] = {}
    model = np.zeros_like(observed, dtype=float)
    model_sup = np.zeros_like(observed, dtype=float)
    First_Time = True
    Last_Time = 0.0
    for i in range(N):
        t_i = float(time[i]+DeltaTime)
        clone_i = float(clone[i])
        # should be 1/12 but may be missing values
        N_Steps = int(round((t_i - Last_Time) * 12))
        N_Steps = 1 if First_Time else N_Steps
        First_Time = False
        j = 1
        while j <= N_Steps:  
            time_i = float(t_i - (N_Steps-j)/12.0)
            v, state, sup = model_step_fn(i, time_i, clone_i, state, Initial, Year, Imp_Stride,Imp_Amp, Periods, PeriodsAmp, PeriodsPhase, 
                                     Aliased, AliasedAmp, AliasedPhase, Hold, Delta_Y, Imp_Amp2, Damp)
            j = j + 1

        model[i] = float(v)
        model_sup[i] = float(sup)
        Last_Time =  t_i

    if os.getenv('FORCING', '') == '1':
        model1 = Scale*model - DC
        return model1, state

    clone = clone - model_sup

    # lte = fit_sinusoidal_regression(model, clone, N_list=Harmonics, k=LTE_Freq, intercept=True, add_linear_x=True, ridge=None)
    # model1 = lte["predict"](model) + model_sup

    lte = fit_pysindy_with_sinusoidal_library(clone, LTE_Freq*model)
    model1 = lte + model_sup

    return model1, state, model


# Metrics -----------------------------------------------------------------------
def compute_metrics(time: np.ndarray, observed: np.ndarray, model: np.ndarray, low: float, high:float) -> Dict[str, Any]:
    """Compute Pearson correlation, residuals, MSE, and variance of squared errors."""
    residuals = model - observed
    mse = float(np.mean(residuals ** 2))
    # variance of squared errors (i.e., variance of residuals**2)
    var_sq_err = float(np.var(residuals ** 2))
    # Pearson r (use scipy if available else numpy)

    CV = 1.0 - compute_metrics_region(time, observed, model, low, high, False)

    try:
        r_val, p_val = pearsonr_truncate(observed, model)  # type: ignore
        pearson_r = float(r_val)
        pearson_p = float(p_val)
    except Exception:
        pearson_r = float(np.nan)
        pearson_p = float(np.nan)
    return {
        'residuals': residuals,
        'mse': mse,
        'var_squared_error': var_sq_err,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'CV': CV
    }

def compute_metrics_scalar(observed: np.ndarray, model: np.ndarray) -> Float:
    if False:
        """Compute CC"""
        r_val, p_val = pearsonr_truncate(observed, model)  # type: ignore
        pearson_r = 1.0-float(r_val)
        return pearson_r
    else:
        """Compute MSE"""
        residuals = model - observed
        mse = float(np.mean(residuals ** 2))
        return mse

def compute_metrics_region(time: np.ndarray,
                           observed: np.ndarray,
                           model: np.ndarray,
                           Time_Low: Optional[float] = None,
                           Time_High: Optional[float] = None,
                           Exclude: Optional[bool] = True) -> float:
    """
    Compute a scalar metric between observed and model. By default computes MSE.
    If Time_Low and Time_High are provided, values with Time_Low <= time <= Time_High
    are excluded from the calculation (useful for cross-validation / holdout).

    Minimal behavior, no heavy validation:
      - If both Time_Low and Time_High are None the full series is used.
      - If only one is provided it is treated as an open bound (i.e. exclude
        times >= Time_Low if Time_High is None, or <= Time_High if Time_Low is None).
      - If Time_Low > Time_High they are swapped.
      - If all points are excluded, returns float('nan').

    Parameters
    - time: 1D array of time values (same length as observed/model)
    - observed: observed values
    - model: model values
    - Time_Low: lower bound (inclusive) of the excluded middle region, or None
    - Time_High: upper bound (inclusive) of the excluded middle region, or None

    Returns
    - scalar metric (MSE by default)
    """
    # Build mask for points to INCLUDE in the metric (True => include)
    if Time_Low is None and Time_High is None:
        mask = np.ones_like(time, dtype=bool)
    else:
        # If one bound is missing, treat it as open-ended
        if Time_Low is None:
            # exclude times <= Time_High
            mask = time > float(Time_High)
        elif Time_High is None:
            # exclude times >= Time_Low
            mask = time < float(Time_Low)
        else:
            # ensure bounds are ordered
            tl = float(Time_Low)
            th = float(Time_High)
            if tl > th:
                tl, th = th, tl
            # include times strictly outside [tl, th]
            if Exclude:
                mask = (time < tl) | (time > th)
            else:
                mask = (time > tl) & (time < th)

    # Select data
    t_sel = time[mask]
    obs_sel = observed[mask]
    mod_sel = model[mask]

    # If nothing remains, return NaN
    if obs_sel.size == 0:
        return float('nan')

    # ---- choose metric ----
    # The original snippet had an unused pearson branch; keep MSE as default.
    # If you want Pearson-based metric, set use_pearson = True.
    global use_pearson
    if use_pearson:
        from scipy.stats import pearsonr  # type: ignore
        r_val, _ = pearsonr_truncate(obs_sel, mod_sel)
        pearson_r = 1.0 - float(r_val)
        return pearson_r
    else:
        residuals = mod_sel - obs_sel
        mse = float(np.mean(residuals ** 2))
        return mse



# Opt ---------------------------------------------------------------------------

Monitored = [
    "Imp_Stride",
    "DeltaTime",
    "Initial",
    "Year",
    "Imp_Amp",
    "Imp_Amp2",
    "PeriodsAmp",
    "PeriodsPhase",
    "AliasedAmp",
    "AliasedPhase",
    "Hold",
    "Delta_Y",
    "Damp",
    "Harmonics"
]

Monitored_Slide = [
    "LTE_Freq",
    "Imp_Amp",
    "Imp_Amp2",
    "Imp_Stride",
    "DeltaTime",
    "Initial",
    "PeriodsPhase",
    "AliasedPhase",
    "Damp",
    "DC",
    "Scale"
]

Monitored_Initial = [
    "Imp_Stride",
    "DeltaTime",
    "Initial"
]


def _is_list_param(name: str) -> bool:
    return name in {
        "Periods",
        "PeriodsAmp",
        "PeriodsPhase",
        "Aliased",
        "AliasedAmp",
        "AliasedPhase"
    }


def simple_descent(
    time,
    observed,
    model_step_fn: Callable,
    params: Dict[str, Any],
    metric_fn: Callable[[Any, Any, Any, float, float], float],
    step: float = 0.01,
    max_iters: int = 100,
    tol: float = 1e-12,
    verbose: bool = False,
    Time_Low: Float = 1920.0,
    Time_High: Float = 1950
) -> Dict[str, Any]:
    """
    Simple descent optimizer.

    Args:
      - time, observed, model_step_fn: as in run_loop_time_series
      - params: dict with the monitored parameters (scalars and lists of floats)
      - metric_fn: function(observed, model_array) -> scalar (smaller is better)
      - step: base step size used for +/- changes (applied to scalars and list elements)
      - max_iters: max number of outer passes over all parameters
      - tol: minimum improvement to consider (metric must decrease by > tol)
      - verbose: print progress

    Returns dict:
      - best_params: dict of accepted parameter values
      - best_metric: scalar
      - best_model: model array for best params
      - best_state: final state from run_loop_time_series for best params
      - history: list of evaluated candidates: dicts with keys
                 ('param', 'index' (or None), 'candidate', 'metric', 'accepted')
      - iterations: number of outer iterations performed
    """

    if os.getenv('ALIGN', '') == '1':
        Monitored_Parameters = Monitored_Slide
        Magnify_Scalar = 10.0
    elif os.getenv('ALIGN', '') == '-1':
        Monitored_Parameters = Monitored_Initial
        Magnify_Scalar = 1.0
    else:
        Monitored_Parameters = Monitored
        Magnify_Scalar = 1.0

    # start from given params (copy to avoid mutating caller's dict)
    best_params = copy.deepcopy(params)
    # ensure list parameters are plain python lists of floats (if present)
    for name in Monitored_Parameters:
        if _is_list_param(name) and name in best_params:
            best_params[name] = [float(x) for x in best_params[name]]

    # evaluate initial
    model_vals, final_state, _ = run_loop_time_series(time, observed, model_step_fn, best_params)
    best_metric = float(metric_fn(time, observed, model_vals, Time_Low, Time_High))

    history: List[Dict[str, Any]] = []
    if verbose:
        print(f"[{time_series_name}] start metric={best_metric:.6g}")

    switched_to_rel = os.getenv('RELATIVE', '') == '1'

    for iteration in range(1, max_iters + 1):
        any_accepted = False
        mode = "rel" if switched_to_rel else "abs"
        if verbose:
            print(f"[{time_series_name}] iteration {iteration} mode={mode} best_metric={best_metric:.6g}")

        # iterate over monitored params in fixed order
        for name in Monitored_Parameters:
            if name == "Imp_Stride":
                max_val = 12
                trials_per_iter = 4
                # sample without replacement when possible
                population = list(range(0, max_val + 1))
                k = min(trials_per_iter, len(population))
                sampled = random.sample(population, k) if k > 0 else []
                for cand in sampled:
                    # skip if same as current
                    if int(best_params.get(name, 0)) == cand:
                        continue
                    candidate_params = copy.deepcopy(best_params)
                    candidate_params[name] = int(cand)
                    mvals_cand, _, _ = run_loop_time_series(time, observed, model_step_fn, candidate_params)
                    metric_cand = float(metric_fn(time, observed, mvals_cand, Time_Low, Time_High))
                    accepted = metric_cand + tol < best_metric
                    history.append({
                        "param": name,
                        "index": None,
                        "candidate": cand,
                        "metric": metric_cand,
                        "accepted": accepted,
                        "method": "discrete_random",
                    })
                    if accepted:
                        best_params = candidate_params
                        best_metric = metric_cand
                        model_vals = mvals_cand
                        any_accepted = True
                        if verbose:
                            print(f"[stride] accepted {name} = {cand} -> metric {best_metric:.6g}")
                        # keep sampling further candidates this iteration (could accept more)
                continue

            if name == "Harmonics":
                max_val = 100
                trials_per_iter = 10
                # sample without replacement when possible
                population = list(range(0, max_val + 1))
                k = min(trials_per_iter, len(population))
                sampled = random.sample(population, k) if k > 0 else []
                for cand in sampled:
                    # skip if same as current

                    lst = best_params.get(name)
                    if not lst:
                        continue
                    # iterate elements
                    for idx in range(len(lst)):
                        cur = int(lst[idx])

                        if cur == cand:
                            continue
                        if cur == 1:  # Leave fundamental alone
                            continue
                        candidate_params = copy.deepcopy(best_params)
                        candidate_params[name][idx] = int(cand)
                        mvals_cand, _, _ = run_loop_time_series(time, observed, model_step_fn, candidate_params)
                        metric_cand = float(metric_fn(time, observed, mvals_cand, Time_Low, Time_High))
                        accepted = metric_cand + tol < best_metric
                        history.append({
                            "param": name,
                            "index": idx,
                            "candidate": cand,
                            "metric": metric_cand,
                            "accepted": accepted,
                            "method": "discrete_random",
                        })
                        if accepted:
                            best_params = candidate_params
                            best_metric = metric_cand
                            model_vals = mvals_cand
                            any_accepted = True
                            if verbose:
                                print(f"[harmonic] accepted {name} = {cand} -> metric {best_metric:.6g}")
                               # keep sampling further candidates this iteration (could accept more)
                continue

            if _is_list_param(name):
                # skip if param not present or empty
                lst = best_params.get(name)
                if not lst:
                    continue
                # iterate elements
                for idx in range(len(lst)):
                    cur = float(lst[idx])
                    accepted_this_element = False
                    for sign in (1.0, -1.0):
                        candidate_params = copy.deepcopy(best_params)
                        # candidate_params[name][idx] = cur + delta*cur
                        if use_random:
                            delta = -step * math.log(random.random())
                        else:
                            delta = step
                        if not switched_to_rel:
                            # absolute candidate
                            candidate = cur + sign * delta
                        else:
                            # relative candidate (fallback to additive if cur==0)
                            if abs(cur) > 0.0:
                                candidate = cur * (1.0 + sign * delta)
                            else:
                                candidate = cur + sign * delta
                        candidate_params[name][idx] = candidate
                        model_vals_cand, _, _ = run_loop_time_series(time, observed, model_step_fn, candidate_params)
                        metric_cand = float(metric_fn(time, observed, model_vals_cand, Time_Low, Time_High))
                        accepted = metric_cand + tol < best_metric
                        history.append({
                            "param": name,
                            "index": idx,
                            "candidate": candidate_params[name][idx],
                            "metric": metric_cand,
                            "accepted": accepted,
                        })
                        if accepted:
                            # keep the candidate
                            best_params = candidate_params
                            best_metric = metric_cand
                            model_vals = model_vals_cand
                            any_accepted = True
                            accepted_this_element = True
                            if verbose:
                                print(f"[{time_series_name}] accepted {name}[{idx}] = {best_params[name][idx]:.6g} -> metric {best_metric:.6g}")
                            # continue trying further +/- steps from the new value:
                            cur = float(best_params[name][idx])
                            break  # stop trying other sign at this pass; move to try again from updated cur
                    # end +/- loop for this element
                    # Optionally attempt additional immediate steps in same direction until no improvement:
                    # (Keep this simple implementation: we will revisit in next outer iteration)
                    pass
            else:
                # scalar parameter
                if name not in best_params:
                    continue
                cur = float(best_params[name])
                accepted_this_param = False
                for sign in (1.0, -1.0):
                    candidate_params = copy.deepcopy(best_params)
                    # candidate_params[name] = cur + delta*cur
                    if use_random:
                        delta = -step * math.log(random.random()) * Magnify_Scalar
                    else:
                        delta = step
                    if not switched_to_rel:
                        # absolute candidate
                        candidate = cur + sign * delta
                    else:
                        # relative candidate (fallback to additive if cur==0)
                        if abs(cur) > 0.0:
                            candidate = cur * (1.0 + sign * delta)
                        else:
                            candidate = cur + sign * delta
                    candidate_params[name] = candidate
                    model_vals_cand, _, _ = run_loop_time_series(time, observed, model_step_fn, candidate_params)
                    metric_cand = float(metric_fn(time, observed, model_vals_cand, Time_Low, Time_High))
                    history.append({
                        "param": name,
                        "index": None,
                        "candidate": candidate_params[name],
                        "metric": metric_cand,
                        "accepted": metric_cand + tol < best_metric,
                    })
                    if metric_cand + tol < best_metric:
                        best_params = candidate_params
                        best_metric = metric_cand
                        model_vals = model_vals_cand
                        any_accepted = True
                        accepted_this_param = True
                        if verbose:
                            CV = float(metric_fn(time, observed, model_vals_cand, Time_Low, Time_High, False))
                            print(f"[{time_series_name}] accepted {name} = {best_params[name]:.6g} -> metric {best_metric:.6g}  CV {CV:.6g} ")
                        # move on from this parameter (we'll potentially refine in the next iteration)
                        break
                # end +/- for scalar
        # end loop over parameters

        if not any_accepted:
            if not switched_to_rel:
                # switch strategies and continue searching with relative updates
                switched_to_rel = True
                if verbose:
                    print(f"[{time_series_name}] no acceptance with absolute steps; switching to relative steps")
                # do not stop yet; continue with next iteration in relative mode
                continue
            else:
                if verbose:
                    print(f"[{time_series_name}] no acceptance in iteration {iteration}; stopping early")
                return {
                    "best_params": best_params,
                    "best_metric": best_metric,
                    "best_model": model_vals,
                    "best_state": final_state,
                    "history": history,
                    "iterations": iteration,
                }

    # max iters reached
    if verbose:
        print(f"[{time_series_name}] reached max_iters={max_iters}, best_metric={best_metric:.6g}")
    return {
        "best_params": best_params,
        "best_metric": best_metric,
        "best_model": model_vals,
        "best_state": final_state,
        "history": history,
        "iterations": max_iters,
    }


# Utilities ---------------------------------------------------------------------
def parse_json_like_arg(s: str | None) -> Dict[str, Any]:
    """Parse a JSON-like string from CLI --params. Returns dict."""
    if s is None:
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --params: {e}")


def main():
    ap = argparse.ArgumentParser(description="Time-series modeler with per-timestamp loop and metrics.")
    ap.add_argument('csv', help='Input CSV file (two columns: time, value).')
    ap.add_argument('--out', default='fitted_out.csv', help='Output CSV file with time, observed, model, residual.')
    ap.add_argument('--plot', action='store_true', help='Show plot of observed vs model (requires matplotlib).')
    ap.add_argument('--low', default=0.0)
    ap.add_argument('--high', default=3000.0)
    ap.add_argument('--cc', action='store_true')
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--scale', default=1.0)

    args = ap.parse_args()

    global use_pearson
    if args.cc:
        use_pearson = True
        print("using CC")
    else:
        use_pearson = False

    global use_random
    if args.random:
        use_random = True
        print("using random")
    else:
        use_random = False

    global time_series_name
    time_series_name = args.csv

    # Read CSV
    time, y = read_two_column_csv(args.csv)
    if time.size == 0:
        raise SystemExit("No data read from CSV.")
    y = normalize_rms_y(y)

    # Read JSON file with same name appended by ".p"
    json_path = args.csv + '.p'
    data_dict = read_json_p(json_path)
    params = data_dict

    model_fn = model_step_algorithm
    Low = float(args.low)
    High = float(args.high)

    # Make a clone of the series (not strictly necessary but requested)
    cloned = clone_series(y)

    result = simple_descent(
        time, y, model_fn, params,
        metric_fn=compute_metrics_region,
        step=float(os.environ.get('STEP', '0.05')), 
        max_iters=int(os.environ.get('MAX_ITERS', '400')), 
        tol=1e-12, verbose=True, Time_Low=Low, Time_High=High
    )
    params = result["best_params"]

    # Run the time-stepping loop (explicit loop per timestamp)
    model_vals, final_state, forcing = run_loop_time_series(time, cloned, model_fn, params)

    # Compute metrics
    metrics = compute_metrics(time, y, model_vals, Low, High)

    # Prepare output DataFrame
#    out_df = pd.DataFrame({
#        'time': time,
#        'observed': y,
#        'model': model_vals,
#        'forcing' : forcing
#        'residual': metrics['residuals']
#    })
#    out_df.to_csv(args.out, index=False)

    # Update data dictionary with metadata and save to .p file
    data_dict_out = dict(params)  # copy new

    write_json_p(json_path, data_dict_out)

    # Print summary
    print(f"Processed {time.size} samples from: {args.csv}")
    print(f"Output CSV: {args.out}")
    print(f"Updated JSON data file: {json_path}")
    print("Metrics:")
    print(f"  CV = {metrics['CV']:.6e}")
    print(f"  MSE = {metrics['mse']:.6e}")
    print(f"  Pearson r = {metrics['pearson_r']}, p-value = {metrics.get('pearson_p', None)}")

    # Optional plotting
    mask = (time > Low) & (time < High)
    plt.figure(figsize=(12, 6))
    plt.plot(time, y, label='observed', lw=0.5)
    plt.plot(time, float(args.scale)*model_vals, label='model', lw=0.5, color='red')
    # plt.plot(time, metrics['residuals'], label='residual', lw=0.8, alpha=0.7)
    plt.plot(time[mask], [0.0] * len(time[mask]), 'k--', label='cross-validation', linewidth=3)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title(f"Model: {args.csv}  Pearson r={metrics['pearson_r']:.4g}  CV={metrics['CV']:.4g}")
    # plt.grid(True)
    if args.plot:
        plt.show()
    else:
        plt.savefig(args.csv+'-'+args.low+'-'+args.high+'.png', bbox_inches='tight')  # Save as PNG with tight bounding box


if __name__ == '__main__':
    main()
