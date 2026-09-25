"""
rf_utils.py - Shared helpers for the post-fire peak flow pipeline

The code that lives here is called by multiple steps within the pipeline. Only mechanical, repeated code lives here.
The main functions here perform the model fitting, scoring, raking importances, computing SHAP values,
reloading saved SHAP values, and generating two standard diagnostic plots.

Imported by steps 01, 02, 03, 04b, 05, 06, 07, 09, 10, and 11

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

# ---------------------------------------Feature Selection-------------------------------------------------------
def read_selection(path): 
    """Read a config feature list: one feature per line, '#' comments a line out."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Selection file {path} does not exist. Did you delete it?")
    stripped = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    features = [line for line in stripped if line]
    if not features:
        raise ValueError(f"No features found in selection file {path}.")
    # One feature per LINE. A comma means several names landed on one line, which would
    # otherwise sail through as a single nonexistent column name.
    commas = [f for f in features if "," in f]
    if commas:
        raise ValueError(f"One feature per line in {path}; these hold several: {commas}")
    return features

def table_loader(path, features, id_col = "GAGE_ID", metric = "PeakArea"): 
    """Load a CSV table from the given path, takes a feature list and subsets the table to those features plus the ID column'"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    missing = [c for c in [id_col, metric] + features if c not in df.columns]
    if missing: 
        raise ValueError(f"requested columns are not in {path}: {missing}")
    return df[[id_col, metric] + features]

#---------------------------------------Model-----------------------------------------------------------------

# A model bounds filter and dict 

#DICTIONARY OF BOUNDS:
BOUNDS = dict(min_drain_sqkm = 50, min_burned_area_pct = 20, min_burned_storm_depth_pct = 70, 
              max_days_since_fire = 1095)

def filter_bounds(df, bounds = BOUNDS):
    """Filter a dataframe to the bounds specified in the bounds dict."""
    return df[(df.DRAIN_SQKM >= bounds['min_drain_sqkm']) &
              (df.MTBS_burnedarea_per >= bounds['min_burned_area_pct']) &
              (df.burned_storm_depth_per >= bounds['min_burned_storm_depth_pct']) &
              (df.DaysSinceFire <= bounds['max_days_since_fire'])]  
# RF model fitting and evaluation functions

def fit_rf(X_train, y_train, **kwargs): 
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train) 
    return model 

def evaluate(y_true, y_pred): 
    mse = mean_squared_error(y_true, y_pred)
    return {"mse": mse, "rmse": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}

def ranked_importances(model, feature_names) -> pd.DataFrame: 
    return(pd.DataFrame({"Feature": list(feature_names), 
                        "Importance": model.feature_importances_})
            .sort_values("Importance", ascending = False)
            .reset_index(drop = True)) 

def load_model_table(path): 
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Please run step 04b to build the model table first.")
    return pd.read_csv(path)

#---------------------------------------SHAP------------------------------------------------------------------
def compute_shap(model, X): 
    explainer = shap.TreeExplainer(model)
    return explainer(X)

def shap_frames(shap_values):
    values = pd.DataFrame(shap_values.values, columns= shap_values.feature_names)
    data = pd.DataFrame(shap_values.data, columns = shap_values.feature_names)
    return values, data

def seed_shap_frames(run_dir, n_seeds, leaf = "", with_data = False):
    """Read each seed's saved SHAP table under <run_dir>/Seed_<x>/<leaf>/.

    `leaf` is the per-seed subfolder, used by the by-watershed run (e.g. "Watershed_USGS9361000").
    Seeds with no SHAP output are skipped, so the result can be shorter than n_seeds, and empty
    when the model step has not been run. Returns a list of values frames, or a list of
    (values, data) pairs when with_data is True.
    """
    frames = []
    for x in range(n_seeds):
        seed_dir = Path(run_dir) / f"Seed_{x}"
        if leaf:
            seed_dir = seed_dir / leaf
        if not (seed_dir / "shap_values.csv").exists():
            continue

        values = pd.read_csv(seed_dir / "shap_values.csv")
        if with_data:
            frames.append((values, pd.read_csv(seed_dir / "shap_data.csv")))
        else:
            frames.append(values)
    return frames

def load_pooled_shap(run_dir, n_seeds, leaf = "", with_data = False):
    """Stack every seed's SHAP table: one row per scored storm per seed, one column per feature.

    ignore_index puts values and data on a shared 0..N index so they stay aligned row for row.
    Returns the pooled values frame, or (values, data) when with_data is True.
    """
    frames = seed_shap_frames(run_dir, n_seeds, leaf, with_data)
    if not frames:
        raise FileNotFoundError(f"No SHAP values found under {run_dir}. Run the model step first.")

    if with_data:
        return (pd.concat([values for values, _ in frames], ignore_index = True),
                pd.concat([data for _, data in frames], ignore_index = True))
    return pd.concat(frames, ignore_index = True)


#---------------------------------------Diagnostic Plots-------------------------------------------------------

def plot_predicted_vs_actual(y_true, y_pred, path, back_transform = False): 
    if back_transform: 
         y_true, y_pred = np.exp(y_true), np.exp(y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2)          # 1:1 line
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_residuals(y_pred, residuals, path): 
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.scatter(y_pred, residuals, alpha= 0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)