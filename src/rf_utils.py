"""
rf_utils.py - Shared helpers for the post-fire peak flow pipeline

The code that lives here is called by multiple steps within the pipeline. Only mechanical, repeated code lives here. 
The main functions here perform the model fitting, scoring, raking importances, computing SHAP values, and 
generating two standard diagnostic plots. 

Imported by steps 02, 03, 05, 06, and 07 

"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap 

#---------------------------------------Model-----------------------------------------------------------------
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

#---------------------------------------SHAP------------------------------------------------------------------
def compute_shap(model, X): 
    explainer = shap.TreeExplainer(model)
    return explainer(X)

def shap_frames(shap_values): 
    values = pd.DataFrame(shap_values.values, columns= shap_values.feature_names)
    data = pd.DataFrame(shap_values.data, columns = shap_values.feature_names)
    return values, data


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