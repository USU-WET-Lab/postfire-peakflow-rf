"""
07_run_sensitivity.py - Step 07 in the post-fire peak flow modeling pipeline

This step is a sensitiviy analysis for a single watershed wjere we sweep each driver of interest one at a time 
while holding all others at baseline, then predict the peak for evey synthetic scenario using all n trained models.
The spread pf predictions across seeds shows how sensitive the modeled peak flow is to each driver. If you are 
familiar with Individual Conditional Expectation (ICE) plots you will see the resemblance. The key differences are that 
we create the curve of ONE instance (watershed) and the loop varies the model, not the instance. See step 10 for the 
complementary, dataset-wide SHAP dependence view. 

Reads:  data/<MODEL_TABLE>, data/<SENSITIVITY_TABLE>
        outputs/seeds/<WITHOLDING>/Seed_<x>/wats_train.csv

Writes: outputs/sensitivity/sensitivity_predictions_USGS<WATERSHED>.csv

"""


#-----------------------------------------------------IMPORTS-----------------------------------------------------------
from pathlib import Path 
import numpy as np 
import pandas as pd 
from rf_utils import fit_rf

#-------------------------------------------------CONFIG--------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent # repo root; auto-derives, no need to edit
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'

MODEL_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv" # training data
SENSITIVITY_TABLE = "RF_AttributeTable_Sensitivity_9361000.csv" #baseline scenario
WATERSHED = 9361000
METRIC = "PeakArea"
ID_COL = "GAGE_ID"
N_SEEDS = 100
WITHHOLDING = "80_20"
RF_KWARGS = dict(n_estimators = 100, random_state = 42) 

#Drivers to sweep, and the values to try for each. Every scenario changes one of these, all other attributes stay at the baseline. Edit if you so desire
SWEEPS = {
    "burned_storm_depth_per":           [70, 75, 80, 85, 90, 95, 100],
    "burned_storm_int_per":             [70, 75, 80, 85, 90, 95, 100],
    "MTBS_burnedarea_per":              [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "SBS_mod_high_area_burned_per_wat": [10, 20, 30, 40, 50, 60, 70, 80],
    "AntecedentPrecip_mm":              [0, 12.7, 25.4, 38.1, 50.8, 63.5,
                                         76.2, 88.9, 101.6, 114.3, 127],
    "AvgMonthly_Precip_cm":             [1, 5, 10, 15, 20, 25, 30, 35],
}

#----------------------------------------------Scenario construction---------------------------------------------
def build_sensitivity_set(baseline: pd.DataFrame) -> pd.DataFrame: 
    blocks = [baseline.assign(swept_variable = "baseline", swept_value = np.nan)]
    for feature, values in SWEEPS.items():
        for _, base in baseline.iterrows(): 
            block = pd.concat([base.to_frame().T] * len(values), ignore_index=True)
            block[feature] = values
            block["swept_variable"] = feature
            block["swept_value"] = values
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)

#-----------------------------------------------main-----------------------------------------------------------------

def main():
    # 1. Build the synthetic scenario set from the baseline attributes.
    baseline  = pd.read_csv(DATA / SENSITIVITY_TABLE)
    scenarios = build_sensitivity_set(baseline)

    # 2. Load training data and pin the feature columns (order matters for sklearn).
    data     = pd.read_csv(DATA / MODEL_TABLE)
    features = [c for c in data.columns if c not in (METRIC, ID_COL)]
    X_scen   = scenarios[features].astype(float)   # selecting `features` drops metadata

    # 3. Train one model per seed and predict every scenario (in real units).
    preds = pd.DataFrame(index=scenarios.index)
    for x in range(N_SEEDS):
        train_ids = pd.read_csv(
            OUTPUTS / "seeds" / WITHHOLDING / f"Seed_{x}" / "wats_train.csv"
        )[ID_COL].to_list()

        # Keep the watershed of interest out of training (predict it out-of-sample).
        train_ids = [w for w in train_ids if w != WATERSHED]
        train = data[data[ID_COL].isin(train_ids)]

        model = fit_rf(train[features], np.log(train[METRIC]), **RF_KWARGS)
        preds[f"pred_seed_{x}"] = np.exp(model.predict(X_scen))   # log -> m3/s/km2

    # 4. Summarize the spread across seeds and save one tidy table.
    scenarios["pred_mean"] = preds.mean(axis = 1)
    scenarios["pred_p05"]  = preds.quantile(0.05, axis=1)
    scenarios["pred_p95"]  = preds.quantile(0.95, axis=1)

    out_dir = OUTPUTS / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sensitivity_predictions_USGS{WATERSHED}.csv
    pd.concat([scenarios, preds], axis=1).to_csv(out_path, index=False)

    print(f"USGS{WATERSHED}: {len(scenarios)s -> {out_path}")


if __name__ == "__main__":
    main()

    