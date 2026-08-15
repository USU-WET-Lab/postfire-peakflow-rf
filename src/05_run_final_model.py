"""
05_run_final_model.py - Step 5 of the post-fire peak flow pipeline

Using the optimized feature table from step 04b, train the final model(s). This step trains one
random forest per seed (with the train/test split from step 01) and for each seed: writes
performance stats, test predictions, feature importances, two diagnostic plots, and SHAP values.
The per-seed outputs from this step feed subsequent summary steps (08, 09, 10). The target
is log-transformed for fitting in log space.

Feature selection does not happen here. Pick the features in config/selected_features.txt
(step 04), build the table with step 04b, and this script models every column in that table
that is not the target or the ID. Run step 04b before this one.

Reads: data / <MODEL_TABLE>
       outputs/seeds/<WITHHOLDING>/Seed_<x>/wats_train.csv, wats_test.csv

Writes: outputs/model_run/Seed_<x>/stats.csv          (mse, rmse, R2)
        outputs/model_run/Seed_<x>/predictions.csv    (GAGE_ID, y_test, y_pred; log space)
        outputs/model_run/Seed_<x>/importances.csv    (Feature, Importance)
        outputs/model_run/Seed_<x>/shap_values.csv, shap_data.csv
        outputs/model_run/Seed_<x>/predicted_vs_actual.png, residuals.png

Run: python src/05_run_final_model.py

"""


#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
from rf_utils import (load_model_table, fit_rf, evaluate, ranked_importances, compute_shap,
                      shap_frames, plot_predicted_vs_actual, plot_residuals)

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

MODEL_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"  # final optimized feature set
METRIC      = "PeakArea"
ID_COL      = "GAGE_ID"
N_SEEDS     = 100
WITHHOLDING  = "80_20"
RF_KWARGS   = dict(n_estimators=100, random_state=42)

#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    data = load_model_table(DATA / MODEL_TABLE)
    features = [c for c in data.columns if c not in (METRIC, ID_COL)]

    for x in range(N_SEEDS):
        seed_dir = OUTPUTS / "model_run" / f"Seed_{x}"
        seed_dir.mkdir(parents= True, exist_ok= True)

        split_dir = OUTPUTS / "seeds" / WITHHOLDING / f"Seed_{x}"
        train_ids = pd.read_csv(split_dir / "wats_train.csv")[ID_COL].tolist()
        test_ids  = pd.read_csv(split_dir / "wats_test.csv")[ID_COL].tolist()

        train = data[data[ID_COL].isin(train_ids)]
        test  = data[data[ID_COL].isin(test_ids)]

        # Fit in log space; the target spans orders of magnitude across watersheds.
        y_train = np.log(train[METRIC])
        y_test  = np.log(test[METRIC])
        model   = fit_rf(train[features], y_train, **RF_KWARGS)
        y_pred  = model.predict(test[features])

        # Performance for this split.
        stats = evaluate(y_test, y_pred)
        pd.DataFrame([stats]).to_csv(seed_dir / "stats.csv", index=False)

        # Predictions stay in log space; the gage id rides along so later steps can group
        # by watershed without re-deriving the split.
        pd.DataFrame({ID_COL: test[ID_COL].to_numpy(),
                      "y_test": y_test.to_numpy(),
                      "y_pred": y_pred}).to_csv(seed_dir / "predictions.csv", index=False)

        ranked_importances(model, features).to_csv(seed_dir / "importances.csv", index=False)

        # Diagnostic plots. Predicted vs. actual is back-transformed to m3/s/km2; residuals
        # stay in log space, where the model's error is actually symmetric.
        plot_predicted_vs_actual(y_test, y_pred, seed_dir / "predicted_vs_actual.png",
                                 back_transform=True)
        plot_residuals(y_pred, y_test - y_pred, seed_dir / "residuals.png")

        # SHAP is the slow part of this step.
        values, shap_data = shap_frames(compute_shap(model, test[features]))
        values.to_csv(seed_dir / "shap_values.csv", index=False)
        shap_data.to_csv(seed_dir / "shap_data.csv", index=False)

        print(f"Seed_{x}: R2 = {stats['R2']:.3f} over {len(test)} test storms")

    print(f"Done! Wrote {N_SEEDS} seed runs to {OUTPUTS / 'model_run'}")


if __name__ == "__main__":
    main()
