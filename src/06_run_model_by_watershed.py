"""
06_run_model_by_watershed.py - Step 6 of the post-fire peak flow pipeline

The same model as step 05, but scored one test watershed at a time. Step 05 reports a single R2
for a seed's whole test set, pooled over ~30 watersheds; this step evaluates each held-out
watershed on its own, so you can ask which watersheds the model handles well and what drives its
predictions at a specific gage. Steps 08 and 11 summarize the results.

Within a seed the training set is the same no matter which test watershed you are scoring, so the
forest is fit ONCE per seed and then applied to each test watershed in turn. The original version
refit an identical model for every watershed; the predictions are unchanged, it is just ~30x less
work per seed.

Note on per-watershed R2: with only a handful of storms at one gage, the denominator of R2 is
computed over very few points, so negative values are common and expected. They mean the model
did worse than predicting that watershed's own mean, which is a benchmark it never had access to.
Read these as relative comparisons between watersheds, not as absolute skill.

Reads: data/<MODEL_TABLE>
       outputs/seeds/<WITHHOLDING>/Seed_<x>/wats_train.csv, wats_test.csv

Writes: outputs/model_run_watersheds/Seed_<x>/importances.csv                       (Feature, Importance)
        outputs/model_run_watersheds/Seed_<x>/Watershed_USGS<id>/stats.csv          (mse, rmse, R2)
        outputs/model_run_watersheds/Seed_<x>/Watershed_USGS<id>/predictions.csv    (GAGE_ID, y_test, y_pred; log space)
        outputs/model_run_watersheds/Seed_<x>/Watershed_USGS<id>/shap_values.csv, shap_data.csv

Run: python src/06_run_model_by_watershed.py

"""


#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
from rf_utils import (load_model_table, fit_rf, evaluate, ranked_importances,
                      compute_shap, shap_frames)

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

MODEL_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"   # built by step 04b
METRIC      = "PeakArea"
ID_COL      = "GAGE_ID"
N_SEEDS     = 100
WITHHOLDING = "80_20"
RF_KWARGS   = dict(n_estimators=100, random_state=42)

# Feature selection lives in config/selected_features.txt (step 04b), so this run and step 05
# model the same thing by construction -- there is nothing to keep in step by hand.

#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    data = load_model_table(DATA / MODEL_TABLE)
    features = [c for c in data.columns if c not in (METRIC, ID_COL)]

    for x in range(N_SEEDS):
        seed_dir = OUTPUTS / "model_run_watersheds" / f"Seed_{x}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        split_dir = OUTPUTS / "seeds" / WITHHOLDING / f"Seed_{x}"
        train_ids = pd.read_csv(split_dir / "wats_train.csv")[ID_COL].tolist()
        test_ids  = pd.read_csv(split_dir / "wats_test.csv")[ID_COL].tolist()

        # One fit per seed; the training set does not depend on which watershed we score.
        train = data[data[ID_COL].isin(train_ids)]
        model = fit_rf(train[features], np.log(train[METRIC]), **RF_KWARGS)
        ranked_importances(model, features).to_csv(seed_dir / "importances.csv", index=False)

        scored = 0
        for watershed in test_ids:
            test = data[data[ID_COL] == watershed]
            if test.empty:
                continue   # no storms for this watershed inside the model bounds

            watershed_dir = seed_dir / f"Watershed_USGS{watershed}"
            watershed_dir.mkdir(parents=True, exist_ok=True)

            y_test = np.log(test[METRIC])
            y_pred = model.predict(test[features])

            stats = evaluate(y_test, y_pred)
            pd.DataFrame([stats]).to_csv(watershed_dir / "stats.csv", index=False)

            pd.DataFrame({ID_COL: test[ID_COL].to_numpy(),
                          "y_test": y_test.to_numpy(),
                          "y_pred": y_pred}).to_csv(watershed_dir / "predictions.csv", index=False)

            # SHAP for just this watershed's storms; TreeSHAP is per-row, so these are the same
            # attributions step 05 produces, restricted to this gage.
            values, shap_data = shap_frames(compute_shap(model, test[features]))
            values.to_csv(watershed_dir / "shap_values.csv", index=False)
            shap_data.to_csv(watershed_dir / "shap_data.csv", index=False)
            scored += 1

        print(f"Seed_{x}: scored {scored} test watersheds")

    print(f"Done! Wrote {N_SEEDS} seed runs to {OUTPUTS / 'model_run_watersheds'}")


if __name__ == "__main__":
    main()
