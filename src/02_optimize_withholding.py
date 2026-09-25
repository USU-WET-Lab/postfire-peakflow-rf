
"""
02_optimize_withholding.py - Step 2 of the post-fire peak flow pipeline

Sweeps train/test withholding percentages across all seeds generated in step 01. 
For each seed, a random forest is trained and subsequently evaluated on the test watersheds. 
The R2 and standard deviation of each seed is recorded and saved to a csv for each withholding percentage.

Reads: outputs/seeds/<withholding>/Seed_<x>/wats_test.csv, outputs/seeds/<withholding>/Seed_<x>/wats_train.csv

Writes: outputs/withholding_optimization/withholding_<withholding>.csv

Run: python src/02_optimize_withholding.py
"""
#-------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
from rf_utils import fit_rf, evaluate, table_loader, filter_bounds, read_selection

# -------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent # repo root; auto-derives, no need to edit
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
FEATURES_FILE = "config/feature_list.txt"  # the feature list used in steps 02 and 03, used here to check that the source table has all the features needed for modeling
SOURCE_TABLE = "data/RF_AttributeTable_PeakMag_FullDataset.csv"  # the source table to draw watersheds from, must be in data/
METRIC = "PeakArea"  #modeling target (log-transformed)
ID_COL = "GAGE_ID"  # watershed key; dropped from the feature set before training the RF
N_SEEDS = 100  # number of random train/test splits generated in step 01, leave as is unless step 1 n-seeds was changed
WITHHOLDINGS = ['50_50', '60_40',  '70_30', '80_20', '90_10']  # list of withholding percentages to sweep across
RF_KWARGS = dict(n_estimators=100, random_state=42)  # keyword arguments for the random forest regressor

#model domain of applicability bounds (applied to the source table before splitting into train/test seeds, see README and Paper for details)
BOUNDS = dict(min_drain_sqkm = 50, min_burned_storm_depth_per = 70, 
              max_days_since_fire = 1095, min_mtbs_burnedarea_per = 20) 

#--------------------------------------------MAIN CODE BLOCK ---------------------------------------------------------------------------

def main (): 
    df = pd.read_csv(DATA / SOURCE_TABLE )
    df = df[(df.DRAIN_SQKM >= BOUNDS["min_drain_sqkm"]) &
            (df.MTBS_burnedarea_per >= BOUNDS["min_mtbs_burnedarea_per"]) &
            (df.burned_storm_depth_per >= BOUNDS["min_burned_storm_depth_per"]) &
            (df.DaysSinceFire <= BOUNDS["max_days_since_fire"])]
    features = 

    out_dir = OUTPUTS / "withholding_optimization" 
    out_dir.mkdir(exist_ok= True, parents = True)

    for withholding in WITHHOLDINGS: 
        rows = []
        for x in range(N_SEEDS):
            seed_dir = OUTPUTS / "seeds" / withholding / f"Seed_{x}"
            if not seed_dir.exists():
                continue   # step 01 has not been run for this withholding percentage

            train_ids = pd.read_csv(seed_dir / "wats_train.csv")[ID_COL].tolist()
            test_ids = pd.read_csv(seed_dir / "wats_test.csv")[ID_COL].tolist()
            train = df[df[ID_COL].isin(train_ids)]
            test = df[df[ID_COL].isin(test_ids)]
            if test.empty:
                continue

            model = fit_rf(train[features], np.log(train[METRIC]), **RF_KWARGS)
            stats = evaluate(np.log(test[METRIC]), model.predict(test[features]))
            rows.append({"seed": f"Seed_{x}", "R2": stats["R2"]})

        if not rows:
            print(f"{withholding}: no seeds found, run step 01 for this withholding first")
            continue

        summary = pd.DataFrame(rows)
        out_path = out_dir / f"withholding_{withholding}.csv"
        summary.to_csv(out_path, index = False)
        print(f"{withholding}: mean R2 = {summary['R2'].mean():.3f} "
              f"over {len(summary)} seeds")

if __name__ == "__main__": 
    main()

