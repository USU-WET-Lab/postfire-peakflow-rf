
"""
02_optimize_withholding.py - Step 2 of the post-fire peak flow pipeline

Sweeps train/test witholding percentages across all seeds generated in step 01. 
For each seed, a random forest is trained then evaluated on the test watersheds. 
The R2 and standard deviation of each seed is recorded and saved to a csv for each witholding percentage.

Reads: outputs/Seeds/<witholding>/Seed_<x>/wats_test.csv, outputs/Seeds/<witholding>/Seed_<x>/wats_train.csv

Writes: outputs/WithholdingOptimization/<witholding>.csv

Run: python src/02_optimize_withholding.py
"""
#-------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# -------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent # repo root; auto-derives, no need to edit
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'

METRIC = "PeakArea"  #modeling target (log-transformed) 
N_SEEDS = 100  # number of random train/test splits to generate
WITHOLDINGS = ['50_50', '60_40',  '70_30', '80_20', '90_10']  # list of witholding percentages to sweep across
RF_KWARGS = dict(n_estimators=100, random_state=42)  # keyword arguments for the random forest regressor

#model domain of applicability bounds (applied to the source table before splitting into train/test seeds, see README for details)
BOUNDS = dict(min_drain_sqkm = 50, min_burned_storm_depth_per = 70, 
              max_days_since_fire = 1095, min_mtbs_burnedarea_per = 20) 

#--------------------------------------------MAIN CODE BLOCK ---------------------------------------------------------------------------

def main (): 
    df = pd.read_csv(DATA / SOURCE_TABLE )
    df = df[df.DRAIN_SQKM >= BOUNDS["min_drained_area_pct"]] & (df.burned_storm_depth_per >=
                            BOUNDS["min_burned_storm_dept_pct"]) & (df.DaysSinceFire <= BOUNDS["max_days_since_fire"])]
    features = [c for c in df.columns if c not in (METRIC, ID_COL)]

    out_dir = OUTPUTS / "witholding_optimization" 
    out_dir.mkdir(exist_ok= True, parents = True)

    for witholding in WITHOLDINGS: 
        rows = []
        for x in range(N_SEEDS): 
            seed_dir = OUTPUTS / "seeds" / witholding / f"Seed_{x}"
            train_ids = pd.read_csv(seed_dir / "wats_train.csv")[ID_COL].tolist()
            tests_ids = pd.read_csv(seed_dir / "wats_test.csv")[ID_COL].tolist()
            train = df[df[ID_COL].isin(train_ids)]
            test = df[df[ID_COL].isin(test_ids)]
            if test.empty: 
                continue

            model = fit_rf(train[features], np.log(train[METRIC]), **RF_KWARGS)
            stats = evaluate(np.log(test[METRIC]), model.predict(test[features]))
            rows.append({"seed": f"Seed_{x}", "R2": stats["R2"]})

            summary = pd.DataFrame(rows)
            out_path = out_dir / f"witholding_{witholding}.csv"
            summary.to_csv(out_path, index = False)
            print(f"{witholding}: mean R2 = {summary['R2'].mean():.3f}"
                  f"over {len(summary)} seeds")

if __name__ == "__main__": 
    main()

