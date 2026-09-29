"""
03_optimize_variables.py - Step 3 of the post-fire peak flow pipeline

Recursive feature elimination, run independently for each seed generated in step 01. 
Starting from the set of features shipped with this repo, train a RF, record the performance and the feature-importance ranking, 
drop the least important feature, and repeat down to a single feature. For each seed, the R2 and stdev of each
iteration is recorded and saved to a csv, along with the feature importance ranking. This step demonstrates how 
model performance changes as features are removed, and which features are most important, informing the final feature set used in step 04.

Feature selection itself is a manual step performed AFTER this script is run, see README and step 04 for details.

Warning: this script is computationally expensive and will take a long time to run. 

Reads: data/<SOURCE_TABLE> 
        outputs/seeds/<WITHHOLDING>/seed_<x>/wats_train.csv, wats_test.csv

Writes: outputs/variable_optimization/Seed_<x>/Stats_Vars<n>.csv
        outputs/variable_optimization/Seed_<x>/Importances_Vars<n>.csv 

Run: python src/03_optimize_variables.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
from rf_utils import fit_rf, evaluate, ranked_importances, table_loader, filter_bounds, read_selection

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent # repo root; auto-derives, no need to edit
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
FEATURES_FILE = "config/feature_list.txt"  # the feature list used in steps 02 and 03, used here to check that the source table has all the features needed for modeling
SOURCE_TABLE = "data/RF_AttributeTable_PeakMag_FullDataset.csv"  # the source table to draw watersheds from, must be in data/
SOURCE_TABLE = 'data/RF_AttributeTable_PeakMag_FullDataset_trimmed.csv'  # source table to use for model training and evaluation
METRIC = "PeakArea"  #modeling target (log-transformed)
ID_COL = "GAGE_ID"  # column name for the watershed identifier
N_SEEDS = 100  # number of random train/test splits to generate
WITHHOLDING = "80_20"  # the withholding percentage, this is the optimized percentage if data remains the same as the repo, otherwise change to the withholding percentage used in step 01
RF_KWARGS = dict(n_estimators=100, random_state=42) 
#---------------------------------------------MAIN CODE BLOCK ---------------------------------------------------------------------------

def main(): 
    #load the full trimmed feature table and restrict to the model domain of applicability
    df = table_loader(SOURCE_TABLE, read_selection(FEATURES_FILE), id_col = ID_COL, metric = METRIC)
    df = filter_bounds(df)
    n_start = df.shape[1] - 2 # remove the ID column and metric columns (not features)
    for x in range(N_SEEDS): 
        seed_dir = OUTPUTS / "seeds" / WITHHOLDING / f"Seed_{x}"
        train_ids = pd.read_csv(seed_dir / "wats_train.csv")[ID_COL].tolist()
        test_ids = pd.read_csv(seed_dir / "wats_test.csv")[ID_COL].tolist()

        out_dir = OUTPUTS / "variable_optimization" / f"Seed_{x}"
        out_dir.mkdir(parents=True, exist_ok=True)

        data = df.copy()  # start with the full feature set for this seed

        #recursively remove the least important feature and evaluate model performance
        features = [c for c in df.columns if c not in (METRIC, ID_COL)]
        while len(features) >= 1:
            n = len(features)
            train = df[df[ID_COL].isin(train_ids)]
            test  = df[df[ID_COL].isin(test_ids)]

            X_train, y_train = train[features], np.log(train[METRIC])
            X_test,  y_test  = test[features],  np.log(test[METRIC])

            model  = fit_rf(X_train, y_train, **RF_KWARGS)
            stats  = evaluate(y_test, model.predict(X_test))
            ranked = ranked_importances(model, features)

            pd.DataFrame([stats]).to_csv(out_dir / f"Stats_Vars{n}.csv", index=False)
            ranked.to_csv(out_dir / f"Importances_Vars{n}.csv", index=False)

            features = [f for f in features if f != ranked["Feature"].iloc[-1]]

        print(f"seed {x}: reduced {n_start} features -> 1")


       