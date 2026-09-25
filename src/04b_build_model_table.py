"""
04b_build_model_table.py - Step 4b of the post-fire peak flow pipeline

Applies the final feature selection entered in config/selected_features.txt 
to the full attribute table to produce the final model table used in step 05, 06, 07
Also restricts the tables to the model's domain of appplicability for use in the final model
training. 

Note that this script overwrites data/RF_AttributeTable_PeakMag_OptimizedModel.csv, 
if previous steps have been run as is, the table will remain the same, if interested use git 
to see the diff. 

Reads: config/selected_features.txt
       data/<SOURCE_TABLE> (full attribute table)

Writes: data/<MODEL_TABLE> (final optimized feature set)

Run: python src/04b_build_model_table.py

"""
#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
from rf_utils import read_selection, filter_bounds, BOUNDS

#------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
DATA = ROOT / "data"
SELECTION_FILE = ROOT / "config" / "selected_features.txt"
SOURCE_TABLE = "RF_AttributeTable_PeakMag_FullDataste_trimmed.csv" 
MODEL_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"  # final optimized feature set
METRIC = "PeakArea"
ID_COL = "GAGE_ID"

#Domain of applicability: identical to step 02 and 03, applied here so the final model trains 
# on thhe same data te set was optimized against 
APPLY_BOUNDS = True   # the bounds themselves live in rf_utils.BOUNDS

#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------


def main(): 
    features = read_selection(SELECTION_FILE)
    df = pd.read_csv(DATA / SOURCE_TABLE)
    missing = [f for f in features if f not in df.columns]
    if missing: 
        raise ValueError(f"features that your're trying to remove are not in {SOURCE_TABLE}: {missing}")
    n_all = len(df)
    if APPLY_BOUNDS:
        df = filter_bounds(df, BOUNDS)
    table = df[[ID_COL, METRIC] + features]
    table.to_csv(DATA / MODEL_TABLE, index=False)
    print(f"Wrote final model table to {DATA / MODEL_TABLE} ({len(table)} rows, {len(features)} features) "
            f"from {n_all} rows in {SOURCE_TABLE} after applying bounds: {APPLY_BOUNDS}")


if __name__ == "__main__":
    main()

    