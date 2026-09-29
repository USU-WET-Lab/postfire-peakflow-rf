"""
01_make_seeds.py - Step 1 of the post-fire peak flow pipeline 

Builds n random train/test "seeds". For each seed, a fraction of gages is held 
out for testing while the rest are what the RF is trained on. Splitting is done at the 
WATERSHED level (every storm event at a given gage/watershed stays together on the same side 
of the split). Each seed is a unique random split, and subsequent steps train one model per seed and aggregate. 

Run this once for each withholding percentage you want to compare in step 02 (change WITHHOLD_FRACTION and re-run)

Note: splits are drawn randomly. Leave RANDOM_SEED as None for fresh random splits, or set it to any integer to make
splits re-producible run-to-run.

Reads: data/<SOURCE_TABLE> (for the GAGE_ID column)

Writes: outputs/seeds/<WITHHOLDING>/Seed_<x>/wats_train.csv
        outputs/seeds/<WITHHOLDING>/Seed_<x>/wats_test.csv

To Run: python src/01_make_seeds.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import pandas as pd
import random
from pathlib import Path
from rf_utils import read_selection, table_loader, filter_bounds 

#--------------------------------------- config: only edit this block ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FEATURES_FILE = "config/feature_list.txt"  # the feature list used in steps 02 and 03, used here to check that the source table has all the features needed for modeling
SOURCE_TABLE = "data/RF_AttributeTable_PeakMag_FullDataset.csv" # the source table to draw watersheds from, must be in data/
N_SEEDS = 100  # number of random train/test splits to generate
WITHHOLD_FRACTION = 0.2  # fraction of watersheds to withhold for testing (e.g., 0.1 = 10% of watersheds are held out for testing)
WITHHOLDING = f"{round((1 - WITHHOLD_FRACTION) * 100)}_{round(WITHHOLD_FRACTION * 100)}" # DO NOT EDIT THIS LINE, it is derived from the WITHHOLD_FRACTION above
RANDOM_SEED = None  

#---------------------------------------main code block ---------------------------------------------------------------------------
def main(): 
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    #load the gage ids from the source table
    table = table_loader(SOURCE_TABLE, read_selection(FEATURES_FILE), id_col = "GAGE_ID", metric = "PeakArea")
    table = filter_bounds(table)
    watersheds = table["GAGE_ID"].unique().tolist()
    print(f"{len(watersheds)} watersheds within the model bounds (of {table['GAGE_ID'].nunique()} in {SOURCE_TABLE})")

    n_test = int(len(watersheds) * WITHHOLD_FRACTION) # number of watersheds to withhold for testing
    print(f"Generating {N_SEEDS} random train/test splits with {n_test} watersheds held out for testing ({WITHHOLDING})")
    seeds_dir = OUTPUTS / "seeds" / WITHHOLDING

    for x in range(N_SEEDS):
        # watershed level split
        test = random.sample(watersheds, n_test)
        train = [w for w in watersheds if w not in test]
        seed_dir = seeds_dir / f"Seed_{x}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(test, columns=["GAGE_ID"]).to_csv(seed_dir / "wats_test.csv", index=False)
        pd.DataFrame(train, columns=["GAGE_ID"]).to_csv(seed_dir / "wats_train.csv", index=False)
    print(f"Done! Wrote {N_SEEDS} random train/test splits to {seeds_dir}")

if __name__ == "__main__":
    main()

