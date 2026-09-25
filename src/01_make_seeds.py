"""
01_make_seeds.py - Step 1 of the post-fire peak flow pipeline 

Builds n random train/test "seeds". For each seed, a fraction of watersheds is held 
out for testing while the rest are what the RF is trained on. Splitting is done at the 
WATERSHED level (every storm event at a given gage/watershed is stays together on the same side 
of the split), effectively preventing both temporal and spatial data leakage between the train and 
test set. Each seed is a unique random split, and subsequent steps train one model per seed and aggregate. 

Run this once for each withholding percentage you want to compare in step 02 (change WITHHOLD_FRACTION and re-run;
the folder label is derived from it automatically)

Note: splits are drawn randomly. Leave RANDOM_SEED as None for fresh random splits each re-run, or set it to any integer to make
splits re-producible run-to-run.

The watershed list comes from the trimmed source table restricted to the model's domain of
applicability -- the same table and the same bounds steps 02 and 03 use. It is deliberately NOT
taken from the optimized table: that one is built by step 04b, which runs after this step, and a
seed set should depend on which watersheds are in the study, not on which features were picked.

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
RANDOM_SEED = None  # set to None for fresh random splits each re-run, or set to any integer (e.g., 42) to make splits reproducible run-to-run

# Model domain of applicability, identical to steps 02, 03 and 04b. Applied here so a seed can
# only ever name watersheds the models are actually trained and scored on.
BOUNDS = dict(min_drain_sqkm = 50, min_burned_area_pct = 20, min_burned_storm_depth_pct = 70,
              max_days_since_fire = 1095)

#---------------------------------------main code block ---------------------------------------------------------------------------
def main(): 
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    #load the gage ids from the source table
    table = pd.read_csv(DATA / SOURCE_TABLE)
    n_all = table["GAGE_ID"].nunique()
    table = table[(table.DRAIN_SQKM >= BOUNDS['min_drain_sqkm']) &
                  (table.MTBS_burnedarea_per >= BOUNDS['min_burned_area_pct']) &
                  (table.burned_storm_depth_per >= BOUNDS['min_burned_storm_depth_pct']) &
                  (table.DaysSinceFire <= BOUNDS['max_days_since_fire'])]
    watersheds = table["GAGE_ID"].unique().tolist()
    print(f"{len(watersheds)} watersheds within the model bounds (of {n_all} in {SOURCE_TABLE})")

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

