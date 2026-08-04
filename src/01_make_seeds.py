"""
01_make_seeds.py - Step 1 of the post-fire peak flow pipeline 

Builds N random train/test "seeds". For each seed, a fraction of watersheds is held 
out for testing while the rest are what the RF is trained on. Splitting is done at the 
WATERSHED level (every storm event at a given gage/watershed is stays together on the same side 
of the split), effectively preventing both temporal and spatial data leakage between the train and 
test set. Each seed is a unique random split, and subsequent steps train one model per seed and aggregate. 

Run this once for each witholding percentage you want to compare in step 02 (change WITHOLD_FRACTION and WITHOLDING and re-run)

Note: splits are drawn randomly. Leave RANDOM_SEED as None for fresh random splits each re-run, or set it to any interger to make 
splits re-producible run-to-run. 

Reads: data/<SOURCE_TABLE> (for the GAGE_ID column)

Writes: outputs/seeds/<WITHOLDING>/Seed_<x>/wats_train.csv
        outputs/seeds/<WITHOLDING>/Seed_<x>/wats_test.csv

To Run: python src/01_make_seeds.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path


#--------------------------------------- config: only edit this block ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SOURCE_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"
N_SEEDS = 100  # number of random train/test splits to generate
WITHOLD_FRACTION = 0.2  # fraction of watersheds to withold for testing (e.g., 0.1 = 10% of watersheds are held out for testing)
WITHOLDING = "80_20"  # string to describe the witholding fraction (e.g., "90_10" for 10% held out, "80_20" for 20% held out)
RANDOM_SEED = None  # set to None for fresh random splits each re-run, or set to any integer to make splits reproducible run-to-run

#---------------------------------------main code block ---------------------------------------------------------------------------
def main(): 
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    #load the gage ids from the source table
    table = pd.read_csv(DATA / SOURCE_TABLE)
    watersheds = table["GAGE_ID"].unique().tolist()

    n_test = int(len(watersheds) * WITHOLD_FRACTION) # number of watersheds to withold for testing
    print(f"Generating {N_SEEDS} random train/test splits with {n_test} watersheds held out for testing ({WITHOLDING})")
    seeds_dir = OUTPUTS / "seeds" / WITHOLDING

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

