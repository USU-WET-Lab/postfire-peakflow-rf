"""
11_summarize_by_watershed.py - Step 11 of the post-fire peak flow pipeline

Summarizes the by-watershed run from step 06 into per-watershed feature influence. For each
watershed, every seed that withheld it produced its own SHAP table; this averages the mean
absolute SHAP value per feature over those seeds, giving one influence profile per watershed.

Where step 09 asks "which drivers matter across the whole dataset?", this asks "which drivers
matter *at this gage?*" - the profiles are what let you say that one watershed is storm-driven
while another is burn-severity-driven.

Three tables come out: the raw influence values, the same values ranked within each watershed,
and a compact top-N summary with one row per watershed.

Reads:  data/<SOURCE_TABLE> (for the watershed list)
        outputs/model_run_watersheds/Seed_<x>/Watershed_USGS<id>/shap_values.csv

Writes: outputs/watershed_summary/watershed_importances.csv       (features x watersheds, mean |SHAP|)
        outputs/watershed_summary/watershed_importance_ranks.csv  (features x watersheds, rank)
        outputs/watershed_summary/watershed_top_features.csv      (GAGE_ID, rank_1 ... rank_N)

Run: python src/11_summarize_by_watershed.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

SOURCE_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"   # only read for its GAGE_ID column
ID_COL  = "GAGE_ID"
N_SEEDS = 100
TOP_N   = 5     # how many top-ranked features to carry into the compact summary table

#---------------------------------------------LOADING-------------------------------------------------------------

def watershed_importance(watershed):
    """Mean absolute SHAP value per feature for one watershed.

    Averaged over every seed that withheld this watershed. Returns a Series indexed by
    feature and named for the watershed, or None if no seed ever withheld it.
    """
    per_seed = []

    for x in range(N_SEEDS):
        shap_file = (OUTPUTS / "model_run_watersheds" / f"Seed_{x}" /
                     f"Watershed_USGS{watershed}" / "shap_values.csv")
        if not shap_file.exists():
            continue   # this watershed was in the training set for this seed
        per_seed.append(pd.read_csv(shap_file).abs().mean().rename(f"seed_{x}"))

    if not per_seed:
        return None

    # concat aligns the seeds on the feature-name index, so column order never matters.
    return pd.concat(per_seed, axis=1).mean(axis=1).rename(watershed)


#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    out_dir = OUTPUTS / "watershed_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    watersheds = pd.read_csv(DATA / SOURCE_TABLE)[ID_COL].unique().tolist()
    print(f"Summarizing feature influence for {len(watersheds)} watersheds")

    # 1. One influence profile per watershed.
    profiles = []
    for watershed in watersheds:
        importance = watershed_importance(watershed)
        if importance is None:
            print(f"USGS{watershed}: no step 06 output found, skipping")
            continue
        profiles.append(importance)
        print(f"USGS{watershed}: top feature {importance.idxmax()}")

    if not profiles:
        raise FileNotFoundError(
            f"No per-watershed SHAP values under {OUTPUTS / 'model_run_watersheds'}. Run step 06 first.")

    # 2. Features down the rows, watersheds across the columns.
    importances = pd.concat(profiles, axis=1)
    importances.index.name = "feature"
    importances.to_csv(out_dir / "watershed_importances.csv")

    # Ranked within each watershed. method="min" keeps ranks whole numbers when features tie.
    ranks = importances.rank(ascending=False, method="min")
    ranks.to_csv(out_dir / "watershed_importance_ranks.csv")

    # 3. Compact view: the TOP_N most influential features at each watershed, one row each.
    top_features = pd.DataFrame(
        [[watershed, *importances[watershed].nlargest(TOP_N).index]
         for watershed in importances.columns],
        columns=[ID_COL, *(f"rank_{n}" for n in range(1, TOP_N + 1))])
    top_features.to_csv(out_dir / "watershed_top_features.csv", index=False)

    print(f"Done! Summarized {len(profiles)} watersheds, wrote 3 tables to {out_dir}")


if __name__ == "__main__":
    main()
