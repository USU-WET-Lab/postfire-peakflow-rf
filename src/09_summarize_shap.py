"""
09_summarize_shap.py - Step 9 of the post-fire peak flow pipeline

Pools the SHAP values from every seed of the step 05 run and ranks features
by mean. SHAP values quantify the contribution od each feature to the prediction
regardless of direction. For feature importance and direction of influence,
see step 10.

Bars are colored by feature category (storm, fire, watershed/climate)

Reads:  outputs/model_run/Seed_<x>/shap_values.csv

Writes: outputs/shap_summary/shap_importance.csv     (feature, mean_abs_shap, category)
        outputs/shap_summary/shap_importance_bar.png

Run: python src/09_summarize_shap.py
"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rf_utils import load_pooled_shap

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
OUTPUTS = ROOT / "outputs"

N_SEEDS = 100

# Bar color by driver category. The original hard-coded one color per bar in importance
# order, which silently mis-colors the figure whenever the ranking shifts; mapping
# feature -> category keeps the colors right no matter how the bars sort.
CATEGORY_COLORS = {"storm": "blue", "fire": "mediumseagreen", "watershed": "brown"}
UNCATEGORIZED_COLOR = "lightgrey"

FEATURE_CATEGORIES = {
    # storm and antecedent conditions
    "AntecedentPrecip_mm":              "storm",
    "AvgMonthly_Precip_cm":             "storm",
    "D1":                               "storm",
    "CoefVar":                          "storm",
    # fire, burn severity, vegetation change, and burned-area/storm overlap
    "DaysSinceFire":                    "fire",
    "MTBS_burnedarea_per":              "fire",
    "SBS_mod_high_area_burned_per_wat": "fire",
    "burned_storm_depth_per":           "fire",
    "burned_storm_int_per":             "fire",
    "ratio_LAI":                        "fire",
    # watershed physiography and long-term climate
    "DRAIN_SQKM":                       "watershed",
    "riparian_area_per":                "watershed",
    "PPTAVG_BASIN":                     "watershed",
    "RH_BASIN":                         "watershed",
    "SNOW_PCT_PRECIP":                  "watershed",
    "PRECIP_SEAS_IND":                  "watershed",
}

#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    out_dir = OUTPUTS / "shap_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mean absolute SHAP value per feature, most influential first.
    shap_values = load_pooled_shap(OUTPUTS / "model_run", N_SEEDS)
    importance = (shap_values.abs().mean()
                  .sort_values(ascending=False)
                  .rename("mean_abs_shap")
                  .rename_axis("feature")
                  .reset_index())
    importance["category"] = importance.feature.map(FEATURE_CATEGORIES).fillna("uncategorized")
    importance.to_csv(out_dir / "shap_importance.csv", index=False)

    uncategorized = importance.loc[importance.category == "uncategorized", "feature"].tolist()
    if uncategorized:
        print(f"No category set for {uncategorized}, drawn in {UNCATEGORIZED_COLOR}")

    # 2. Bar chart, colored by category.
    colors = [CATEGORY_COLORS.get(c, UNCATEGORIZED_COLOR) for c in importance.category]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(importance.feature, importance.mean_abs_shap, color=colors)
    ax.set_ylabel("Mean |SHAP value|")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(handles=[Patch(facecolor=color, label=category)
                       for category, color in CATEGORY_COLORS.items()])

    fig.tight_layout()
    out_path = out_dir / "shap_importance_bar.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Done! Ranked {len(importance)} features, wrote {out_path}")


if __name__ == "__main__":
    main()
