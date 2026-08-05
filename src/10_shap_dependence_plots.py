"""
10_shap_dependence_plots.py - Step 10 of the post-fire peak flow pipeline

One SHAP dependence plot per feature, pooled across every seed of the step 05 run. Each plot
puts the feature's value on the x axis and its SHAP value (its influence on that single
prediction) on the y axis, so the shape of the cloud shows both the direction of the effect
and where in the feature's range it turns on.

Raw points are noisy, so a centered rolling mean and two percentile bands are drawn over them
after sorting by feature value. The bands are quantiles of the SHAP values inside the rolling
window, not a confidence interval: they show how much the influence of a given feature value
varies across storms and seeds.

Step 09 collapses these same values to one bar per feature; this is the feature-by-feature view.

Reads:  outputs/model_run/Seed_<x>/shap_values.csv, shap_data.csv

Writes: outputs/shap_summary/dependence/dependence_<feature>.png

Run: python src/10_shap_dependence_plots.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rf_utils import load_pooled_shap

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
OUTPUTS = ROOT / "outputs"

N_SEEDS = 100

ROLLING_WINDOW = 1000        # points per rolling window; narrowed automatically on small runs
INNER_BAND = (0.25, 0.75)    # dark band: interquartile spread of influence
OUTER_BAND = (0.10, 0.90)    # light band
FIGSIZE = (4, 3)

#---------------------------------------------PLOT----------------------------------------------------------------

def dependence_frame(feature_data, feature_shap, window):
    """Sort one feature's points by feature value and add the rolling mean and bands."""
    frame = pd.DataFrame({"data": feature_data, "shap": feature_shap}).sort_values("data")

    rolling = frame["shap"].rolling(window=window, center=True)
    frame["mean"] = rolling.mean()
    for quantile in (*INNER_BAND, *OUTER_BAND):
        frame[f"q{int(quantile * 100)}"] = rolling.quantile(quantile)
    return frame


def plot_dependence(frame, feature, path):
    """Scatter of feature value vs. influence, with the rolling mean and bands on top."""
    inner_lo, inner_hi = (f"q{int(q * 100)}" for q in INNER_BAND)
    outer_lo, outer_hi = (f"q{int(q * 100)}" for q in OUTER_BAND)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(frame.data, frame.shap, s=5, alpha=0.1, c="grey")
    ax.plot(frame.data, frame["mean"], color="black")
    ax.fill_between(frame.data, frame[inner_lo], frame[inner_hi], color="black", alpha=0.5)
    ax.fill_between(frame.data, frame[inner_hi], frame[outer_hi], color="black", alpha=0.2)
    ax.fill_between(frame.data, frame[outer_lo], frame[inner_lo], color="black", alpha=0.2)

    ax.set_xlabel("Feature value")
    ax.set_ylabel("Influence (shap value)")
    ax.set_title(feature)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    out_dir = OUTPUTS / "shap_summary" / "dependence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # `data` holds the feature values that produced `values`, aligned row for row.
    values, data = load_pooled_shap(OUTPUTS / "model_run", N_SEEDS, with_data=True)

    # A window wider than the data returns all NaN, which silently draws a blank figure.
    window = min(ROLLING_WINDOW, max(len(values) // 10, 1))
    if window < ROLLING_WINDOW:
        print(f"Only {len(values)} pooled points, narrowing rolling window to {window}")

    for feature in values.columns:
        frame = dependence_frame(data[feature], values[feature], window)
        plot_dependence(frame, feature, out_dir / f"dependence_{feature}.png")
        print(f"  {feature}")

    print(f"Done! Wrote {len(values.columns)} dependence plots to {out_dir}")


if __name__ == "__main__":
    main()
