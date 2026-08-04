"""
08_summarize_performance.py - Step 8 of the post-fire peak flow pipeline

Summarizes model performance across all seeds. The first step is to plot
test set predictions vs actual values for all seeds. Then, for 
waterseds of interest, the test set predictions are plotted as boxplots on 
top of the overall scatter plot. The mean R2, RMSE, and MSE for each
watershed is also calculated and saved to a CSV file.

Predictions are stored in log space, everything is exponentiated before plotting, and 
the axes are set to log scale. 

Reads:  outputs/model_run/Seed_<x>/stats.csv, predictions.csv
        outputs/model_run_watersheds/Seed_<x>/Watershed_USGS<id>/stats.csv, predictions.csv

Writes: outputs/performance_summary/seed_stats_summary.csv        (per-seed mse, rmse, R2)
        outputs/performance_summary/watershed_stats_summary.csv   (per-watershed mean/std)
        outputs/performance_summary/predicted_vs_actual.png

Run: python src/08_summarize_performance.py

"""
#---------------------------------------------------IMPORTS----------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#---------------------------------------------------CONFIG: only edit this block--------------------------------------------------------------

ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
OUTPUTS = ROOT / "outputs"

N_SEEDS = 100
ID_COL  = "GAGE_ID"
HIGHLIGHT_WATERSHEDS = [11458500, 11065000, 9361000]
HIGHLIGHT_COLORS     = ["royalblue", "orange", "mediumvioletred"]
ONE_TO_ONE = (0.001, 2)   # extent of the 1:1 reference line, m3/s/km2
BOX_WIDTH_FRACTION = 8    # box width = position / this, keeps boxes even on a log axis

#---------------------------------------------------LOADING FUNCTIONS----------------------------------------------------------------
def load_seed_predictions(): 
    """Pool every seed's predictions from step 05"""
    predictions_frames = []
    stat_frames = []
    for x in range(N_SEEDS):
        seed_dir = OUTPUTS / "model_run" / f"Seed_{x}"
        if not seed_dir.exists():
            continue
        predictions = pd.read_csv(seed_dir / "predictions.csv") 
        predictions["seed"] = x
        predictions_frames.append(predictions)
        stats = pd.read_csv(seed_dir / "stats.csv")
        stats["seed"] = x
        stat_frames.append(stats)
    if not predictions_frames:
        raise FileNotFoundError("No seed predictions found, run step 05 first.")
    return (pd.concat(predictions_frames, ignore_index = True), 
            pd.concat(stat_frames, ignore_index = True))

def load_watershed_run(watershed): 
    """Collect step 6 leave-one-watershed-out predictions for a given watershed 
    """
    predictions_by_seed = pd.DataFrame()
    stats_frames = []
    observed = None
    for x in range(N_SEEDS):
        watershed_dir = OUTPUTS / "model_run_watersheds" / f"Seed_{x}" / f"Watershed_USGS{watershed}"
        if not watershed_dir.exists():
            continue
        predictions = pd.read_csv(watershed_dir / "predictions.csv", reset_index(drop = True) )
        predictions_by_seed[f"seed{x}"] = predictions["y_pred"]
        if observed is None: 
            observed = predictions["y_test"]
        stat_frames.append(pd.read_csv(watershed_dir / "stats.csv"))
        if observed is None:
            return None, None

#---------------------------------------------------PLOTTING--------------------------------------------------------------

def plot_density_cloud(ax, predictions):
    """Scatter pooled predicted vs. actual in native units, colored by point density."""
    density = gaussian_kde(np.vstack([predictions.y_test, predictions.y_pred]))
    shading = density(np.vstack([predictions.y_test, predictions.y_pred]))
    scatter = ax.scatter(np.exp(predictions.y_test), np.exp(predictions.y_pred),
                         c=shading, cmap="viridis", alpha=0.05)
    ax.plot(ONE_TO_ONE, ONE_TO_ONE, "--", color="black", lw=2)   # 1:1 line
    ax.set_xlabel("Actual peak (m3/s/km2)")
    ax.set_ylabel("Predicted peak (m3/s/km2)")
    return scatter

def plot_watershed_boxes(ax, predictions_by_seed, color):
    """Overlay one box plot per storm event: spread of seed predictions at the observed peak."""
    seed_columns = [c for c in predictions_by_seed.columns if c != "y_test"]
     for _, event in predictions_by_seed.iterrows():
        position = float(np.exp(event["y_test"]))
        ax.boxplot(np.exp(event[seed_columns].to_numpy(dtype=float)),
                   positions=[position],
                   widths=[position / BOX_WIDTH_FRACTION],
                   whis=[5, 95], showfliers=False, patch_artist=True,
                   medianprops=dict(color="black"),
                   boxprops=dict(facecolor=color, alpha=0.7))


#---------------------------------------------------MAIN--------------------------------------------------------------

def main():
    def main():
    out_dir = OUTPUTS / "performance_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
 # 1. Pool the step 05 run and save the per-seed stats.
    predictions, seed_stats = load_seed_predictions()
    seed_stats.to_csv(out_dir / "seed_stats_summary.csv", index=False)
    print(f"Pooled {len(predictions)} predictions from {len(seed_stats)} seeds "
          f"(mean R2 = {seed_stats.R2.mean():.3f})")

    # 2. Predicted vs. actual density cloud.
    print("estimating point density (this is the slow part)")
    fig, ax = plt.subplots()
    scatter = plot_density_cloud(ax, predictions)
    fig.colorbar(scatter, ax=ax, label="Density")

    # 3. Overlay the highlighted watersheds from the step 06 run.
    watershed_rows = []
    for watershed, color in zip(HIGHLIGHT_WATERSHEDS, HIGHLIGHT_COLORS):
        predictions_by_seed, watershed_stats = load_watershed_run(watershed)
        if predictions_by_seed is None:
            print(f"USGS{watershed}: no step 06 output found, skipping")
            continue

        print(f"USGS{watershed}: {len(predictions_by_seed)} storm events, "
              f"mean R2 = {watershed_stats.R2.mean():.3f}")
        plot_watershed_boxes(ax, predictions_by_seed, color)
        watershed_rows.append({ID_COL: watershed,
                               "events": len(predictions_by_seed),
                               "R2_mean": watershed_stats.R2.mean(),
                               "R2_std": watershed_stats.R2.std(),
                               "rmse_mean": watershed_stats.rmse.mean(),
                               "mse_mean": watershed_stats.mse.mean()})

    if watershed_rows:
        pd.DataFrame(watershed_rows).to_csv(out_dir / "watershed_stats_summary.csv", index=False)

    # 4. Save. Box plots reset the tick locators, so set the log scale last.
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    out_path = out_dir / "predicted_vs_actual.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Done! Wrote summary tables and {out_path}")


if __name__ == "__main__":
    main()



