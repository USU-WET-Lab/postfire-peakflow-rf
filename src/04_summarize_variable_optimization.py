"""
04_summarize_variable_optimization.py - Step 4 of the post-fire peak flow pipeline

Aggregates the recursive feature elimination output from step 03 into the two things needed to
pick a final feature set:

  1. An R2-versus-feature-count curve, one line per seed plus the across-seed mean. Performance
     climbs steeply over the first few features, plateaus, then drifts as noise features
     are added back. The left edge of that plateau is what to look for: the fewest features that keep
     the prediction skill without adding noise. The mean R2 is saved to a csv and plotted to a png.

  2. An average feature ranking at each feature count.

FEATURE SELECTION IS MANUAL. This script produces the information needed to pick a final feature set.
You will need to look at the R2 curve and the average rankings, pick a feature count, then pick the features 
to keep at that count. This may be the top-ranked features, but you may also keep features that rank lower but are 
critical to the research question (e.g., burn variables). The final feature set is saved by hand as 
data/RF_AttributeTable_PeakMag_OptimizedModel.csv, which is what step 05 reads.

Reads:  outputs/variable_optimization/Seed_<x>/Stats_Vars<n>.csv
        outputs/variable_optimization/Seed_<x>/Importances_Vars<n>.csv

Writes: outputs/variable_optimization_summary/r2_by_feature_count.csv   (rows = feature count, cols = seeds)
        outputs/variable_optimization_summary/r2_mean.csv               (feature count, mean R2)
        outputs/variable_optimization_summary/r2_curve.png
        outputs/variable_optimization_summary/rankings/average_rank_vars<n>.csv

Run: python src/04_summarize_variable_optimization.py

"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
OUTPUTS = ROOT / "outputs"

RUN_DIR = OUTPUTS / "variable_optimization"   # where step 03 wrote its per-seed folders
FIGSIZE = (7, 5)

#---------------------------------------------LOADING-------------------------------------------------------------

def seed_dirs():
    """Return a sorted list of the Seed_<x> directories under RUN_DIR."""
    dirs = [d for d in RUN_DIR.glob("Seed_*") if d.is_dir()]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def feature_counts(seed_dir):
    counts = [int(f.stem.replace("Stats_Vars", "")) for f in seed_dir.glob("Stats_Vars*.csv")]
    return sorted(counts)


def r2_by_feature_count(dirs):
    """Assemble R2 for every (feature count, seed) pair into one frame indexed by feature count."""
    columns = {}
    for seed_dir in dirs:
        r2 = {}
        for n in feature_counts(seed_dir):
            stats = pd.read_csv(seed_dir / f"Stats_Vars{n}.csv")
            r2[n] = stats["R2"].iloc[0]
        columns[seed_dir.name] = pd.Series(r2)

    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "n_features"
    return frame


def average_ranks(dirs, n):
    ranks_by_seed = {}
    for seed_dir in dirs:
        importances_file = seed_dir / f"Importances_Vars{n}.csv"
        if not importances_file.exists():
            continue
        ordered = pd.read_csv(importances_file)["Feature"].tolist()
        ranks_by_seed[seed_dir.name] = {feature: i + 1 for i, feature in enumerate(ordered)}

    if not ranks_by_seed:
        return None

    every_feature = sorted({f for ranks in ranks_by_seed.values() for f in ranks})
    penalty = len(every_feature) + 1

    ranks = pd.DataFrame(
        {seed: [r.get(feature, penalty) for feature in every_feature]
         for seed, r in ranks_by_seed.items()},
        index=every_feature)
    ranks.index.name = "feature"

    return (ranks.mean(axis=1)
                 .sort_values()
                 .rename("mean_rank")
                 .reset_index())


#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main():
    out_dir = OUTPUTS / "variable_optimization_summary"
    rankings_dir = out_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    dirs = seed_dirs()
    if not dirs:
        raise FileNotFoundError(f"No Seed_<x> folders under {RUN_DIR}. Run step 03 first.")
    print(f"Found {len(dirs)} seeds under {RUN_DIR}")

    # 1. R2 vs feature count, per seed and averaged.
    r2 = r2_by_feature_count(dirs)
    r2_mean = r2.mean(axis=1).rename("mean_R2")

    r2.to_csv(out_dir / "r2_by_feature_count.csv")
    r2_mean.to_csv(out_dir / "r2_mean.csv")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for seed in r2.columns:
        ax.plot(r2.index, r2[seed], color="grey", alpha=0.3, linewidth=0.8)
    ax.plot(r2_mean.index, r2_mean, color="black", linewidth=2, label="Mean across seeds")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("R2 (log space)")
    ax.set_title("Recursive feature elimination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "r2_curve.png", dpi=300)
    plt.close(fig)

    best = r2_mean.idxmax()
    print(f"Mean R2 peaks at {best} features (R2 = {r2_mean.loc[best]:.3f}); ")

    # 2. Average feature ranking at each feature count.
    written = 0
    for n in r2.index:
        ranking = average_ranks(dirs, n)
        if ranking is None:
            continue
        ranking.to_csv(rankings_dir / f"average_rank_vars{n}.csv", index=False)
        written += 1

    print(f"Wrote the R2 curve and {written} ranking tables to {out_dir}")


if __name__ == "__main__":
    main()
