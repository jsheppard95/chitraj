import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    import scienceplots

    plt.style.use(["science"])
    mpl.rcParams["text.usetex"] = True
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 1,
            "lines.linewidth": 2,
            "xtick.major.size": 5,
            "xtick.major.width": 1,
            "xtick.minor.size": 2,
            "xtick.minor.width": 1,
            "ytick.major.size": 5,
            "ytick.major.width": 1,
            "ytick.minor.size": 2,
            "ytick.minor.width": 1,
        }
    )


BASE_DIR = Path(__file__).parent

MODEL_FILES = [
    {
        "name": "R1M_sample_none_clusters",
        "title": "R1 Accessible Volume",
        "dist_csv_1bar": BASE_DIR / "cluster_output/R1_sample_none_1bar_cluster_distributions.csv",
        "dist_csv_3kbar": BASE_DIR / "cluster_output/R1_sample_none_3kbar_cluster_distributions.csv",
        "summary_csv": BASE_DIR / "cluster_output/R1_sample_none_per_cluster_summary.csv",
    },
    {
        "name": "R1M_sample_5000_clusters",
        "title": "R1 Off-Rotamer Sampling",
        "dist_csv_1bar": BASE_DIR / "cluster_output/R1_sample_5000_1bar_cluster_distributions.csv",
        "dist_csv_3kbar": BASE_DIR / "cluster_output/R1_sample_5000_3kbar_cluster_distributions.csv",
        "summary_csv": BASE_DIR / "cluster_output/R1_sample_5000_per_cluster_summary.csv",
    },
    {
        "name": "GTN_sample_5000_clusters",
        "title": "GTN (Gd) Off-Rotamer Sampling",
        "dist_csv_1bar": BASE_DIR / "cluster_output/GTN_sample_5000_1bar_cluster_distributions.csv",
        "dist_csv_3kbar": BASE_DIR / "cluster_output/GTN_sample_5000_3kbar_cluster_distributions.csv",
        "summary_csv": BASE_DIR / "cluster_output/GTN_sample_5000_per_cluster_summary.csv",
    },
]


COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"


def weighted_mean_from_cluster_distributions(
    dist_df: pd.DataFrame,
    summary_sub: pd.DataFrame,
) -> np.ndarray:
    """
    dist_df columns:
      r_nm, cluster_1, cluster_2, ...

    summary_sub columns:
      condition, cluster_id, cluster_size, weight, mean_distance_nm
    """
    cluster_cols = [c for c in dist_df.columns if c.startswith("cluster_")]
    if not cluster_cols:
        raise ValueError("No cluster_* columns found in distribution file")

    # Map cluster ID -> weight from summary
    weight_map = {
        int(row["cluster_id"]): float(row["weight"])
        for _, row in summary_sub.iterrows()
    }

    weights = []
    for col in cluster_cols:
        cluster_id = int(col.split("_")[1])
        if cluster_id not in weight_map:
            raise ValueError(f"Missing weight for {col}")
        weights.append(weight_map[cluster_id])

    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    P = dist_df[cluster_cols].to_numpy()  # shape (n_r, n_clusters)
    weighted_mean = P @ weights
    return weighted_mean


def get_condition_weighted_mean_distance(summary_df: pd.DataFrame, condition: str) -> float:
    sub = summary_df.loc[summary_df["condition"] == condition].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for condition='{condition}'")

    # should already be weighted in spirit, but compute explicitly
    w = sub["weight"].to_numpy(dtype=float)
    x = sub["mean_distance_nm"].to_numpy(dtype=float)
    w = w / w.sum()
    return float(np.sum(w * x))


def load_model_data(model_cfg: dict) -> dict:
    dist_1bar_df = pd.read_csv(model_cfg["dist_csv_1bar"])
    dist_3kbar_df = pd.read_csv(model_cfg["dist_csv_3kbar"])
    summary_df = pd.read_csv(model_cfg["summary_csv"])

    if "r_nm" not in dist_1bar_df.columns or "r_nm" not in dist_3kbar_df.columns:
        raise ValueError(f"Missing 'r_nm' column in cluster distribution files for {model_cfg['name']}")

    r_nm_1 = dist_1bar_df["r_nm"].to_numpy()
    r_nm_3 = dist_3kbar_df["r_nm"].to_numpy()

    if not np.allclose(r_nm_1, r_nm_3):
        raise ValueError(f"1 bar and 3 kbar r_nm grids do not match for {model_cfg['name']}")

    summary_1bar = summary_df.loc[summary_df["condition"] == "1bar"].copy()
    summary_3kbar = summary_df.loc[summary_df["condition"] == "3kbar"].copy()

    P_1bar = weighted_mean_from_cluster_distributions(dist_1bar_df, summary_1bar)
    P_3kbar = weighted_mean_from_cluster_distributions(dist_3kbar_df, summary_3kbar)

    exp_1bar = get_condition_weighted_mean_distance(summary_df, "1bar")
    exp_3kbar = get_condition_weighted_mean_distance(summary_df, "3kbar")

    cluster_cols_1bar = [c for c in dist_1bar_df.columns if c.startswith("cluster_")]
    cluster_cols_3kbar = [c for c in dist_3kbar_df.columns if c.startswith("cluster_")]

    return {
        "title": model_cfg["title"],
        "r_nm": r_nm_1,
        "P_1bar": P_1bar,
        "P_3kbar": P_3kbar,
        "clusters_1bar": dist_1bar_df[cluster_cols_1bar].to_numpy(),
        "clusters_3kbar": dist_3kbar_df[cluster_cols_3kbar].to_numpy(),
        "exp_1bar": exp_1bar,
        "exp_3kbar": exp_3kbar,
    }


def main() -> None:
    model_data = [load_model_data(cfg) for cfg in MODEL_FILES]

    ymax = 0.0
    for d in model_data:
        ymax = max(
            ymax,
            np.nanmax(d["P_1bar"]),
            np.nanmax(d["P_3kbar"]),
            np.nanmax(d["clusters_1bar"]),
            np.nanmax(d["clusters_3kbar"]),
        )

    ylimit = 1.5 * ymax

    fig, axes = plt.subplots(
        3, 1, figsize=(5.2, 6.2), sharex=True, sharey=True
    )

    for ax, d in zip(axes, model_data):
        r_nm = d["r_nm"]

        # faint individual cluster curves in background
        for j in range(d["clusters_1bar"].shape[1]):
            ax.plot(
                r_nm,
                d["clusters_1bar"][:, j],
                color=COLOR_1BAR,
                alpha=0.06,
                linewidth=0.8,
                zorder=1,
            )

        for j in range(d["clusters_3kbar"].shape[1]):
            ax.plot(
                r_nm,
                d["clusters_3kbar"][:, j],
                color=COLOR_3KBAR,
                alpha=0.06,
                linewidth=0.8,
                zorder=1,
            )

        # weighted ensemble-average curves on top
        ax.plot(
            r_nm,
            d["P_1bar"],
            color=COLOR_1BAR,
            linewidth=2.5,
            zorder=3,
            label=rf"1 bar ($\langle r \rangle={d['exp_1bar']:.2f}$ nm)",
        )
        ax.plot(
            r_nm,
            d["P_3kbar"],
            color=COLOR_3KBAR,
            linewidth=2.5,
            zorder=3,
            label=rf"3 kbar ($\langle r \rangle={d['exp_3kbar']:.2f}$ nm)",
        )

        ax.set_ylabel(r"$P(r)$", fontsize=12)
        ax.set_title(d["title"], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
        ax.set_xlim(1.5, 6.0)
        ax.set_ylim(0.0, ylimit)

    axes[-1].set_xlabel(r"Distance (nm)", fontsize=12)

    fig.tight_layout()

    out_png = BASE_DIR / "cluster_output/cluster_label_comparison_3panel.png"
    out_pdf = BASE_DIR / "cluster_output/cluster_label_comparison_3panel.pdf"
    fig.savefig(out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {out_png}")
    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
    plt.show()