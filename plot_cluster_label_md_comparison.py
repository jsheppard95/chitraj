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

# MD distance files from gmx distance
MD_DIST_1BAR = BASE_DIR / "res406_res537_1bar_dist.xvg"
MD_DIST_3KBAR = BASE_DIR / "res406_res537_3kbar_dist.xvg"

COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"
MD_LINESTYLE = "--"
MD_LINEWIDTH = 2.0


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

    # ChiLife distributions appear to be normalized as densities in Å^-1.
    # Convert to nm^-1 for comparison to MD distributions plotted versus nm.
    P_1bar = 10.0 * P_1bar
    P_3kbar = 10.0 * P_3kbar

    exp_1bar = get_condition_weighted_mean_distance(summary_df, "1bar")
    exp_3kbar = get_condition_weighted_mean_distance(summary_df, "3kbar")

    cluster_cols_1bar = [c for c in dist_1bar_df.columns if c.startswith("cluster_")]
    cluster_cols_3kbar = [c for c in dist_3kbar_df.columns if c.startswith("cluster_")]

    return {
        "title": model_cfg["title"],
        "r_nm": r_nm_1,
        "P_1bar": P_1bar,
        "P_3kbar": P_3kbar,
        "clusters_1bar": 10 * dist_1bar_df[cluster_cols_1bar].to_numpy(),
        "clusters_3kbar": 10 * dist_3kbar_df[cluster_cols_3kbar].to_numpy(),
        "exp_1bar": exp_1bar,
        "exp_3kbar": exp_3kbar,
    }


def load_gmx_distance_xvg(xvg_path: Path) -> np.ndarray:
    """
    Read a gmx distance XVG file and return the distance column (nm).
    Assumes standard 2-column output:
        time(ps)   distance(nm)
    Ignores comment/header lines beginning with '#' or '@'.
    """
    distances = []
    with open(xvg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                dist_nm = float(parts[1])
            except ValueError:
                continue

            distances.append(dist_nm)

    distances = np.asarray(distances, dtype=float)

    if distances.size == 0:
        raise ValueError(f"No numeric distance data read from {xvg_path}")

    return distances


def make_bin_edges_from_centers(r_nm: np.ndarray) -> np.ndarray:
    """
    Convert a center grid r_nm into histogram bin edges.
    Works for uniform or mildly nonuniform spacing.
    """
    r_nm = np.asarray(r_nm, dtype=float)
    if r_nm.ndim != 1 or r_nm.size < 2:
        raise ValueError("r_nm must be a 1D array with at least 2 points")

    mids = 0.5 * (r_nm[:-1] + r_nm[1:])
    left = r_nm[0] - 0.5 * (r_nm[1] - r_nm[0])
    right = r_nm[-1] + 0.5 * (r_nm[-1] - r_nm[-2])
    edges = np.concatenate(([left], mids, [right]))
    return edges


def histogram_on_grid(values_nm: np.ndarray, r_nm: np.ndarray) -> np.ndarray:
    """
    Histogram the raw MD distances onto the same r_nm grid as the ChiLife curves.
    Returns a probability density P(r) with units 1/nm.
    """
    edges = make_bin_edges_from_centers(r_nm)
    hist, _ = np.histogram(values_nm, bins=edges, density=True)
    return hist


def main() -> None:
    model_data = [load_model_data(cfg) for cfg in MODEL_FILES]

    md_dist_1bar = load_gmx_distance_xvg(MD_DIST_1BAR)
    md_dist_3kbar = load_gmx_distance_xvg(MD_DIST_3KBAR)

    # Use the first model's r grid as the MD comparison grid.
    # We also verify all models share the same grid.
    r_ref = model_data[0]["r_nm"]
    for d in model_data[1:]:
        if not np.allclose(d["r_nm"], r_ref):
            raise ValueError("Not all model r_nm grids match; cannot overlay a single MD histogram cleanly.")

    md_P_1bar = histogram_on_grid(md_dist_1bar, r_ref)
    md_P_3kbar = histogram_on_grid(md_dist_3kbar, r_ref)

    md_mean_1bar = float(np.mean(md_dist_1bar))
    md_mean_3kbar = float(np.mean(md_dist_3kbar))

    # Check Normalization
    print("ChiLife sum(P_1bar)       =", np.sum(d["P_1bar"]))
    print("ChiLife trapz(P_1bar, r)  =", np.trapz(d["P_1bar"], d["r_nm"]))
    print("ChiLife trapz(P_1bar*10, r_nm)     =", np.trapz(10.0 * d["P_1bar"], d["r_nm"]))

    print("ChiLife sum(P_3kbar)      =", np.sum(d["P_3kbar"]))
    print("ChiLife trapz(P_3kbar, r) =", np.trapz(d["P_3kbar"], d["r_nm"]))
    print("ChiLife trapz(P_3kbar*10, r_nm)     =", np.trapz(10.0 * d["P_3kbar"], d["r_nm"]))

    print("MD sum(hist)              =", np.sum(md_P_1bar))
    print("MD trapz(hist, r)         =", np.trapz(md_P_1bar, r_ref))

    ymax = 0.0
    for d in model_data:
        ymax = max(
            ymax,
            np.nanmax(d["P_1bar"]),
            np.nanmax(d["P_3kbar"]),
            np.nanmax(d["clusters_1bar"]),
            np.nanmax(d["clusters_3kbar"]),
        )

    ymax = max(ymax, np.nanmax(md_P_1bar), np.nanmax(md_P_3kbar))
    ylimit = 1.5 * ymax

    fig, axes = plt.subplots(
        3, 1, figsize=(5.2, 6.8), sharex=True, sharey=True
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

        # ChiLife ensemble-average curves
        ax.plot(
            r_nm,
            d["P_1bar"],
            color=COLOR_1BAR,
            linewidth=2.5,
            zorder=3,
            label=rf"ChiLife 1 bar ($\langle r \rangle={d['exp_1bar']:.2f}$ nm)",
        )
        ax.plot(
            r_nm,
            d["P_3kbar"],
            color=COLOR_3KBAR,
            linewidth=2.5,
            zorder=3,
            label=rf"ChiLife 3 kbar ($\langle r \rangle={d['exp_3kbar']:.2f}$ nm)",
        )

        # MD distributions from raw residue-residue distances
        ax.plot(
            r_nm,
            md_P_1bar,
            color=COLOR_1BAR,
            linestyle=MD_LINESTYLE,
            linewidth=MD_LINEWIDTH,
            zorder=4,
            label=rf"MD 1 bar ($\langle r \rangle={md_mean_1bar:.2f}$ nm)",
        )
        ax.plot(
            r_nm,
            md_P_3kbar,
            color=COLOR_3KBAR,
            linestyle=MD_LINESTYLE,
            linewidth=MD_LINEWIDTH,
            zorder=4,
            label=rf"MD 3 kbar ($\langle r \rangle={md_mean_3kbar:.2f}$ nm)",
        )

        ax.set_ylabel(r"$P(r)$", fontsize=12)
        ax.set_title(d["title"], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
        ax.set_xlim(1.5, 6.0)
        ax.set_ylim(0.0, ylimit)

    axes[-1].set_xlabel(r"Distance (nm)", fontsize=12)

    fig.tight_layout()

    out_png = BASE_DIR / "cluster_output/cluster_label_vs_md_comparison_3panel.png"
    out_pdf = BASE_DIR / "cluster_output/cluster_label_vs_md_comparison_3panel.pdf"
    fig.savefig(out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {out_png}")
    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
    plt.show()