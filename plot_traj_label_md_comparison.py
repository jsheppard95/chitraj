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
        "name": "R1M_sample_none",
        "title": "R1 Accessible Volume",
        "dist_csv": BASE_DIR / "tip4p/stride_1/r1m_sample_none/combined_output/R1_no_sample_trajectory_distributions.csv",
        "summary_csv": BASE_DIR / "tip4p/stride_1/r1m_sample_none/combined_output/R1_no_sample_per_frame_summary.csv",
    },
    {
        "name": "R1M_sample_5000",
        "title": "R1 Off-Rotamer Sampling",
        "dist_csv": BASE_DIR / "tip4p/stride_1/r1m_sample_5000/combined_output/R1_sample_5000_trajectory_distributions.csv",
        "summary_csv": BASE_DIR / "tip4p/stride_1/r1m_sample_5000/combined_output/R1_sample_5000_per_frame_summary.csv",
    },
    {
        "name": "GTN",
        "title": "GTN (Gd) Off-Rotamer Sampling",
        "dist_csv": BASE_DIR / "tip4p/stride_1/gtn_sample_5000/combined_output/GTN_sample_5000_trajectory_distributions.csv",
        "summary_csv": BASE_DIR / "tip4p/stride_1/gtn_sample_5000/combined_output/GTN_sample_5000_per_frame_summary.csv",
    },
]

# Raw MD residue-residue distances from gmx distance
MD_DIST_1BAR = BASE_DIR / "tip4p/res406_res537_1bar_dist.xvg"
MD_DIST_3KBAR = BASE_DIR / "tip4p/res406_res537_3kbar_dist.xvg"

COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"

MD_LINESTYLE = "--"
MD_LINEWIDTH = 2.0

# ChiLife trajectory distributions appear to be stored as densities in Å^-1
# even though r_nm is plotted in nm. Convert to nm^-1 for comparison to MD.
CHILIFE_YSCALE = 10.0


def get_condition_mean(summary_df: pd.DataFrame, condition: str) -> float:
    """
    Expect summary CSV to contain:
      - condition
      - mean_distance_nm
    """
    sub = summary_df.loc[summary_df["condition"] == condition, "mean_distance_nm"]
    if len(sub) == 0:
        raise ValueError(f"No rows found for condition='{condition}'")
    return float(sub.mean())


def infer_mean_columns(dist_df: pd.DataFrame) -> tuple[str, str]:
    """
    Try to find the 1bar and 3kbar mean-distribution columns.
    Prefer canonical names:
      P_1bar_mean, P_3kbar_mean
    """
    cols = list(dist_df.columns)

    if "P_1bar_mean" in cols and "P_3kbar_mean" in cols:
        return "P_1bar_mean", "P_3kbar_mean"

    onebar_candidates = [
        c for c in cols if ("1bar" in c.lower() and "mean" in c.lower())
    ]
    threekbar_candidates = [
        c for c in cols if ("3kbar" in c.lower() and "mean" in c.lower())
    ]

    if len(onebar_candidates) == 1 and len(threekbar_candidates) == 1:
        return onebar_candidates[0], threekbar_candidates[0]

    raise ValueError(
        "Could not infer mean distribution columns. "
        "Expected columns like 'P_1bar_mean' and 'P_3kbar_mean'."
    )


def infer_ci_columns(dist_df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Look for optional CI columns:
      P_1bar_ci_low, P_1bar_ci_high, P_3kbar_ci_low, P_3kbar_ci_high
    """
    cols = set(dist_df.columns)

    c1_lo = "P_1bar_ci_low" if "P_1bar_ci_low" in cols else None
    c1_hi = "P_1bar_ci_high" if "P_1bar_ci_high" in cols else None
    c3_lo = "P_3kbar_ci_low" if "P_3kbar_ci_low" in cols else None
    c3_hi = "P_3kbar_ci_high" if "P_3kbar_ci_high" in cols else None

    return c1_lo, c1_hi, c3_lo, c3_hi


def load_gmx_distance_xvg(xvg_path: Path) -> np.ndarray:
    """
    Read a standard gmx distance XVG file and return the distance column (nm).
    Assumes 2-column data:
        time(ps)   distance(nm)
    Ignores lines starting with # or @.
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
    """
    r_nm = np.asarray(r_nm, dtype=float)
    if r_nm.ndim != 1 or r_nm.size < 2:
        raise ValueError("r_nm must be a 1D array with at least 2 points")

    mids = 0.5 * (r_nm[:-1] + r_nm[1:])
    left = r_nm[0] - 0.5 * (r_nm[1] - r_nm[0])
    right = r_nm[-1] + 0.5 * (r_nm[-1] - r_nm[-2])
    return np.concatenate(([left], mids, [right]))


def histogram_density_on_grid(values_nm: np.ndarray, r_nm: np.ndarray) -> np.ndarray:
    """
    Histogram raw MD distances onto the same r_nm grid as the ChiLife curves.
    Returns a probability density in nm^-1.
    """
    edges = make_bin_edges_from_centers(r_nm)
    hist, _ = np.histogram(values_nm, bins=edges, density=True)
    return hist


def load_model_data(model_cfg: dict) -> dict:
    dist_df = pd.read_csv(model_cfg["dist_csv"])
    summary_df = pd.read_csv(model_cfg["summary_csv"])

    if "r_nm" not in dist_df.columns:
        raise ValueError(f"{model_cfg['dist_csv']} is missing 'r_nm' column")

    mean_col_1bar, mean_col_3kbar = infer_mean_columns(dist_df)
    ci_1_lo, ci_1_hi, ci_3_lo, ci_3_hi = infer_ci_columns(dist_df)

    r_nm = dist_df["r_nm"].to_numpy()

    P_1bar = CHILIFE_YSCALE * dist_df[mean_col_1bar].to_numpy()
    P_3kbar = CHILIFE_YSCALE * dist_df[mean_col_3kbar].to_numpy()

    P_1bar_ci_low = CHILIFE_YSCALE * dist_df[ci_1_lo].to_numpy() if ci_1_lo else None
    P_1bar_ci_high = CHILIFE_YSCALE * dist_df[ci_1_hi].to_numpy() if ci_1_hi else None
    P_3kbar_ci_low = CHILIFE_YSCALE * dist_df[ci_3_lo].to_numpy() if ci_3_lo else None
    P_3kbar_ci_high = CHILIFE_YSCALE * dist_df[ci_3_hi].to_numpy() if ci_3_hi else None

    exp_1bar = get_condition_mean(summary_df, "1bar")
    exp_3kbar = get_condition_mean(summary_df, "3kbar")

    return {
        "title": model_cfg["title"],
        "r_nm": r_nm,
        "P_1bar": P_1bar,
        "P_3kbar": P_3kbar,
        "P_1bar_ci_low": P_1bar_ci_low,
        "P_1bar_ci_high": P_1bar_ci_high,
        "P_3kbar_ci_low": P_3kbar_ci_low,
        "P_3kbar_ci_high": P_3kbar_ci_high,
        "exp_1bar": exp_1bar,
        "exp_3kbar": exp_3kbar,
    }


def main() -> None:
    model_data = [load_model_data(cfg) for cfg in MODEL_FILES]

    md_dist_1bar = load_gmx_distance_xvg(MD_DIST_1BAR)
    md_dist_3kbar = load_gmx_distance_xvg(MD_DIST_3KBAR)

    # Use first model grid as MD histogram grid, and require all panels to match
    r_ref = model_data[0]["r_nm"]
    for d in model_data[1:]:
        if not np.allclose(d["r_nm"], r_ref):
            raise ValueError("Not all model r_nm grids match; cannot overlay a single MD histogram cleanly.")

    md_P_1bar = histogram_density_on_grid(md_dist_1bar, r_ref)
    md_P_3kbar = histogram_density_on_grid(md_dist_3kbar, r_ref)

    md_mean_1bar = float(np.mean(md_dist_1bar))
    md_mean_3kbar = float(np.mean(md_dist_3kbar))

    # Optional sanity checks
    print("MD trapz(P_1bar, r_ref)     =", np.trapz(md_P_1bar, r_ref))
    print("MD trapz(P_3kbar, r_ref)    =", np.trapz(md_P_3kbar, r_ref))
    print("ChiLife trapz(P_1bar, r_nm) =", np.trapz(model_data[0]["P_1bar"], model_data[0]["r_nm"]))
    print("ChiLife trapz(P_3kbar, r_nm)=", np.trapz(model_data[0]["P_3kbar"], model_data[0]["r_nm"]))

    ymax = 0.0
    for d in model_data:
        ymax = max(
            ymax,
            np.nanmax(d["P_1bar"]),
            np.nanmax(d["P_3kbar"]),
        )
        if d["P_1bar_ci_high"] is not None:
            ymax = max(ymax, np.nanmax(d["P_1bar_ci_high"]))
        if d["P_3kbar_ci_high"] is not None:
            ymax = max(ymax, np.nanmax(d["P_3kbar_ci_high"]))

    ymax = max(ymax, np.nanmax(md_P_1bar), np.nanmax(md_P_3kbar))
    ylimit = 1.5 * ymax

    fig, axes = plt.subplots(
        3, 1, figsize=(5.2, 6.6), sharex=True, sharey=True
    )

    for ax, d in zip(axes, model_data):
        r_nm = d["r_nm"]

        # ChiLife mean curves
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

        # Optional CI shading
        if d["P_1bar_ci_low"] is not None and d["P_1bar_ci_high"] is not None:
            ax.fill_between(
                r_nm,
                d["P_1bar_ci_low"],
                d["P_1bar_ci_high"],
                color=COLOR_1BAR,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )

        if d["P_3kbar_ci_low"] is not None and d["P_3kbar_ci_high"] is not None:
            ax.fill_between(
                r_nm,
                d["P_3kbar_ci_low"],
                d["P_3kbar_ci_high"],
                color=COLOR_3KBAR,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )

        # MD residue-residue distributions
        ax.plot(
            r_ref,
            md_P_1bar,
            color=COLOR_1BAR,
            linestyle=MD_LINESTYLE,
            linewidth=MD_LINEWIDTH,
            zorder=4,
            label=rf"MD 1 bar ($\langle r \rangle={md_mean_1bar:.2f}$ nm)",
        )
        ax.plot(
            r_ref,
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
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
        ax.set_xlim(1.5, 8.0)
        ax.set_ylim(0.0, ylimit)

    axes[-1].set_xlabel(r"Distance (nm)", fontsize=12)

    fig.tight_layout()

    out_png = BASE_DIR / "tip4p/stride_1/trajectory_label_vs_md_comparison_3panel.png"
    out_pdf = BASE_DIR / "tip4p/stride_1/trajectory_label_vs_md_comparison_3panel.pdf"
    fig.savefig(out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {out_png}")
    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
    plt.show()