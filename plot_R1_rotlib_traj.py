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


COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"


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
    but fall back to looser matching if needed.
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


def load_model_data(model_cfg: dict) -> dict:
    dist_df = pd.read_csv(model_cfg["dist_csv"])
    summary_df = pd.read_csv(model_cfg["summary_csv"])

    if "r_nm" not in dist_df.columns:
        raise ValueError(f"{model_cfg['dist_csv']} is missing 'r_nm' column")

    mean_col_1bar, mean_col_3kbar = infer_mean_columns(dist_df)
    ci_1_lo, ci_1_hi, ci_3_lo, ci_3_hi = infer_ci_columns(dist_df)

    exp_1bar = get_condition_mean(summary_df, "1bar")
    exp_3kbar = get_condition_mean(summary_df, "3kbar")

    return {
        "title": model_cfg["title"],
        "r_nm": dist_df["r_nm"].to_numpy(),
        "P_1bar": dist_df[mean_col_1bar].to_numpy(),
        "P_3kbar": dist_df[mean_col_3kbar].to_numpy(),
        "P_1bar_ci_low": dist_df[ci_1_lo].to_numpy() if ci_1_lo else None,
        "P_1bar_ci_high": dist_df[ci_1_hi].to_numpy() if ci_1_hi else None,
        "P_3kbar_ci_low": dist_df[ci_3_lo].to_numpy() if ci_3_lo else None,
        "P_3kbar_ci_high": dist_df[ci_3_hi].to_numpy() if ci_3_hi else None,
        "exp_1bar": exp_1bar,
        "exp_3kbar": exp_3kbar,
    }


def main() -> None:
    model_data = [load_model_data(cfg) for cfg in MODEL_FILES]

    # Global y-limit for consistent scaling across panels
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

    ylimit = 1.5 * ymax

    fig, axes = plt.subplots(
        3, 1, figsize=(5.2, 6.2), sharex=True, sharey=True
    )

    for ax, d in zip(axes, model_data):
        r_nm = d["r_nm"]

        ax.plot(
            r_nm,
            d["P_1bar"],
            color=COLOR_1BAR,
            linewidth=2.5,
            label=rf"1 bar ($\langle r \rangle={d['exp_1bar']:.2f}$ nm)",
        )
        ax.plot(
            r_nm,
            d["P_3kbar"],
            color=COLOR_3KBAR,
            linewidth=2.5,
            label=rf"3 kbar ($\langle r \rangle={d['exp_3kbar']:.2f}$ nm)",
        )

        if d["P_1bar_ci_low"] is not None and d["P_1bar_ci_high"] is not None:
            ax.fill_between(
                r_nm,
                d["P_1bar_ci_low"],
                d["P_1bar_ci_high"],
                color=COLOR_1BAR,
                alpha=0.22,
                linewidth=0,
            )

        if d["P_3kbar_ci_low"] is not None and d["P_3kbar_ci_high"] is not None:
            ax.fill_between(
                r_nm,
                d["P_3kbar_ci_low"],
                d["P_3kbar_ci_high"],
                color=COLOR_3KBAR,
                alpha=0.22,
                linewidth=0,
            )

        ax.set_ylabel(r"$P(r)$", fontsize=12)
        ax.set_title(d["title"], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
        ax.set_xlim(1.5, 8.0)
        ax.set_ylim(0.0, ylimit)

    axes[-1].set_xlabel(r"Distance (nm)", fontsize=12)

    fig.tight_layout()

    out_png = BASE_DIR / "tip4p/stride_1/trajectory_label_comparison_3panel.png"
    out_pdf = BASE_DIR / "tip4p/stride_1/trajectory_label_comparison_3panel.pdf"
    fig.savefig(out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {out_png}")
    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
    plt.show()