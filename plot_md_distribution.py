"""
Plot the MD residue-residue distance distributions written by
make_md_distribution_csv.py, so they can be visually sanity-checked against
the MD curves shown in plot_traj_label_md_comparison.py.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

DEFAULT_CSV = BASE_DIR / "tip4p/stride_1/MD_trajectory_distributions.csv"
DEFAULT_OUT_PNG = BASE_DIR / "tip4p/stride_1/MD_trajectory_distributions.png"
DEFAULT_OUT_PDF = BASE_DIR / "tip4p/stride_1/MD_trajectory_distributions.pdf"

COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"

MD_LINESTYLE = "--"
MD_LINEWIDTH = 2.0

# CSV stores densities in Angstrom^-1 (see make_md_distribution_csv.py) but
# r_nm is plotted in nm; convert to nm^-1 for plotting/integration, matching
# the convention used in plot_traj_label_md_comparison.py.
ANGSTROM_TO_NM_YSCALE = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = (
        "r_nm",
        "P_1bar_mean",
        "P_1bar_ci_low",
        "P_1bar_ci_high",
        "P_3kbar_mean",
        "P_3kbar_ci_low",
        "P_3kbar_ci_high",
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{args.csv} is missing '{col}' column")

    r_nm = df["r_nm"].to_numpy()
    P_1bar = ANGSTROM_TO_NM_YSCALE * df["P_1bar_mean"].to_numpy()
    P_1bar_lo = ANGSTROM_TO_NM_YSCALE * df["P_1bar_ci_low"].to_numpy()
    P_1bar_hi = ANGSTROM_TO_NM_YSCALE * df["P_1bar_ci_high"].to_numpy()
    P_3kbar = ANGSTROM_TO_NM_YSCALE * df["P_3kbar_mean"].to_numpy()
    P_3kbar_lo = ANGSTROM_TO_NM_YSCALE * df["P_3kbar_ci_low"].to_numpy()
    P_3kbar_hi = ANGSTROM_TO_NM_YSCALE * df["P_3kbar_ci_high"].to_numpy()

    mean_1bar = float(np.trapz(r_nm * P_1bar, r_nm))
    mean_3kbar = float(np.trapz(r_nm * P_3kbar, r_nm))

    print("trapz(P_1bar, r_nm)  =", np.trapz(P_1bar, r_nm))
    print("trapz(P_3kbar, r_nm) =", np.trapz(P_3kbar, r_nm))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(
        r_nm,
        P_1bar,
        color=COLOR_1BAR,
        linestyle=MD_LINESTYLE,
        linewidth=MD_LINEWIDTH,
        label=rf"MD 1 bar ($\langle r \rangle={mean_1bar:.2f}$ nm)",
    )
    ax.fill_between(r_nm, P_1bar_lo, P_1bar_hi, color=COLOR_1BAR, alpha=0.25, linewidth=0)

    ax.plot(
        r_nm,
        P_3kbar,
        color=COLOR_3KBAR,
        linestyle=MD_LINESTYLE,
        linewidth=MD_LINEWIDTH,
        label=rf"MD 3 kbar ($\langle r \rangle={mean_3kbar:.2f}$ nm)",
    )
    ax.fill_between(r_nm, P_3kbar_lo, P_3kbar_hi, color=COLOR_3KBAR, alpha=0.25, linewidth=0)

    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel(r"$P(r)$")
    ax.set_title("MD Residue-Residue Distance Distributions")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
    ax.set_xlim(1.5, 8.0)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=1200, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {args.out_png}")
    print(f"Saved figure to: {args.out_pdf}")


if __name__ == "__main__":
    main()
    plt.show()
