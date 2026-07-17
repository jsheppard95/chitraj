"""
Histogram the raw MD residue-residue distances (MD_DIST_1BAR, MD_DIST_3KBAR)
onto the same r_nm grid used by the ChiLife trajectory-distribution CSVs, and
write the result in the same CSV format as e.g.
tip4p/stride_1/gtn_sample_5000/combined_output/GTN_sample_5000_trajectory_distributions.csv

Columns: r_nm, P_1bar_mean, P_1bar_ci_low, P_1bar_ci_high,
         P_3kbar_mean, P_3kbar_ci_low, P_3kbar_ci_high

P values are densities in Angstrom^-1 (probability per Angstrom), matching
the native units of the ChiLife trajectory-distribution CSVs, even though the
r_nm grid itself is in nm.

ci_low/ci_high come from a moving-block bootstrap over the raw distance time
series (same block bootstrap used for the ChiLife distributions in
combine_chilife_chunks.py), so --block-size should be chosen the same way,
e.g. via estimate_autocorrelation_time.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

# Raw MD residue-residue distances from gmx distance
MD_DIST_1BAR = BASE_DIR / "tip4p/res406_res537_1bar_dist.xvg"
MD_DIST_3KBAR = BASE_DIR / "tip4p/res406_res537_3kbar_dist.xvg"

# Source of the r_nm grid to histogram onto
GRID_SOURCE_CSV = (
    BASE_DIR
    / "tip4p/stride_1/r1m_sample_none/combined_output/R1_no_sample_trajectory_distributions.csv"
)

DEFAULT_OUT_CSV = BASE_DIR / "tip4p/stride_1/MD_trajectory_distributions.csv"

# Histogramming onto the nm grid gives a density in nm^-1; divide by this to
# convert to Angstrom^-1 (1 nm = 10 Angstrom), matching ChiLife's native units.
NM_TO_ANGSTROM = 10.0


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
    Returns a probability density in Angstrom^-1.
    """
    edges = make_bin_edges_from_centers(r_nm)
    hist, _ = np.histogram(values_nm, bins=edges, density=True)
    return hist / NM_TO_ANGSTROM


def block_bootstrap_histogram_density(
    values_nm: np.ndarray,
    r_nm: np.ndarray,
    block_size: int,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moving-block bootstrap CI for the histogram density estimate.
    values_nm must be in original time order (as read from the xvg file).
    Returns lower/upper 95% CI, one value per r_nm bin.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    values_nm = np.asarray(values_nm, dtype=float)
    n = len(values_nm)
    if n == 0:
        raise ValueError("values_nm is empty")

    n_blocks = int(np.ceil(n / block_size))
    boot_hist = np.empty((n_boot, len(r_nm)), dtype=float)

    for i in range(n_boot):
        resampled = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n - block_size + 1))
            resampled.append(values_nm[start:start + block_size])
        resampled = np.concatenate(resampled)[:n]
        boot_hist[i] = histogram_density_on_grid(resampled, r_nm)

    lower = np.percentile(boot_hist, 2.5, axis=0)
    upper = np.percentile(boot_hist, 97.5, axis=0)
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-source-csv",
        type=Path,
        default=GRID_SOURCE_CSV,
        help="CSV file to take the r_nm grid from (must have an 'r_nm' column).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Path to write the MD histogram distribution CSV.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help=(
            "Block size (in frames) for the moving-block bootstrap CI. "
            "Choose via estimate_autocorrelation_time.py, not a guess."
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Bootstrap replicates for the histogram density CI.",
    )
    args = parser.parse_args()

    grid_df = pd.read_csv(args.grid_source_csv)
    if "r_nm" not in grid_df.columns:
        raise ValueError(f"{args.grid_source_csv} is missing 'r_nm' column")
    r_nm = grid_df["r_nm"].to_numpy()

    md_dist_1bar = load_gmx_distance_xvg(MD_DIST_1BAR)
    md_dist_3kbar = load_gmx_distance_xvg(MD_DIST_3KBAR)

    md_P_1bar = histogram_density_on_grid(md_dist_1bar, r_nm)
    md_P_3kbar = histogram_density_on_grid(md_dist_3kbar, r_nm)

    ci_1bar_lo, ci_1bar_hi = block_bootstrap_histogram_density(
        md_dist_1bar, r_nm, block_size=args.block_size, n_boot=args.n_boot
    )
    ci_3kbar_lo, ci_3kbar_hi = block_bootstrap_histogram_density(
        md_dist_3kbar, r_nm, block_size=args.block_size, n_boot=args.n_boot
    )

    out_df = pd.DataFrame(
        {
            "r_nm": r_nm,
            "P_1bar_mean": md_P_1bar,
            "P_1bar_ci_low": ci_1bar_lo,
            "P_1bar_ci_high": ci_1bar_hi,
            "P_3kbar_mean": md_P_3kbar,
            "P_3kbar_ci_low": ci_3kbar_lo,
            "P_3kbar_ci_high": ci_3kbar_hi,
        }
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved MD distribution CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
