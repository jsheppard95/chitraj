from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


if __name__ == "__main__":
    import scienceplots
    plt.style.use(["science"])
    mpl.rcParams["text.usetex"] = False
    plt.rcParams.update({
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
    })


def mean_from_distribution(r_nm: np.ndarray, P: np.ndarray) -> float:
    denom = np.sum(P)
    if denom <= 0:
        return np.nan
    return np.sum(r_nm * P) / denom


def read_clust_size_xvg(path: str | Path) -> np.ndarray:
    """
    Read cluster populations from gmx cluster clust-size.xvg.

    Assumes data lines are:
        cluster_index   cluster_size
    with header/comment lines starting with @ or #.

    Returns:
        sizes: array of cluster sizes in file order
    """
    sizes = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            sizes.append(float(fields[1]))

    if not sizes:
        raise ValueError(f"No cluster sizes found in {path}")

    return np.asarray(sizes, dtype=float)


def analyze_cluster_ensemble(
    pdb_path,
    clust_size_path,
    site1: int,
    site2: int,
    label_name: str,
    r: np.ndarray,
    sample: int | None = None,
    rotlib: str | None = None,
):
    """
    Analyze a multi-model clusters.pdb file.

    Each MODEL in the PDB is treated as one cluster representative.
    Cluster populations are read from clust-size.xvg and used as weights.
    """
    import MDAnalysis as mda
    import chilife as xl

    r_nm = r / 10.0

    sizes = read_clust_size_xvg(clust_size_path)
    weights = sizes / np.sum(sizes)

    # Each MODEL in the PDB becomes one trajectory frame in MDAnalysis
    U = mda.Universe(str(pdb_path))

    n_models = len(U.trajectory)
    if n_models != len(sizes):
        raise ValueError(
            f"Mismatch between number of models in {pdb_path} ({n_models}) "
            f"and number of populations in {clust_size_path} ({len(sizes)})"
        )

    P_clusters = []
    mean_r_clusters = []
    cluster_ids = []

    for i, ts in enumerate(U.trajectory):
        cluster_id = i + 1
        print(
            f"Cluster {cluster_id:4d} / {n_models:4d}   "
            f"frame={ts.frame:4d}   weight={weights[i]:.6f}",
            flush=True,
        )

        kwargs = {"site": site1, "protein": U}
        if sample is not None:
            kwargs["sample"] = sample
        if rotlib is not None:
            kwargs["rotlib"] = rotlib
        L1 = xl.SpinLabel(label_name, **kwargs)

        kwargs = {"site": site2, "protein": U}
        if sample is not None:
            kwargs["sample"] = sample
        if rotlib is not None:
            kwargs["rotlib"] = rotlib
        L2 = xl.SpinLabel(label_name, **kwargs)

        P = xl.distance_distribution(L1, L2, r=r)

        if not np.all(np.isfinite(P)) or np.sum(P) <= 0:
            raise ValueError(f"Invalid distribution for cluster {cluster_id}")

        mean_r = mean_from_distribution(r_nm, P)

        cluster_ids.append(cluster_id)
        P_clusters.append(P)
        mean_r_clusters.append(mean_r)

    P_clusters = np.asarray(P_clusters, dtype=float)
    mean_r_clusters = np.asarray(mean_r_clusters, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=int)

    # Weighted ensemble average
    weighted_mean_P = np.sum(P_clusters * weights[:, None], axis=0)
    weighted_mean_r = np.sum(mean_r_clusters * weights)

    return {
        "cluster_ids": cluster_ids,
        "sizes": sizes,
        "weights": weights,
        "P_clusters": P_clusters,
        "mean_r_clusters": mean_r_clusters,
        "mean_P": weighted_mean_P,
        "mean_r": weighted_mean_r,
        "r_nm": r_nm,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Population-weighted ChiLife DEER distributions from GROMACS cluster representatives."
    )

    parser.add_argument("--pdb1", required=True, help="1 bar clusters.pdb")
    parser.add_argument("--xvg1", required=True, help="1 bar clust-size.xvg")
    parser.add_argument("--pdb2", required=True, help="3 kbar clusters.pdb")
    parser.add_argument("--xvg2", required=True, help="3 kbar clust-size.xvg")

    parser.add_argument("--site1-cond1", type=int, required=True)
    parser.add_argument("--site2-cond1", type=int, required=True)
    parser.add_argument("--site1-cond2", type=int, required=True)
    parser.add_argument("--site2-cond2", type=int, required=True)

    parser.add_argument("--label-name", required=True, choices=["R1M", "GTN"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--rotlib", default=None)

    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=80.0)
    parser.add_argument("--r-npts", type=int, default=256)

    parser.add_argument("--outdir", default="cluster_output")
    parser.add_argument("--prefix", default="cluster_weighted")

    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = np.linspace(args.r_min, args.r_max, args.r_npts)

    print("Analyzing 1 bar clusters...", flush=True)
    res_1bar = analyze_cluster_ensemble(
        pdb_path=args.pdb1,
        clust_size_path=args.xvg1,
        site1=args.site1_cond1,
        site2=args.site2_cond1,
        label_name=args.label_name,
        r=r,
        sample=args.sample,
        rotlib=args.rotlib,
    )

    print("Analyzing 3 kbar clusters...", flush=True)
    res_3kbar = analyze_cluster_ensemble(
        pdb_path=args.pdb2,
        clust_size_path=args.xvg2,
        site1=args.site1_cond2,
        site2=args.site2_cond2,
        label_name=args.label_name,
        r=r,
        sample=args.sample,
        rotlib=args.rotlib,
    )

    print(
        f"1 bar  weighted <r> = {res_1bar['mean_r']:.3f} nm",
        flush=True,
    )
    print(
        f"3 kbar weighted <r> = {res_3kbar['mean_r']:.3f} nm",
        flush=True,
    )

    # Per-cluster summary
    summary_1 = pd.DataFrame({
        "condition": "1bar",
        "cluster_id": res_1bar["cluster_ids"],
        "cluster_size": res_1bar["sizes"],
        "weight": res_1bar["weights"],
        "mean_distance_nm": res_1bar["mean_r_clusters"],
    })
    summary_3 = pd.DataFrame({
        "condition": "3kbar",
        "cluster_id": res_3kbar["cluster_ids"],
        "cluster_size": res_3kbar["sizes"],
        "weight": res_3kbar["weights"],
        "mean_distance_nm": res_3kbar["mean_r_clusters"],
    })
    pd.concat([summary_1, summary_3], ignore_index=True).to_csv(
        outdir / f"{args.prefix}_per_cluster_summary.csv",
        index=False,
    )

    # Per-cluster distributions
    cluster_dist_1 = pd.DataFrame(
        res_1bar["P_clusters"].T,
        columns=[f"cluster_{i}" for i in res_1bar["cluster_ids"]],
    )
    cluster_dist_1.insert(0, "r_nm", res_1bar["r_nm"])
    cluster_dist_1.to_csv(
        outdir / f"{args.prefix}_1bar_cluster_distributions.csv",
        index=False,
    )

    cluster_dist_3 = pd.DataFrame(
        res_3kbar["P_clusters"].T,
        columns=[f"cluster_{i}" for i in res_3kbar["cluster_ids"]],
    )
    cluster_dist_3.insert(0, "r_nm", res_3kbar["r_nm"])
    cluster_dist_3.to_csv(
        outdir / f"{args.prefix}_3kbar_cluster_distributions.csv",
        index=False,
    )

    # Final weighted distributions in same format as trajectory script
    # CI columns are NaN placeholders for now
    dist_df = pd.DataFrame({
        "r_nm": res_1bar["r_nm"],
        "P_1bar_mean": res_1bar["mean_P"],
        "P_1bar_ci_low": np.full_like(res_1bar["mean_P"], np.nan),
        "P_1bar_ci_high": np.full_like(res_1bar["mean_P"], np.nan),
        "P_3kbar_mean": res_3kbar["mean_P"],
        "P_3kbar_ci_low": np.full_like(res_3kbar["mean_P"], np.nan),
        "P_3kbar_ci_high": np.full_like(res_3kbar["mean_P"], np.nan),
    })
    dist_df.to_csv(
        outdir / f"{args.prefix}_trajectory_distributions.csv",
        index=False,
    )

    # Save raw arrays
    np.savez_compressed(
        outdir / f"{args.prefix}_cluster_weighted_raw.npz",
        r_nm=res_1bar["r_nm"],
        cluster_ids_1bar=res_1bar["cluster_ids"],
        sizes_1bar=res_1bar["sizes"],
        weights_1bar=res_1bar["weights"],
        P_clusters_1bar=res_1bar["P_clusters"],
        mean_r_clusters_1bar=res_1bar["mean_r_clusters"],
        mean_P_1bar=res_1bar["mean_P"],
        mean_r_1bar=res_1bar["mean_r"],
        cluster_ids_3kbar=res_3kbar["cluster_ids"],
        sizes_3kbar=res_3kbar["sizes"],
        weights_3kbar=res_3kbar["weights"],
        P_clusters_3kbar=res_3kbar["P_clusters"],
        mean_r_clusters_3kbar=res_3kbar["mean_r_clusters"],
        mean_P_3kbar=res_3kbar["mean_P"],
        mean_r_3kbar=res_3kbar["mean_r"],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(
        res_1bar["r_nm"],
        res_1bar["mean_P"],
        label=rf"1 bar ($\langle r \rangle={res_1bar['mean_r']:.2f}$ nm)",
        linewidth=2.5,
    )
    ax.plot(
        res_3kbar["r_nm"],
        res_3kbar["mean_P"],
        label=rf"3 kbar ($\langle r \rangle={res_3kbar['mean_r']:.2f}$ nm)",
        linewidth=2.5,
    )

    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel(r"$P(r)$")
    ax.set_xlim(1.5, 6.0)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        outdir / f"{args.prefix}_trajectory_average.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()