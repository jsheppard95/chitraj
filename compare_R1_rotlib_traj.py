from __future__ import annotations

import argparse
from pathlib import Path
import warnings

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


def analyze_trajectory_label_pair(
    universe,
    site1: int,
    site2: int,
    label_name: str,
    r: np.ndarray,
    sample: int | None = None,
    rotlib: str | None = None,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
):
    import chilife as xl

    r_nm = r / 10.0
    times_ps = []
    frame_ids = []
    P_frames = []
    mean_r_frames = []

    skipped_frames = []

    if max_frames is None:
        stop_frame = None
    else:
        stop_frame = start_frame + max_frames * stride

    traj_slice = universe.trajectory[start_frame:stop_frame:stride]

    print(
        f"Processing frames starting at {start_frame}, "
        f"stop={stop_frame}, stride={stride}"
    )

    for ts in traj_slice:
        print(f"Frame {ts.frame:6d}  time = {ts.time:10.3f} ps", flush=True)

        try:
            kwargs = {"site": site1, "protein": universe}
            if sample is not None:
                kwargs["sample"] = sample
            if rotlib is not None:
                kwargs["rotlib"] = rotlib
            L1 = xl.SpinLabel(label_name, **kwargs)

            kwargs = {"site": site2, "protein": universe}
            if sample is not None:
                kwargs["sample"] = sample
            if rotlib is not None:
                kwargs["rotlib"] = rotlib
            L2 = xl.SpinLabel(label_name, **kwargs)

            P = xl.distance_distribution(L1, L2, r=r)

            if not np.all(np.isfinite(P)) or np.sum(P) <= 0:
                raise ValueError("Non-finite or zero-sum distribution")

        except Exception as e:
            print(f"Skipping frame {ts.frame}: {e}", flush=True)
            skipped_frames.append(ts.frame)
            continue

        mean_r = mean_from_distribution(r_nm, P)

        frame_ids.append(ts.frame)
        times_ps.append(ts.time)
        P_frames.append(P)
        mean_r_frames.append(mean_r)

    print(
        f"\nSkipped {len(skipped_frames)} frames out of "
        f"{len(frame_ids) + len(skipped_frames)}",
        flush=True,
    )

    frame_ids = np.asarray(frame_ids, dtype=int)
    times_ps = np.asarray(times_ps, dtype=float)
    mean_r_frames = np.asarray(mean_r_frames, dtype=float)

    if len(P_frames) == 0:
        raise RuntimeError("No valid frames were processed in this chunk.")

    P_frames = np.asarray(P_frames, dtype=float)
    mean_P = np.mean(P_frames, axis=0)
    std_P = np.std(P_frames, axis=0, ddof=1) if len(P_frames) > 1 else np.zeros_like(mean_P)

    return {
        "frame_ids": frame_ids,
        "times_ps": times_ps,
        "P_frames": P_frames,
        "mean_r_frames": mean_r_frames,
        "mean_P": mean_P,
        "std_P": std_P,
        "r_nm": r_nm,
        "skipped_frames": np.asarray(skipped_frames, dtype=int),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunked ChiLife distance-distribution analysis over trajectory frames."
    )

    parser.add_argument("--top1", required=True, help="Topology for condition 1")
    parser.add_argument("--traj1", required=True, help="Trajectory for condition 1")
    parser.add_argument("--top2", required=True, help="Topology for condition 2")
    parser.add_argument("--traj2", required=True, help="Trajectory for condition 2")

    parser.add_argument("--site1-cond1", type=int, required=True)
    parser.add_argument("--site2-cond1", type=int, required=True)
    parser.add_argument("--site1-cond2", type=int, required=True)
    parser.add_argument("--site2-cond2", type=int, required=True)

    parser.add_argument("--label-name", required=True, choices=["R1M", "GTN"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--rotlib", default=None)

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=80.0)
    parser.add_argument("--r-npts", type=int, default=256)

    parser.add_argument("--outdir", default="chunk_output")
    parser.add_argument("--tag", default=None, help="Optional tag for output filenames")

    return parser.parse_args()


def main():
    import MDAnalysis as mda

    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = np.linspace(args.r_min, args.r_max, args.r_npts)

    print("Loading universes...", flush=True)
    U1 = mda.Universe(args.top1, args.traj1)
    U2 = mda.Universe(args.top2, args.traj2)

    # keep your original segid/chainID fix if needed
    U2.atoms.segments.segids = ["A"]
    U2.atoms.chainIDs = ["A"] * U2.atoms.n_atoms

    tag = args.tag
    if tag is None:
        sample_tag = "no_sample" if args.sample is None else f"sample_{args.sample}"
        tag = f"{args.label_name}_{sample_tag}_start{args.start_frame}_n{args.max_frames}"

    print(f"Running chunk tag: {tag}", flush=True)

    res_1 = analyze_trajectory_label_pair(
        universe=U1,
        site1=args.site1_cond1,
        site2=args.site2_cond1,
        label_name=args.label_name,
        r=r,
        sample=args.sample,
        rotlib=args.rotlib,
        stride=args.stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    res_2 = analyze_trajectory_label_pair(
        universe=U2,
        site1=args.site1_cond2,
        site2=args.site2_cond2,
        label_name=args.label_name,
        r=r,
        sample=args.sample,
        rotlib=args.rotlib,
        stride=args.stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    summary_1 = pd.DataFrame({
        "condition": "1bar",
        "frame": res_1["frame_ids"],
        "time_ps": res_1["times_ps"],
        "mean_distance_nm": res_1["mean_r_frames"],
    })

    summary_2 = pd.DataFrame({
        "condition": "3kbar",
        "frame": res_2["frame_ids"],
        "time_ps": res_2["times_ps"],
        "mean_distance_nm": res_2["mean_r_frames"],
    })

    pd.concat([summary_1, summary_2], ignore_index=True).to_csv(
        outdir / f"{tag}_per_frame_summary.csv",
        index=False,
    )

    np.savez_compressed(
        outdir / f"{tag}_raw_distributions.npz",
        r_nm=res_1["r_nm"],
        frame_ids_1bar=res_1["frame_ids"],
        times_ps_1bar=res_1["times_ps"],
        P_frames_1bar=res_1["P_frames"],
        mean_r_frames_1bar=res_1["mean_r_frames"],
        skipped_frames_1bar=res_1["skipped_frames"],
        frame_ids_3kbar=res_2["frame_ids"],
        times_ps_3kbar=res_2["times_ps"],
        P_frames_3kbar=res_2["P_frames"],
        mean_r_frames_3kbar=res_2["mean_r_frames"],
        skipped_frames_3kbar=res_2["skipped_frames"],
    )

    print("Chunk complete.", flush=True)


if __name__ == "__main__":
    main()