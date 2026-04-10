from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def bootstrap_mean_distribution(
    P_frames: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive bootstrap over frames.
    P_frames shape = (n_frames, n_r)
    Returns lower/upper 95% CI for mean P(r).
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    n_frames = P_frames.shape[0]
    if n_frames == 0:
        raise ValueError("P_frames is empty")

    boot_means = np.empty((n_boot, P_frames.shape[1]), dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n_frames, size=n_frames)
        boot_means[i] = P_frames[idx].mean(axis=0)

    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)
    return lower, upper


def block_bootstrap_mean_scalar(
    values: np.ndarray,
    block_size: int,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Block bootstrap CI for a scalar time series, e.g. <r>_t.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan

    n_blocks = int(np.ceil(n / block_size))
    boot_stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        resampled = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n - block_size + 1))
            resampled.append(values[start:start + block_size])
        resampled = np.concatenate(resampled)[:n]
        boot_stats[i] = np.mean(resampled)

    return np.percentile(boot_stats, 2.5), np.percentile(boot_stats, 97.5)


def extract_chunk_index(path: Path) -> int:
    """
    Extract integer from ..._chunk7_... or ..._chunk7.npz
    """
    m = re.search(r"_chunk(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def load_and_concat_chunks(npz_files: list[Path]) -> dict[str, np.ndarray]:
    if not npz_files:
        raise FileNotFoundError("No chunk .npz files found.")

    npz_files = sorted(npz_files, key=extract_chunk_index)

    r_nm_ref = None

    frame_ids_1bar = []
    times_ps_1bar = []
    P_frames_1bar = []
    mean_r_frames_1bar = []
    skipped_frames_1bar = []

    frame_ids_3kbar = []
    times_ps_3kbar = []
    P_frames_3kbar = []
    mean_r_frames_3kbar = []
    skipped_frames_3kbar = []

    for f in npz_files:
        print(f"Loading {f}")
        data = np.load(f)

        r_nm = data["r_nm"]
        if r_nm_ref is None:
            r_nm_ref = r_nm
        elif not np.allclose(r_nm_ref, r_nm):
            raise ValueError(f"Inconsistent r_nm grid in {f}")

        frame_ids_1bar.append(data["frame_ids_1bar"])
        times_ps_1bar.append(data["times_ps_1bar"])
        P_frames_1bar.append(data["P_frames_1bar"])
        mean_r_frames_1bar.append(data["mean_r_frames_1bar"])
        skipped_frames_1bar.append(data["skipped_frames_1bar"])

        frame_ids_3kbar.append(data["frame_ids_3kbar"])
        times_ps_3kbar.append(data["times_ps_3kbar"])
        P_frames_3kbar.append(data["P_frames_3kbar"])
        mean_r_frames_3kbar.append(data["mean_r_frames_3kbar"])
        skipped_frames_3kbar.append(data["skipped_frames_3kbar"])

    result = {
        "r_nm": r_nm_ref,
        "frame_ids_1bar": np.concatenate(frame_ids_1bar),
        "times_ps_1bar": np.concatenate(times_ps_1bar),
        "P_frames_1bar": np.vstack(P_frames_1bar),
        "mean_r_frames_1bar": np.concatenate(mean_r_frames_1bar),
        "skipped_frames_1bar": np.concatenate(skipped_frames_1bar) if skipped_frames_1bar else np.array([], dtype=int),
        "frame_ids_3kbar": np.concatenate(frame_ids_3kbar),
        "times_ps_3kbar": np.concatenate(times_ps_3kbar),
        "P_frames_3kbar": np.vstack(P_frames_3kbar),
        "mean_r_frames_3kbar": np.concatenate(mean_r_frames_3kbar),
        "skipped_frames_3kbar": np.concatenate(skipped_frames_3kbar) if skipped_frames_3kbar else np.array([], dtype=int),
    }

    return result


def sort_by_frame(frame_ids: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(frame_ids)
    out = [frame_ids[order]]
    for arr in arrays:
        out.append(arr[order])
    return tuple(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine chunked ChiLife outputs and reproduce final CSV/plot."
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("chunk_output"),
        help="Directory containing chunk .npz files",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix identifying one mode, e.g. GTN_sample_5000 or R1_no_sample",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("combined_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Bootstrap replicates for mean P(r)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Block size for scalar mean-distance CI",
    )
    parser.add_argument(
        "--plot-xmin",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--plot-xmax",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    npz_files = list(args.chunk_dir.glob(f"{args.prefix}_chunk*_raw_distributions.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No files found matching {args.prefix}_chunk*_raw_distributions.npz "
            f"in {args.chunk_dir}"
        )

    combined = load_and_concat_chunks(npz_files)

    (
        frame_ids_1bar,
        times_ps_1bar,
        P_frames_1bar,
        mean_r_frames_1bar,
    ) = sort_by_frame(
        combined["frame_ids_1bar"],
        combined["times_ps_1bar"],
        combined["P_frames_1bar"],
        combined["mean_r_frames_1bar"],
    )

    (
        frame_ids_3kbar,
        times_ps_3kbar,
        P_frames_3kbar,
        mean_r_frames_3kbar,
    ) = sort_by_frame(
        combined["frame_ids_3kbar"],
        combined["times_ps_3kbar"],
        combined["P_frames_3kbar"],
        combined["mean_r_frames_3kbar"],
    )

    r_nm = combined["r_nm"]

    # Mean distributions
    mean_P_1bar = np.mean(P_frames_1bar, axis=0)
    mean_P_3kbar = np.mean(P_frames_3kbar, axis=0)

    # Bootstrap 95% CI for mean distribution
    lo_1bar, hi_1bar = bootstrap_mean_distribution(P_frames_1bar, n_boot=args.n_boot)
    lo_3kbar, hi_3kbar = bootstrap_mean_distribution(P_frames_3kbar, n_boot=args.n_boot)

    # Scalar mean distances and block bootstrap CI
    mean_r_1bar = np.mean(mean_r_frames_1bar)
    mean_r_3kbar = np.mean(mean_r_frames_3kbar)

    ci1_lo, ci1_hi = block_bootstrap_mean_scalar(
        mean_r_frames_1bar,
        block_size=args.block_size,
        n_boot=1000,
    )
    ci3_lo, ci3_hi = block_bootstrap_mean_scalar(
        mean_r_frames_3kbar,
        block_size=args.block_size,
        n_boot=1000,
    )

    print(f"1 bar  <r> = {mean_r_1bar:.3f} nm   95% CI [{ci1_lo:.3f}, {ci1_hi:.3f}]")
    print(f"3 kbar <r> = {mean_r_3kbar:.3f} nm   95% CI [{ci3_lo:.3f}, {ci3_hi:.3f}]")

    # Per-frame summary CSV
    summary_1 = pd.DataFrame({
        "condition": "1bar",
        "frame": frame_ids_1bar,
        "time_ps": times_ps_1bar,
        "mean_distance_nm": mean_r_frames_1bar,
    })
    summary_3 = pd.DataFrame({
        "condition": "3kbar",
        "frame": frame_ids_3kbar,
        "time_ps": times_ps_3kbar,
        "mean_distance_nm": mean_r_frames_3kbar,
    })
    summary_df = pd.concat([summary_1, summary_3], ignore_index=True)
    summary_df.to_csv(
        args.outdir / f"{args.prefix}_per_frame_summary.csv",
        index=False,
    )

    # Final distribution CSV in the same format as before
    dist_df = pd.DataFrame({
        "r_nm": r_nm,
        "P_1bar_mean": mean_P_1bar,
        "P_1bar_ci_low": lo_1bar,
        "P_1bar_ci_high": hi_1bar,
        "P_3kbar_mean": mean_P_3kbar,
        "P_3kbar_ci_low": lo_3kbar,
        "P_3kbar_ci_high": hi_3kbar,
    })
    dist_df.to_csv(
        args.outdir / f"{args.prefix}_trajectory_distributions.csv",
        index=False,
    )

    # Save combined raw arrays too
    np.savez_compressed(
        args.outdir / f"{args.prefix}_combined_raw.npz",
        r_nm=r_nm,
        frame_ids_1bar=frame_ids_1bar,
        times_ps_1bar=times_ps_1bar,
        P_frames_1bar=P_frames_1bar,
        mean_r_frames_1bar=mean_r_frames_1bar,
        frame_ids_3kbar=frame_ids_3kbar,
        times_ps_3kbar=times_ps_3kbar,
        P_frames_3kbar=P_frames_3kbar,
        mean_r_frames_3kbar=mean_r_frames_3kbar,
        skipped_frames_1bar=combined["skipped_frames_1bar"],
        skipped_frames_3kbar=combined["skipped_frames_3kbar"],
    )

    # Optional skipped-frame CSVs
    pd.DataFrame({"frame": combined["skipped_frames_1bar"]}).to_csv(
        args.outdir / f"{args.prefix}_skipped_frames_1bar.csv",
        index=False,
    )
    pd.DataFrame({"frame": combined["skipped_frames_3kbar"]}).to_csv(
        args.outdir / f"{args.prefix}_skipped_frames_3kbar.csv",
        index=False,
    )

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(
            r_nm,
            mean_P_1bar,
            label=rf"1 bar ($\langle r \rangle={mean_r_1bar:.2f}$ nm)",
            linewidth=2.5,
        )
        ax.fill_between(r_nm, lo_1bar, hi_1bar, alpha=0.25)

        ax.plot(
            r_nm,
            mean_P_3kbar,
            label=rf"3 kbar ($\langle r \rangle={mean_r_3kbar:.2f}$ nm)",
            linewidth=2.5,
        )
        ax.fill_between(r_nm, lo_3kbar, hi_3kbar, alpha=0.25)

        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel(r"$P(r)$")
        ax.set_xlim(args.plot_xmin, args.plot_xmax)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            args.outdir / f"{args.prefix}_trajectory_average.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()