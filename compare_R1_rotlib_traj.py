import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

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


def analyze_trajectory_label_pair(
    universe,
    site1: int,
    site2: int,
    label_name: str,
    r: np.ndarray,
    sample: int | None = None,
    rotlib: str | None = None,
    stride: int = 100,
    max_frames: int | None = None,
):
    import chilife as xl

    r_nm = r / 10.0
    times_ps = []
    frame_ids = []
    P_frames = []
    mean_r_frames = []

    n_done = 0
    skipped_frames = []
    for ts in universe.trajectory[::stride]:
        if max_frames is not None and n_done >= max_frames:
            break

        print(f"Frame {ts.frame:6d}  time = {ts.time:10.3f} ps")
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
            print(f"Skipping frame {ts.frame}: {e}")
            skipped_frames.append(ts.frame)
            continue

        mean_r = mean_from_distribution(r_nm, P)

        frame_ids.append(ts.frame)
        times_ps.append(ts.time)
        P_frames.append(P)
        mean_r_frames.append(mean_r)
        n_done += 1
    print(f"\nSkipped {len(skipped_frames)} frames out of {len(frame_ids)+len(skipped_frames)}")
    pd.DataFrame({"frame": skipped_frames}).to_csv(
        "skipped_frames.csv", index=False
    )
    P_frames = np.asarray(P_frames, dtype=float)
    mean_r_frames = np.asarray(mean_r_frames, dtype=float)
    times_ps = np.asarray(times_ps, dtype=float)
    frame_ids = np.asarray(frame_ids, dtype=int)

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
    }


def main():
    import MDAnalysis as mda

    base = Path(__file__).parent

    # Example: replace with topology+trajectory
    U1 = mda.Universe("./JS_1bar_equib_tip4p.pdb", "./JS_1bar.xtc")
    U2 = mda.Universe("./JS_3kbar_equib_tip4p.pdb", "./JS_3kbar.xtc")
    U2.atoms.segments.segids = ["A"]
    U2.atoms.chainIDs = ["A"] * U2.atoms.n_atoms

    sites_1bar = (406, 537)
    sites_3kbar = (4, 135)

    r = np.linspace(0, 80, 256)
    rotlib = "GTN_rotlib.npz"  # GTN
    #rotlib = None  # R1

    stride = 1
    sample = 5000

    print("Analyzing GTN trajectories (Sampling 5000)...")
    res_1bar = analyze_trajectory_label_pair(
        universe=U1,
        site1=sites_1bar[0],
        site2=sites_1bar[1],
        label_name="GTN",
        r=r,
        sample=sample,
        rotlib=rotlib,
        stride=stride,
        max_frames=None,
    )

    res_3kbar = analyze_trajectory_label_pair(
        universe=U2,
        site1=sites_3kbar[0],
        site2=sites_3kbar[1],
        label_name="GTN",
        r=r,
        sample=sample,
        rotlib=rotlib,
        stride=stride,
        max_frames=None,
    )

    # Bootstrap CI for mean distribution
    lo_1, hi_1 = bootstrap_mean_distribution(res_1bar["P_frames"], n_boot=500)
    lo_3, hi_3 = bootstrap_mean_distribution(res_3kbar["P_frames"], n_boot=500)

    # Block-bootstrap CI for scalar <r>
    ci1_lo, ci1_hi = block_bootstrap_mean_scalar(
        res_1bar["mean_r_frames"], block_size=5, n_boot=1000
    )
    ci3_lo, ci3_hi = block_bootstrap_mean_scalar(
        res_3kbar["mean_r_frames"], block_size=5, n_boot=1000
    )

    mean_r_1 = np.mean(res_1bar["mean_r_frames"])
    mean_r_3 = np.mean(res_3kbar["mean_r_frames"])

    print(f"1 bar  <r> = {mean_r_1:.3f} nm   95% CI [{ci1_lo:.3f}, {ci1_hi:.3f}]")
    print(f"3 kbar <r> = {mean_r_3:.3f} nm   95% CI [{ci3_lo:.3f}, {ci3_hi:.3f}]")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))

    r_nm = res_1bar["r_nm"]

    ax.plot(
        r_nm,
        res_1bar["mean_P"],
        label=rf"1 bar ($\langle r \rangle={mean_r_1:.2f}$ nm)",
        linewidth=2.5,
    )
    ax.fill_between(r_nm, lo_1, hi_1, alpha=0.25)

    ax.plot(
        r_nm,
        res_3kbar["mean_P"],
        label=rf"3 kbar ($\langle r \rangle={mean_r_3:.2f}$ nm)",
        linewidth=2.5,
    )
    ax.fill_between(r_nm, lo_3, hi_3, alpha=0.25)

    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel(r"$P(r)$")
    ax.set_xlim(1.5, 6.0)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    fig.savefig(base / "stride_1/GTN_sample_5000_trajectory_average.png", dpi=600, bbox_inches="tight")

    # Save per-frame summary
    summary_1 = pd.DataFrame({
        "condition": "1bar",
        "frame": res_1bar["frame_ids"],
        "time_ps": res_1bar["times_ps"],
        "mean_distance_nm": res_1bar["mean_r_frames"],
    })
    summary_3 = pd.DataFrame({
        "condition": "3kbar",
        "frame": res_3kbar["frame_ids"],
        "time_ps": res_3kbar["times_ps"],
        "mean_distance_nm": res_3kbar["mean_r_frames"],
    })
    pd.concat([summary_1, summary_3], ignore_index=True).to_csv(
        base / "stride_1/GTN_sample_5000_per_frame_summary.csv", index=False
    )

    # Save averaged distributions
    dist_df = pd.DataFrame({
        "r_nm": r_nm,
        "P_1bar_mean": res_1bar["mean_P"],
        "P_1bar_ci_low": lo_1,
        "P_1bar_ci_high": hi_1,
        "P_3kbar_mean": res_3kbar["mean_P"],
        "P_3kbar_ci_low": lo_3,
        "P_3kbar_ci_high": hi_3,
    })
    dist_df.to_csv(base / "stride_1/GTN_sample_5000_trajectory_distributions.csv", index=False)


if __name__ == "__main__":
    main()