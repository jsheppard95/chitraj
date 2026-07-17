"""
Estimate the frame-to-frame autocorrelation time of the raw MD
residue-residue distance (res406-res537), to pick a defensible
--block-size for the block bootstrap CIs in combine_chilife_chunks.py.

Why the raw MD distance and not ChiLife's P(r)/mean_r_frames:
ChiLife regenerates an independent spin-label rotamer ensemble at every
frame from that frame's backbone geometry alone. Any frame-to-frame
correlation in its output is therefore inherited from the underlying
protein/backbone dynamics, plus extra high-frequency noise from ChiLife's
own finite rotamer sampling. The raw MD distance (gmx distance output) is
the direct, unconfounded measurement of that physical decorrelation time.

Assumes ChiLife was run at MD stride 1 (one ChiLife frame per MD frame,
e.g. the tip4p/stride_1/... outputs), so a block size in "frames" here
matches the frame units of combine_chilife_chunks.py's --block-size
directly. If ChiLife was run at a different stride, divide the
recommended block size by that stride.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

MD_DIST_1BAR = BASE_DIR / "tip4p/res406_res537_1bar_dist.xvg"
MD_DIST_3KBAR = BASE_DIR / "tip4p/res406_res537_3kbar_dist.xvg"

DEFAULT_OUTDIR = BASE_DIR / "tip4p/stride_1"

COLOR_1BAR = "#457B9D"
COLOR_3KBAR = "#E63946"


def load_gmx_distance_xvg(xvg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a standard gmx distance XVG file.
    Assumes 2-column data: time(ps)   distance(nm)
    Ignores lines starting with # or @.
    Returns (times_ps, distances_nm).
    """
    times = []
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
                t_ps = float(parts[0])
                dist_nm = float(parts[1])
            except ValueError:
                continue

            times.append(t_ps)
            distances.append(dist_nm)

    times = np.asarray(times, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        raise ValueError(f"No numeric distance data read from {xvg_path}")

    return times, distances


def acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation function via FFT, acf[0] == 1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()

    size = 1
    while size < 2 * n:
        size *= 2

    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[: max_lag + 1]
    acov /= n
    return acov / acov[0]


def integrated_autocorr_time(acf: np.ndarray, n_samples: int) -> tuple[float, int]:
    """
    Geyer's initial positive sequence estimator with Chodera's finite-sample
    (1 - t/n) correction: sum the ACF up to (but not including) the first
    lag where it crosses zero, to avoid integrating long-lag noise.
    Returns (tau, cutoff_lag) with tau in units of frames.
    """
    tau = 1.0
    cutoff = len(acf) - 1
    for t in range(1, len(acf)):
        if acf[t] <= 0:
            cutoff = t
            break
        tau += 2.0 * (1.0 - t / n_samples) * acf[t]
    return tau, cutoff


def analyze_condition(
    label: str, xvg_path: Path, max_lag: int, safety_factor: float
) -> dict:
    times_ps, dist_nm = load_gmx_distance_xvg(xvg_path)
    n = len(dist_nm)

    dt_ps = float(np.median(np.diff(times_ps)))

    max_lag = min(max_lag, n - 2)
    acf = acf_fft(dist_nm, max_lag=max_lag)
    tau_frames, cutoff = integrated_autocorr_time(acf, n_samples=n)
    tau_ps = tau_frames * dt_ps

    block_size = int(np.ceil(safety_factor * tau_frames))
    n_eff = n / tau_frames

    print(f"[{label}] n_frames={n}  dt={dt_ps:.3f} ps")
    print(
        f"[{label}] integrated autocorrelation time = {tau_frames:.2f} frames "
        f"({tau_ps:.1f} ps), ACF crossed zero at lag {cutoff} frames"
    )
    print(f"[{label}] effective independent samples ~= {n_eff:.1f}")
    print(
        f"[{label}] recommended block_size = {safety_factor:g} x tau "
        f"= {block_size} frames"
    )

    return {
        "label": label,
        "times_ps": times_ps,
        "dist_nm": dist_nm,
        "dt_ps": dt_ps,
        "acf": acf,
        "tau_frames": tau_frames,
        "tau_ps": tau_ps,
        "cutoff": cutoff,
        "n_eff": n_eff,
        "block_size": block_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-dist-1bar", type=Path, default=MD_DIST_1BAR)
    parser.add_argument("--md-dist-3kbar", type=Path, default=MD_DIST_3KBAR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5000,
        help="Max lag (in frames) to compute/plot the ACF over.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=2.0,
        help="Recommended block_size = safety_factor * integrated autocorrelation time.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    result_1bar = analyze_condition(
        "1bar", args.md_dist_1bar, args.max_lag, args.safety_factor
    )
    result_3kbar = analyze_condition(
        "3kbar", args.md_dist_3kbar, args.max_lag, args.safety_factor
    )

    recommended_block_size = max(
        result_1bar["block_size"], result_3kbar["block_size"]
    )
    print(
        f"\nRecommended --block-size for combine_chilife_chunks.py "
        f"(conservative, max over conditions): {recommended_block_size} frames"
    )

    plot_lag = min(
        args.max_lag, max(result_1bar["cutoff"], result_3kbar["cutoff"]) * 4
    )
    plot_lag = max(plot_lag, 10)

    out_df = pd.DataFrame(
        {
            "lag_frames": np.arange(plot_lag + 1),
            "lag_ps_1bar": np.arange(plot_lag + 1) * result_1bar["dt_ps"],
            "acf_1bar": result_1bar["acf"][: plot_lag + 1],
            "lag_ps_3kbar": np.arange(plot_lag + 1) * result_3kbar["dt_ps"],
            "acf_3kbar": result_3kbar["acf"][: plot_lag + 1],
        }
    )
    out_df.to_csv(args.outdir / "md_distance_autocorrelation.csv", index=False)

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        for result, color in ((result_1bar, COLOR_1BAR), (result_3kbar, COLOR_3KBAR)):
            lag_ps = np.arange(plot_lag + 1) * result["dt_ps"]
            ax.plot(
                lag_ps,
                result["acf"][: plot_lag + 1],
                color=color,
                linewidth=2.0,
                label=rf"{result['label']} ($\tau={result['tau_ps']:.0f}$ ps)",
            )
            ax.axvline(
                result["cutoff"] * result["dt_ps"], color=color, linestyle=":", linewidth=1.0
            )

        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("Lag (ps)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("MD distance autocorrelation (res406-res537)")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)

        fig.tight_layout()
        fig.savefig(args.outdir / "md_distance_autocorrelation.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to: {args.outdir / 'md_distance_autocorrelation.png'}")

    print(f"Saved ACF CSV to: {args.outdir / 'md_distance_autocorrelation.csv'}")


if __name__ == "__main__":
    main()
