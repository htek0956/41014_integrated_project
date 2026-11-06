"""Plot an EKF observing a true-state-controlled run, or replot a saved NPZ.

Errors use truth minus estimate, as in Assessment 3 Checkpoint B. Camera
observations are fused once per image; truth is used here only for evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from models import wrap


RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "figures" / "estimator"


def estimator_diagnostics(log):
    """Calculate errors, marginal 2-sigma bounds and consistency statistics."""
    time = np.asarray(log["time"], dtype=float)
    truth = np.asarray(log["truth"], dtype=float)
    estimate = np.asarray(log["estimate"], dtype=float)
    covariance = np.asarray(log["covariance"], dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("There are no estimator samples to plot.")
    if truth.shape != (len(time), 3) or estimate.shape != truth.shape:
        raise ValueError("Truth and estimate must each contain one [x, y, heading] per time.")
    if covariance.shape != (len(time), 3, 3):
        raise ValueError("Covariance must have shape (samples, 3, 3).")
    if not all(np.all(np.isfinite(a)) for a in (time, truth, estimate, covariance)):
        raise ValueError("Estimator state, truth, time and covariance must be finite.")

    error = truth - estimate
    error[:, 2] = wrap(error[:, 2])
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    sigma = np.full_like(diagonal, np.nan)
    np.sqrt(diagonal, out=sigma, where=diagonal >= 0)
    symmetric = np.all(np.isclose(covariance, covariance.transpose(0, 2, 1),
                                  atol=1e-10, rtol=0), axis=(1, 2))
    positive = np.linalg.eigvalsh((covariance + covariance.transpose(0, 2, 1))/2)[:, 0] > 0
    valid_covariance = symmetric & positive

    # Do not hide an invalid covariance by clipping variances or using a pseudoinverse.
    nees = np.full(len(time), np.nan)
    if np.any(valid_covariance):
        e, P = error[valid_covariance], covariance[valid_covariance]
        solved = np.linalg.solve(P, e[..., None])[..., 0]
        nees[valid_covariance] = np.einsum("ni,ni->n", e, solved)

    return dict(time=time, truth=truth, estimate=estimate, error=error,
                sigma=sigma, nees=nees, valid_covariance=valid_covariance)


def estimator_metrics(log, diagnostics=None):
    d = estimator_diagnostics(log) if diagnostics is None else diagnostics
    error = d["error"]
    position_error = np.linalg.norm(error[:, :2], axis=1)
    nis = np.asarray(log["nis"], dtype=float)
    finite_nis = nis[np.isfinite(nis)]
    finite_nees = d["nees"][np.isfinite(d["nees"])]
    updated = np.asarray(log["camera_updated"], dtype=bool)
    available = np.asarray(log["camera_valid"], dtype=bool)
    return {
        "samples": len(error),
        "duration_s": float(d["time"][-1] - d["time"][0]),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_max_error_m": float(position_error.max()),
        "position_final_error_m": float(position_error[-1]),
        "component_rmse_m_m_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "component_bias_m_m_rad": error.mean(axis=0).tolist(),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 2]**2)))),
        "heading_max_abs_error_deg": float(np.rad2deg(np.max(np.abs(error[:, 2])))),
        "fraction_inside_2sigma_x_y_heading": np.mean(
            np.abs(error) <= 2*d["sigma"], axis=0).tolist(),
        "invalid_covariance_samples": int(np.count_nonzero(~d["valid_covariance"])),
        "camera_frames": int(updated.sum()),
        "valid_camera_frames": int(np.count_nonzero(updated & available)),
        "camera_valid_fraction": (float(available[updated].mean()) if updated.any() else None),
        "mean_nees": float(finite_nees.mean()) if finite_nees.size else None,
        "mean_nis": float(finite_nis.mean()) if finite_nis.size else None,
        "consistency_note": "Single-run diagnostics, not proof of consistency; Q/R require validation.",
    }


def plot_estimator(log):
    """Return trajectory, Checkpoint-B errors, and consistency/sensor figures."""
    import matplotlib.pyplot as plt

    d = estimator_diagnostics(log)
    metrics = estimator_metrics(log, d)
    t = d["time"]
    camera = np.asarray(log["camera_pose"], dtype=float)
    available = np.asarray(log["camera_valid"], dtype=bool)

    trajectory, ax = plt.subplots(figsize=(9, 8))
    ax.plot(*d["truth"][:, :2].T, color="black", lw=2, label="Ground truth")
    ax.plot(*d["estimate"][:, :2].T, color="tab:blue", lw=1.2, label="EKF estimate")
    valid_camera = np.all(np.isfinite(camera), axis=1)
    ax.scatter(*camera[valid_camera, :2].T, s=12, color="tab:green", alpha=.5,
               label="Camera pose (new valid frames)")
    ax.scatter(*d["truth"][0, :2], marker="o", color="black", label="Truth start")
    ax.scatter(*d["truth"][-1, :2], marker="x", color="black", label="Truth end")
    ax.set(xlabel="World x [m]", ylabel="World y [m]",
           title=f"EKF vs ground-truth route | position RMSE {metrics['position_rmse_m']:.3f} m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls="--", alpha=.5)
    ax.legend()
    trajectory.tight_layout()

    # Match Assessment 3 Checkpoint B: cm, cm, degrees and shaded +/-2 sigma.
    errors, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (label, scale) in enumerate((("x error [cm]", 100),
                                      ("y error [cm]", 100),
                                      ("heading error [deg]", 180/np.pi))):
        ax = axes[i]
        bound = 2*scale*d["sigma"][:, i]
        ax.fill_between(t, -bound, bound, color="tab:blue", alpha=.2, label=r"$\pm 2\sigma$")
        ax.plot(t, scale*d["error"][:, i], color="tab:blue", lw=1, label="Truth - estimate")
        ax.axhline(0, color="black", lw=.6)
        ax.fill_between(t, 0, 1, where=~available, transform=ax.get_xaxis_transform(),
                        color=".5", alpha=.2, step="post", label="No valid camera pose")
        coverage = metrics["fraction_inside_2sigma_x_y_heading"][i]
        ax.text(.01, .96, f"Inside 2 sigma: {coverage:.1%}", transform=ax.transAxes, va="top")
        ax.set_ylabel(label)
        ax.grid(True, ls="--", alpha=.5)
    axes[0].set_title("EKF pose errors\nErrors = truth - estimate")
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Simulation time [s]")
    errors.tight_layout()

    consistency, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, d["nees"], lw=1, label="NEES (3-state pose)")
    axes[0].axhline(7.815, color="tab:red", ls="--", label="95% upper reference (3 dof)")
    axes[0].set_ylabel("NEES")
    nis = np.asarray(log["nis"], dtype=float)
    mask = np.isfinite(nis)
    axes[1].plot(t[mask], nis[mask], ".", label="NIS (new valid camera frames)")
    axes[1].axhline(7.815, color="tab:red", ls="--", label="95% upper reference (3 dof)")
    axes[1].set_ylabel("NIS")
    axes[2].step(t, log["tag_count"], where="post", label="Tags used in latest image")
    axes[2].fill_between(t, 0, 1, where=~available, transform=axes[2].get_xaxis_transform(),
                         color=".5", alpha=.2, step="post", label="No valid camera pose")
    axes[2].set(ylabel="Tag count", xlabel="Simulation time [s]")
    for ax in axes:
        ax.grid(True, ls="--", alpha=.5)
        ax.legend(loc="upper right")
    zero_q = not np.any(log.get("settings", {}).get("process_covariance_per_prediction", [0]))
    axes[0].set_title("Estimator consistency and camera availability"
                      + (" — Q = 0 (current baseline)" if zero_q else ""))
    consistency.tight_layout()
    return (trajectory, errors, consistency), metrics


def save_estimator_results(log, show=False, results_folder=None, run_name=None,
                           output_folder=None, close=True):
    import matplotlib.pyplot as plt

    figures, metrics = plot_estimator(log)
    root = RESULTS_FOLDER if results_folder is None else Path(results_folder)
    if output_folder is None:
        root.mkdir(parents=True, exist_ok=True)
    if output_folder is not None:
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "full_system"
    elif run_name is not None:
        # A named run replaces its previous outputs when the experiment is rerun.
        folder = root / run_name
        folder.mkdir(exist_ok=True)
        suffix = run_name
    else:
        index = 1 + max((int(p.name[4:]) for p in root.glob("run_*")
                         if p.is_dir() and p.name[4:].isdigit()), default=0)
        while True:
            folder = root / f"run_{index:03d}"
            try:
                folder.mkdir()
                break
            except FileExistsError:
                index += 1
        suffix = f"{index:03d}"
    for figure, name in zip(figures, ("estimator_trajectory", "estimator_errors", "estimator_consistency")):
        figure.savefig(folder / f"{name}_{suffix}.png", dpi=150)
    arrays = {key: np.asarray(value) for key, value in log.items() if key != "settings"}
    arrays["settings"] = np.array(json.dumps(log.get("settings", {})))
    np.savez_compressed(folder / f"estimator_data_{suffix}.npz", **arrays)
    metrics_name = "estimator_metrics" if output_folder is not None else "metrics"
    (folder / f"{metrics_name}_{suffix}.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved estimator results: {folder}")
    print(f"Estimator position RMSE: {metrics['position_rmse_m']:.3f} m; "
          f"heading RMSE: {metrics['heading_rmse_deg']:.2f} deg")
    if show:
        plt.show()
    if close:
        for figure in figures:
            plt.close(figure)
    return folder


def load_estimator_results(path):
    with np.load(path, allow_pickle=False) as saved:
        log = {key: saved[key].copy() for key in saved.files if key != "settings"}
        log["settings"] = json.loads(str(saved["settings"]))
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Replot an estimator_data_NNN.npz file.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening them.")
    parser.add_argument("--headless", action="store_true", help="Run the loop without simulation windows.")
    parser.add_argument("--max-time", type=float, help="Limit simulated seconds for a quick check.")
    args = parser.parse_args()
    if args.input:
        save_estimator_results(load_estimator_results(args.input), show=not args.no_show)
    else:
        from main import main
        main(show_viewer=not (args.headless or args.no_show), max_time=args.max_time,
             estimator_diagnostics=True, reports=("estimator",))
