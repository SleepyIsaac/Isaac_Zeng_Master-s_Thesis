import re
import time
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt
import optuna

from ls_pde_gpu import evolve_level_set_gpu

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# ============================== CONFIG ==============================
TUMOR_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/tumor_seg")
BRAIN_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/brain_seg")

# Test split (names without "_t0.nii.gz")
TEST_NAMES = {
    "patient_25_pair1",
    "patient_31_pair1",
}

# Where to save best-test predictions (same as your original)
PRED_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/predictions_test")

# NEW: Where to save *every* Optuna trial (params + metrics; optionally masks)
TRIALS_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/optuna_trials_train")
TRIALS_DIR.mkdir(parents=True, exist_ok=True)

# If True, save predicted TRAIN masks for every trial and every case (can explode disk!)
SAVE_TRAIN_MASKS = False

# If True, save predicted TRAIN masks only for the current "best so far" trial (safer)
SAVE_BEST_SO_FAR_MASKS = True

# Summary CSV (one row per trial) + per-trial per-case CSV
SUMMARY_CSV = TRIALS_DIR / "trials_summary.csv"

# Regex (not strictly needed, kept from your original)
MASK_PAT = re.compile(r"^(?P<patient>.+?)_pair(?P<pair>\d+)_t(?P<t>[01])\.nii\.gz$")
# ===================================================================


def load_bin(path: Path):
    """Load NIfTI mask as binary array with spacing and affine."""
    img = nib.load(str(path))
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    spacing = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
    return data, spacing, img.affine


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice score between two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    return float(2 * inter / (a.sum() + b.sum() + 1e-8))


def hd95_voxel(a: np.ndarray, b: np.ndarray) -> float:
    """
    Approximate 95th percentile Hausdorff distance in voxel units.
    NOTE: This is a rough implementation; you can replace with a more standard surface distance.
    """
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return 50.0

    # Distance-to-background maps
    da = distance_transform_edt(1 - a)
    db = distance_transform_edt(1 - b)

    # "Surface" approximations (kept as-is from your original logic)
    sa = (a.astype(bool) & (distance_transform_edt(a) == 0))
    sb = (b.astype(bool) & (distance_transform_edt(b) == 0))

    d_ab = da[sb]
    d_ba = db[sa]
    if d_ab.size == 0 or d_ba.size == 0:
        return 50.0
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def find_cases():
    """Match tumor t0/t1 masks with corresponding brain masks by identical filenames."""
    cases = []
    for t0_path in TUMOR_DIR.glob("*_t0.nii.gz"):
        name = t0_path.name
        t1_path = TUMOR_DIR / name.replace("_t0.nii.gz", "_t1.nii.gz")
        if not t1_path.exists():
            continue

        # Brain masks have same name under BRAIN_DIR
        brain_t0 = BRAIN_DIR / name
        brain_t1 = BRAIN_DIR / name.replace("_t0.nii.gz", "_t1.nii.gz")
        if not (brain_t0.exists() and brain_t1.exists()):
            print(f"[WARN] Missing brain mask for {name}, skip")
            continue

        # Optional delta_t: you currently hardcode 30 in your pasted code
        # Replace this with reading your CSV if you want true delta_t per pair.
        delta_t = 30.0

        cases.append(
            dict(
                t0=t0_path,
                t1=t1_path,
                brain_t0=brain_t0,
                brain_t1=brain_t1,
                delta_t=delta_t,
                name=name.replace("_t0.nii.gz", ""),
            )
        )
    return cases


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    cases = find_cases()
    print(f"Found {len(cases)} cases")
    assert cases, "No valid (t0,t1,brain) triplets found"

    # Split train / test
    train_cases, test_cases = [], []
    for C in cases:
        if C["name"] in TEST_NAMES:
            test_cases.append(C)
        else:
            train_cases.append(C)

    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases : {len(test_cases)}")

    # -----------------------
    # Helper builders
    # -----------------------
    def sdf_from_mask(mask: np.ndarray, spacing):
        """
        Signed distance function (outside positive, inside negative).
        """
        mask = (mask > 0).astype(np.uint8)
        dist_out = distance_transform_edt(mask == 0, sampling=spacing)
        dist_in = distance_transform_edt(mask == 1, sampling=spacing)
        return (dist_out - dist_in).astype(np.float32)

    def brain_distance_to_boundary(brain: np.ndarray, spacing):
        """
        Distance to nearest brain boundary (mm), using inside/outside EDT.
        """
        brain = (brain > 0).astype(np.uint8)
        d_in = distance_transform_edt(brain == 1, sampling=spacing)
        d_out = distance_transform_edt(brain == 0, sampling=spacing)
        d = np.where(brain == 1, d_in, d_out)
        return d.astype(np.float32)

    def normals_from_sdf(phi_brain: np.ndarray, spacing):
        """
        Approximate normals by central-difference-like gradient and normalize.
        """
        def fwd(a, ax, h): return (np.roll(a, -1, axis=ax) - a) / h
        def bwd(a, ax, h): return (a - np.roll(a, 1, axis=ax)) / h

        grads = []
        for ax, h in enumerate(spacing):
            g = fwd(phi_brain, ax, h) + bwd(phi_brain, ax, h)
            grads.append(g.astype(np.float32))

        gx, gy, gz = (grads + [np.zeros_like(phi_brain, dtype=np.float32)] * (3 - len(grads)))[:3]
        norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        return (gx / norm).astype(np.float32), (gy / norm).astype(np.float32), (gz / norm).astype(np.float32)

    def build_dataset(case_list):
        """
        Precompute everything needed for the level set forward so objective() is fast.
        """
        out = []
        for C in case_list:
            S0, sp0, affine = load_bin(C["t0"])
            S1, sp1, _ = load_bin(C["t1"])
            brain0, _, _ = load_bin(C["brain_t0"])

            if S0.shape != S1.shape or S0.shape != brain0.shape:
                print(f"[WARN] Shape mismatch for {C['name']}, skip")
                continue

            phi0 = sdf_from_mask(S0, sp0)
            phi_brain = sdf_from_mask(brain0, sp0)
            d_brain = brain_distance_to_boundary(brain0, sp0)
            nx, ny, nz = normals_from_sdf(phi_brain, sp0)

            # delta_t robust: if missing/unset, store None
            delta_t = C.get("delta_t", None)
            if isinstance(delta_t, (np.ndarray, list)):
                try:
                    delta_t = float(np.asarray(delta_t).ravel()[0])
                except Exception:
                    delta_t = None

            out.append(
                dict(
                    name=C["name"],
                    S0=S0,
                    S1=S1,
                    brain=brain0,
                    spacing=sp0,
                    affine=affine,
                    delta_t=delta_t,
                    phi0=phi0,
                    d_brain=d_brain,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                )
            )
        return out

    # -----------------------
    # Build TRAIN dataset
    # -----------------------
    dataset = build_dataset(train_cases)

    # Loss weights
    w_dice, w_hd = 0.6, 0.4
    t_final_fallback = 30.0

    # Track best-so-far within this Python process (useful for saving only best masks)
    best_so_far = {"loss": float("inf"), "trial": None}

    def objective_with_details(theta, trial_number: int):
        """
        Run forward on TRAIN dataset and return:
          - mean loss
          - per-case metrics list (dict rows)
        """
        v0, alpha, sigma_mm, a_par, a_perp, smooth_nrm, t_scale = theta

        # Sanity checks
        if not (0.0 < v0 < 5.0):
            return 1e3, []
        if not (0.0 < alpha < 5.0):
            return 1e3, []
        if not (0.5 <= sigma_mm <= 20.0):
            return 1e3, []
        if not (a_perp > 0 and a_par >= a_perp and a_par <= 5.0):
            return 1e3, []
        if not (0.0 <= smooth_nrm <= 3.0):
            return 1e3, []
        if not (0.2 <= t_scale <= 5.0):
            return 1e3, []

        rows = []
        losses = []

        for C in dataset:
            dt = C["delta_t"]
            t_final = (t_scale * dt) if (dt is not None) else t_final_fallback

            # Forward (GPU)
            _, S1_pred = evolve_level_set_gpu(
                phi0_cpu=C["phi0"],
                d_brain_cpu=C["d_brain"],
                nx_cpu=C["nx"],
                ny_cpu=C["ny"],
                nz_cpu=C["nz"],
                spacing=C["spacing"],
                t_final=float(t_final),
                v0=v0,
                alpha=alpha,
                sigma_mm=sigma_mm,
                a_parallel=a_par,
                a_perp=a_perp,
                reinit_every=50,
            )

            dsc = dice(S1_pred, C["S1"])
            hd = hd95_voxel(S1_pred, C["S1"])
            L = w_dice * (1.0 - dsc) + w_hd * (min(hd, 50.0) / 50.0)

            losses.append(L)
            rows.append(
                dict(
                    trial=trial_number,
                    case=C["name"],
                    dice=float(dsc),
                    hd95=float(hd),
                    loss=float(L),
                    t_final=float(t_final),
                )
            )

        mean_loss = float(np.mean(losses)) if losses else 1e3
        return mean_loss, rows

    def save_trial_artifacts(
        trial_number: int,
        mean_loss: float,
        params: dict,
        rows: list[dict],
        save_masks: bool = False,
        save_best_so_far_masks: bool = False,
    ):
        """
        Save params + metrics for this trial. Optionally save TRAIN masks.
        """
        trial_dir = TRIALS_DIR / f"trial_{trial_number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1) meta.json
        meta = {
            "trial": int(trial_number),
            "mean_loss": float(mean_loss),
            "params": {k: float(v) for k, v in params.items()},
        }
        (trial_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # 2) Per-case metrics CSV
        if rows:
            pd.DataFrame(rows).to_csv(trial_dir / "train_case_metrics.csv", index=False)

        # 3) Summary CSV (one row per trial)
        # NOTE: Safe if n_jobs=1. If you use parallel Optuna (n_jobs>1), use a file lock.
        summary_row = {
            "trial": int(trial_number),
            "mean_loss": float(mean_loss),
            **{k: float(v) for k, v in params.items()},
        }
        df_row = pd.DataFrame([summary_row])
        if SUMMARY_CSV.exists():
            df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
        else:
            df_row.to_csv(SUMMARY_CSV, index=False)

        # 4) Optional: save TRAIN masks for this trial (huge disk if enabled)
        if not (save_masks or save_best_so_far_masks):
            return

        # If saving only best-so-far masks, check condition
        if save_best_so_far_masks:
            if mean_loss >= best_so_far["loss"]:
                return  # not improving

        # If we get here, we will save masks for this trial
        masks_dir = trial_dir / "train_pred_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Re-run forward to save masks (to avoid holding huge arrays in memory).
        # This doubles compute for saved trials, but keeps memory simple and avoids storing per-case preds in objective.
        v0 = params["v0"]
        alpha = params["alpha"]
        sigma_mm = params["sigma_mm"]
        a_par = params["a_parallel"]
        a_perp = params["a_perp"]
        t_scale = params["t_scale"]

        for C in dataset:
            dt = C["delta_t"]
            t_final = (t_scale * dt) if (dt is not None) else t_final_fallback
            _, S1_pred = evolve_level_set_gpu(
                phi0_cpu=C["phi0"],
                d_brain_cpu=C["d_brain"],
                nx_cpu=C["nx"],
                ny_cpu=C["ny"],
                nz_cpu=C["nz"],
                spacing=C["spacing"],
                t_final=float(t_final),
                v0=v0,
                alpha=alpha,
                sigma_mm=sigma_mm,
                a_parallel=a_par,
                a_perp=a_perp,
                reinit_every=50,
            )
            pred_img = nib.Nifti1Image(S1_pred.astype(np.uint8), C["affine"])
            nib.save(pred_img, str(masks_dir / f"{C['name']}_pred_t1.nii.gz"))

    def objective_optuna(trial: optuna.Trial):
        # Suggest parameters
        v0 = trial.suggest_float("v0", 0.3, 1.00)
        alpha = trial.suggest_float("alpha", 0.01, 2.00)
        sigma_mm = trial.suggest_float("sigma_mm", 1.0, 12.0)
        a_par = trial.suggest_float("a_parallel", 0.5, 3.0)
        a_perp = trial.suggest_float("a_perp", 0.05, 1.0)
        smooth_nrm = trial.suggest_float("smooth_normals_sigma", 0.0, 2.0)
        t_scale = trial.suggest_float("t_scale", 0.5, 2.0)

        if a_par < a_perp:
            # Invalid anisotropy setting
            mean_loss = 1e3
            rows = []
        else:
            theta = (v0, alpha, sigma_mm, a_par, a_perp, smooth_nrm, t_scale)
            mean_loss, rows = objective_with_details(theta, trial_number=trial.number)

        # Save everything for this trial (params + metrics; optional masks)
        params = {
            "v0": v0,
            "alpha": alpha,
            "sigma_mm": sigma_mm,
            "a_parallel": a_par,
            "a_perp": a_perp,
            "smooth_normals_sigma": smooth_nrm,
            "t_scale": t_scale,
        }

        # Update best-so-far in this process
        if mean_loss < best_so_far["loss"]:
            best_so_far["loss"] = mean_loss
            best_so_far["trial"] = trial.number

        save_trial_artifacts(
            trial_number=trial.number,
            mean_loss=float(mean_loss),
            params=params,
            rows=rows,
            save_masks=SAVE_TRAIN_MASKS,
            save_best_so_far_masks=SAVE_BEST_SO_FAR_MASKS,
        )

        return float(mean_loss)

    # -----------------------
    # Optimize (TRAIN)
    # -----------------------
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # IMPORTANT: If you plan to run Optuna in parallel (n_jobs > 1),
    # you must use a file lock for SUMMARY_CSV appends.
    study.optimize(objective_optuna, n_trials=200, show_progress_bar=True)

    best = study.best_trial
    print("Best θ:", best.params, " loss=", best.value, " time(s)=", round(time.time() - t0, 2))

    theta_best = (
        best.params["v0"],
        best.params["alpha"],
        best.params["sigma_mm"],
        best.params["a_parallel"],
        best.params["a_perp"],
        best.params["smooth_normals_sigma"],
        best.params["t_scale"],
    )

    # -----------------------
    # Evaluate + Save (TEST)
    # -----------------------
    print("\nEvaluating on TEST set...")
    test_dataset = build_dataset(test_cases)

    if not test_dataset:
        print("[WARN] No valid test cases after filtering (shape mismatch etc.)")
        return

    test_losses = []
    for C in test_dataset:
        dt = C["delta_t"]
        t_final = (theta_best[6] * dt) if (dt is not None) else t_final_fallback

        _, S1_pred = evolve_level_set_gpu(
            phi0_cpu=C["phi0"],
            d_brain_cpu=C["d_brain"],
            nx_cpu=C["nx"],
            ny_cpu=C["ny"],
            nz_cpu=C["nz"],
            spacing=C["spacing"],
            t_final=float(t_final),
            v0=theta_best[0],
            alpha=theta_best[1],
            sigma_mm=theta_best[2],
            a_parallel=theta_best[3],
            a_perp=theta_best[4],
            reinit_every=50,
        )

        # Save test prediction mask
        pred_img = nib.Nifti1Image(S1_pred.astype(np.uint8), C["affine"])
        save_path = PRED_DIR / f"{C['name']}_pred_t1_2.nii.gz"
        nib.save(pred_img, str(save_path))
        print(f"Saved prediction: {save_path}")

        dsc = dice(S1_pred, C["S1"])
        hd = hd95_voxel(S1_pred, C["S1"])
        L = w_dice * (1.0 - dsc) + w_hd * (min(hd, 50.0) / 50.0)
        test_losses.append(L)

        print(f"Case {C['name']} - Dice: {dsc:.4f}, HD95: {hd:.2f}, Loss: {L:.4f}")

    print("TEST loss:", float(np.mean(test_losses)))


if __name__ == "__main__":
    main()
