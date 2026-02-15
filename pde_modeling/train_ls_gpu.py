import re, time
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.ndimage import distance_transform_edt
import optuna

from ls_pde_gpu import evolve_level_set_gpu


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------- CONFIG ----------------
TUMOR_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/tumor_seg_nii")
BRAIN_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/brain_masks")


# Regex to parse tumor/brain filenames
MASK_PAT = re.compile(r"^(?P<patient>.+?)_pair(?P<pair>\d+)_t(?P<t>[01])\.nii\.gz$")

# ----------------------------------------

def load_bin(path: Path):
    """Load NIfTI mask as binary array with spacing and affine."""
    img = nib.load(str(path))
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    spacing = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
    return data, spacing, img.affine

def dice(a,b):
    """Dice score between two binary masks."""
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum()
    return float(2*inter / (a.sum() + b.sum() + 1e-8))

def hd95_voxel(a,b):
    """95th percentile Hausdorff distance (approx, voxel units)."""
    if a.sum()==0 and b.sum()==0: return 0.0
    if a.sum()==0 or b.sum()==0:  return 50.0
    da = distance_transform_edt(1-a)
    db = distance_transform_edt(1-b)
    sa = (a.astype(bool) & (distance_transform_edt(a)==0))
    sb = (b.astype(bool) & (distance_transform_edt(b)==0))
    d_ab = da[sb]; d_ba = db[sa]
    if d_ab.size==0 or d_ba.size==0: return 50.0
    return float(np.percentile(np.concatenate([d_ab,d_ba]), 95))

def find_cases():
    """Match tumor t0/t1 masks with corresponding brain masks by identical filenames."""
    cases = []
    for t0_path in TUMOR_DIR.glob("*_t0.nii.gz"):
        name = t0_path.name
        t1_path = TUMOR_DIR / name.replace("_t0.nii.gz", "_t1.nii.gz")
        if not t1_path.exists():
            continue
        # brain masks have same name under BRAIN_DIR
        brain_t0 = BRAIN_DIR / name
        brain_t1 = BRAIN_DIR / name.replace("_t0.nii.gz", "_t1.nii.gz")
        if not (brain_t0.exists() and brain_t1.exists()):
            print(f"[WARN] Missing brain mask for {name}, skip")
            continue
        # optional Δt
        df_info = pd.read_csv('/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/MRI_Info_clean.csv')
        # dt_path = TUMOR_DIR / name.replace("_t0.nii.gz", "_dt.txt")
        # delta_t = None
        # if dt_path.exists():
        #     try:
        #         delta_t = float(dt_path.read_text().strip())
        #     except Exception:
        #         pass

        try:
            delta_t = float(df_info.loc[df_info['pair'] == name.split("_t0.nii.gz")[0], 'time_gap (days)'])
        except:
            pass

        cases.append(dict(
            t0=t0_path, t1=t1_path,
            brain_t0=brain_t0, brain_t1=brain_t1,
            delta_t=delta_t, name=name.replace("_t0.nii.gz","")
        ))
    return cases

def main():
    PRED_DIR = Path("/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/predictions_test")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    cases = find_cases()
    print(f"Found {len(cases)} cases")
    assert cases, "No valid (t0,t1,brain) triplets found"

    TEST_NAMES = {
        "patient_25_pair1",
        "patient_31_pair1",
    }

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
        mask = (mask > 0).astype(np.uint8)
        dist_out = distance_transform_edt(mask == 0, sampling=spacing)
        dist_in = distance_transform_edt(mask == 1, sampling=spacing)
        return (dist_out - dist_in).astype(np.float32)

    def brain_distance_to_boundary(brain: np.ndarray, spacing):
        brain = (brain > 0).astype(np.uint8)
        d_in = distance_transform_edt(brain == 1, sampling=spacing)
        d_out = distance_transform_edt(brain == 0, sampling=spacing)
        d = np.where(brain == 1, d_in, d_out)
        return d.astype(np.float32)

    def normals_from_sdf(phi_brain: np.ndarray, spacing):
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
                # if you accidentally got a series/array, try to coerce
                try:
                    delta_t = float(np.asarray(delta_t).ravel()[0])
                except Exception:
                    delta_t = None

            out.append(dict(
                name=C["name"],
                S0=S0, S1=S1, brain=brain0,
                spacing=sp0, affine=affine,
                delta_t=delta_t,
                phi0=phi0, d_brain=d_brain,
                nx=nx, ny=ny, nz=nz
            ))
        return out

    # -----------------------
    # Build TRAIN dataset
    # -----------------------
    dataset = build_dataset(train_cases)

    # Loss weights
    w_dice, w_hd = 0.6, 0.4
    t_final_fallback = 30.0

    def objective(theta):
        v0, alpha, sigma_mm, a_par, a_perp, smooth_nrm, t_scale = theta

        # Sanity checks
        if not (0.0 < v0 < 5.0): return 1e3
        if not (0.0 < alpha < 5.0): return 1e3
        if not (0.5 <= sigma_mm <= 20.0): return 1e3
        if not (a_perp > 0 and a_par >= a_perp and a_par <= 5.0): return 1e3
        if not (0.0 <= smooth_nrm <= 3.0): return 1e3
        if not (0.2 <= t_scale <= 5.0): return 1e3

        Ls = []
        for C in dataset:
            dt = C["delta_t"]
            t_final = (t_scale * dt) if (dt is not None) else t_final_fallback

            _, S1_pred = evolve_level_set_gpu(
                phi0_cpu=C["phi0"],
                d_brain_cpu=C["d_brain"],
                nx_cpu=C["nx"], ny_cpu=C["ny"], nz_cpu=C["nz"],
                spacing=C["spacing"],
                t_final=float(t_final),
                v0=v0, alpha=alpha, sigma_mm=sigma_mm,
                a_parallel=a_par, a_perp=a_perp,
                reinit_every=10,
            )

            d = dice(S1_pred, C["S1"])
            hd = hd95_voxel(S1_pred, C["S1"])
            L = w_dice * (1.0 - d) + w_hd * (min(hd, 50.0) / 50.0)
            Ls.append(L)

        return float(np.mean(Ls)) if Ls else 1e3

    def objective_optuna(trial: optuna.Trial):
        v0 = trial.suggest_float("v0", 0.01, 1.00)
        alpha = trial.suggest_float("alpha", 0.01, 2.00)
        sigma_mm = trial.suggest_float("sigma_mm", 1.0, 12.0)
        a_par = trial.suggest_float("a_parallel", 0.5, 3.0)
        a_perp = trial.suggest_float("a_perp", 0.05, 1.0)
        smooth_nrm = trial.suggest_float("smooth_normals_sigma", 0.0, 2.0)
        t_scale = trial.suggest_float("t_scale", 0.5, 2.0)

        if a_par < a_perp:
            return 1e3

        theta = (v0, alpha, sigma_mm, a_par, a_perp, smooth_nrm, t_scale)
        return objective(theta)

    # -----------------------
    # Optimize (TRAIN)
    # -----------------------
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
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
            nx_cpu=C["nx"], ny_cpu=C["ny"], nz_cpu=C["nz"],
            spacing=C["spacing"],
            t_final=float(t_final),
            v0=theta_best[0],
            alpha=theta_best[1],
            sigma_mm=theta_best[2],
            a_parallel=theta_best[3],
            a_perp=theta_best[4],
            reinit_every=10,
        )

        # save mask
        pred_img = nib.Nifti1Image(S1_pred.astype(np.uint8), C["affine"])
        save_path = PRED_DIR / f"{C['name']}_pred_t1.nii.gz"
        nib.save(pred_img, str(save_path))
        print(f"Saved prediction: {save_path}")

        d = dice(S1_pred, C["S1"])
        hd = hd95_voxel(S1_pred, C["S1"])
        L = w_dice * (1.0 - d) + w_hd * (min(hd, 50.0) / 50.0)
        test_losses.append(L)
        print(f"Case {C['name']} - Dice: {d:.4f}, HD95: {hd:.2f}, Loss: {L:.4f}")

    print("TEST loss:", float(np.mean(test_losses)))



if __name__ == "__main__":
    main()
