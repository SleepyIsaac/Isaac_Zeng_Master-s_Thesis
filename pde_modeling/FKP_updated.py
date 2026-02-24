import os
import re
import numpy as np
import SimpleITK as itk

from scipy.linalg import eigh
from scipy.optimize import minimize
from joblib import Parallel, delayed


# =========================
# User configuration
# =========================
BRAIN_DIR = "/home/zengy2/isilon/Isaac/Clinic/MRI_data/Simulation/data_T1/registered/brain_seg"
TUMOR_DIR = "/home/zengy2/isilon/Isaac/Clinic/MRI_data/Simulation/data_T1/registered/tumor_seg"

TEST_SET = {"patient_25_pair1", "patient_31_pair1"}

DX = 1.0     # grid spacing used in Laplacian eigenvalues (set to your resampled spacing if needed)
DT = 1.0     # time step
STEPS = 90   # number of steps from t0 -> t1

N_JOBS = 16  # parallelism for loss evaluation
TUMOR_IS_PROB = False  # True if tumor NIfTI is already a probability map in [0,1]

# Output
OUT_DIR = "/home/zengy2/isilon/Isaac/Clinic/MRI_data/Simulation/data_T1/predictions_test_FKP"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# NIfTI I/O utilities
# =========================
def load_nii_xyz(path, dtype=np.float32):
    """
    Read NIfTI -> numpy array in (x, y, z) order.

    SimpleITK returns array in (z, y, x), we transpose to (x, y, z).
    """
    img = itk.ReadImage(path)
    arr_zyx = itk.GetArrayFromImage(img).astype(dtype)    # (z, y, x)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))            # -> (x, y, z)
    return arr_xyz


def binarize(x, thr=0.5):
    """Binarize a volume to {0,1} float32."""
    return (x > thr).astype(np.float32)


def save_pred_mask_as_nii(pred_xyz, reference_nii_path, out_path, threshold=0.5):
    """
    Save predicted volume (x,y,z) as NIfTI using reference header (spacing/origin/direction).

    pred_xyz: numpy array (x,y,z), can be continuous or binary
    threshold: if not None, save a binary mask (uint8) using pred_xyz > threshold
    """
    if threshold is None:
        pred_to_save_xyz = pred_xyz.astype(np.float32)
    else:
        pred_to_save_xyz = (pred_xyz > threshold).astype(np.uint8)

    # Convert back to (z,y,x) for SimpleITK
    pred_to_save_zyx = np.transpose(pred_to_save_xyz, (2, 1, 0))
    pred_img = itk.GetImageFromArray(pred_to_save_zyx)

    ref_img = itk.ReadImage(reference_nii_path)
    pred_img.SetSpacing(ref_img.GetSpacing())
    pred_img.SetOrigin(ref_img.GetOrigin())
    pred_img.SetDirection(ref_img.GetDirection())

    itk.WriteImage(pred_img, out_path)


# =========================
# Dataset builder for your naming scheme
# =========================
def build_dataset_from_registered(brain_dir, tumor_dir, tumor_is_prob=False):
    """
    Build paired datasets from tumor_dir containing *_t0 and *_t1 NIfTIs.
    Returns:
      list_c0:   (N, x, y, z) tumor at t0
      list_c1:   (N, x, y, z) tumor at t1
      list_brain:(N, x, y, z) brain mask
      case_ids:  (N,) string ids like "patient_25_pair1"
      ref_t1_paths: (N,) path to tumor t1 (used as NIfTI header reference for saving)
    """
    # Find all tumor t0 files
    tumor_t0_files = sorted([
        f for f in os.listdir(tumor_dir)
        if (f.endswith(".nii") or f.endswith(".nii.gz")) and re.search(r"_t0(\.|_)", f)
    ])

    def t0_to_t1(fname):
        # Replace "_t0." or "_t0_" with "_t1." / "_t1_"
        return re.sub(r"_t0(\.|_)", r"_t1\1", fname)

    def strip_case_id(fname):
        # patient_25_pair1_t0.nii.gz -> patient_25_pair1
        x = re.sub(r"(\.nii(\.gz)?)$", "", fname)
        x = re.sub(r"_(t0|t1)$", "", x)
        return x

    def find_brain_filename(tumor_filename):
        """
        brain mask file priority:
        1) exact same filename exists in brain_dir
        2) if brain_dir stores without t0/t1: patient_25_pair1.nii.gz
        """
        # 1) same name
        if os.path.exists(os.path.join(brain_dir, tumor_filename)):
            return tumor_filename

        # 2) remove _t0/_t1 but keep extension
        base = re.sub(r"^(.*)_(t0|t1)(\..+)$", r"\1\3", tumor_filename)  # patient_pair_t0.nii.gz -> patient_pair.nii.gz
        if os.path.exists(os.path.join(brain_dir, base)):
            return base

        return None

    list_c0, list_c1, list_brain, case_ids, ref_t1_paths = [], [], [], [], []
    skipped = 0

    for t0_name in tumor_t0_files:
        t1_name = t0_to_t1(t0_name)

        p_t0 = os.path.join(tumor_dir, t0_name)
        p_t1 = os.path.join(tumor_dir, t1_name)
        if not os.path.exists(p_t1):
            skipped += 1
            continue

        b0_name = find_brain_filename(t0_name)
        b1_name = find_brain_filename(t1_name)
        if b0_name is None or b1_name is None:
            skipped += 1
            continue

        # Load arrays in (x,y,z)
        brain0 = binarize(load_nii_xyz(os.path.join(brain_dir, b0_name)))
        brain1 = binarize(load_nii_xyz(os.path.join(brain_dir, b1_name)))

        c0 = load_nii_xyz(p_t0).astype(np.float32)
        c1 = load_nii_xyz(p_t1).astype(np.float32)

        if not tumor_is_prob:
            c0 = binarize(c0)
            c1 = binarize(c1)

        # Enforce tumor only inside brain
        c0 *= brain0
        c1 *= brain1

        # Basic shape check
        if c0.shape != c1.shape or c0.shape != brain0.shape:
            skipped += 1
            continue

        # Use intersection of brain masks (conservative)
        brain = (brain0 * brain1).astype(np.float32)

        list_c0.append(c0)
        list_c1.append(c1)
        list_brain.append(brain)
        case_ids.append(strip_case_id(t0_name))
        ref_t1_paths.append(p_t1)

    print(f"Loaded pairs: {len(list_c0)} | Skipped: {skipped}")
    return (
        np.array(list_c0, np.float32),
        np.array(list_c1, np.float32),
        np.array(list_brain, np.float32),
        np.array(case_ids),
        np.array(ref_t1_paths),
    )


# =========================
# Eigen-decomposition for 1D Laplacian
# =========================
def eig_1d_laplacian_matrix(n, dx):
    """
    Build 1D Laplacian matrix with second-order finite difference stencil,
    then compute its eigen-decomposition.

    Returns:
      Q: eigenvectors
      eigvals: eigenvalues (for L/dx^2)
    """
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)
    L = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = eigh(L / dx**2)
    return eigvecs, eigvals


# =========================
# Kronecker frequency transforms (no full Kronecker matrix)
# =========================
def forward_transform(c, Qx, Qy, Qz):
    """Compute c_hat = (Qx^T ⊗ Qy^T ⊗ Qz^T) c without building Kronecker matrix."""
    tmp = np.einsum('ia,abc->ibc', Qx.T, c)      # axis 0
    tmp = np.einsum('jb,ibc->ijc', Qy.T, tmp)    # axis 1
    c_hat = np.einsum('kc,ijc->ijk', Qz.T, tmp)  # axis 2
    return c_hat


def inverse_transform(c_hat, Qx, Qy, Qz):
    """Compute c = (Qx ⊗ Qy ⊗ Qz) c_hat without building Kronecker matrix."""
    tmp = np.einsum('ia,abc->ibc', Qx, c_hat)    # axis 0
    tmp = np.einsum('jb,ibc->ijc', Qy, tmp)      # axis 1
    c = np.einsum('kc,ijc->ijk', Qz, tmp)        # axis 2
    return c


# =========================
# Fisher-KPP reaction term
# =========================
def reaction(c, rho, K=1.0):
    """Logistic reaction term rho*c*(1 - c/K)."""
    reac = rho * c * (1 - c / K)
    reac = np.nan_to_num(reac, nan=0.0, posinf=0.0, neginf=0.0)
    return reac


# =========================
# Time stepping (your original structure + brain mask)
# =========================
def step_forward_separable(c0, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho, brain_mask=None):
    """
    One step update using separable spectral method + exponential integrator style update.

    NOTE: This follows your original implementation structure. Only change is:
      - Apply brain_mask at the end to force outside-brain values to zero.
    """
    c_hat = forward_transform(c0, Qx, Qy, Qz)
    Nc_hat = forward_transform(reaction(c0, rho), Qx, Qy, Qz)

    result_hat = np.zeros_like(c_hat)

    for i in range(c_hat.shape[0]):
        for j in range(c_hat.shape[1]):
            for k in range(c_hat.shape[2]):
                lam = Lx[i] + Ly[j] + Lz[k]
                e_term = np.exp(-dt * lam)
                # Keep the same branch structure as your original code
                phi1 = (e_term - 1) / (dt * lam) if abs(lam) > 1e-8 else dt
                result_hat[i, j, k] = e_term * c_hat[i, j, k] + phi1 * Nc_hat[i, j, k]

    c = inverse_transform(result_hat, Qx, Qy, Qz)
    c = np.clip(c, 0.0, 1.5)

    if brain_mask is not None:
        c *= brain_mask

    return c


def simulate_N_steps(c0, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho, steps, brain_mask=None, verbose=False):
    """Simulate multiple steps from t0 to t1."""
    c = c0.copy()
    for step in range(steps):
        c = step_forward_separable(c, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho, brain_mask=brain_mask)
        if verbose:
            print(f"Step {step} finished")
    return c


# =========================
# Loss & optimization
# =========================
def joint_loss_separable(params, list_c0, list_c1, list_brain, Qx, Qy, Qz, Lx, Ly, Lz, dt, steps=90, n_jobs=16):
    """
    Sum of squared L2 errors across all training cases.
    """
    D, rho = params
    if D <= 0 or rho <= 0:
        return np.inf

    Lx_scaled, Ly_scaled, Lz_scaled = D * Lx, D * Ly, D * Lz

    def loss_one(c0, c1_true, brain):
        c1_pred = simulate_N_steps(
            c0, Qx, Qy, Qz,
            Lx_scaled, Ly_scaled, Lz_scaled,
            dt, rho, steps,
            brain_mask=brain,
            verbose=False
        )
        return np.linalg.norm(c1_pred - c1_true) ** 2

    loss_list = Parallel(n_jobs=n_jobs)(
        delayed(loss_one)(c0, c1, brain) for c0, c1, brain in zip(list_c0, list_c1, list_brain)
    )
    return float(np.sum(loss_list))


def fit_parameters_separable(list_c0_3d, list_c1_3d, list_brain_3d, dx=1.0, dt=1.0, steps=90, n_jobs=16):
    """
    Fit D and rho using L-BFGS-B on the training set.
    """
    n = list_c0_3d.shape[1]
    Qx, Lx = eig_1d_laplacian_matrix(n, dx)
    Qy, Ly = eig_1d_laplacian_matrix(n, dx)
    Qz, Lz = eig_1d_laplacian_matrix(n, dx)

    initial = [0.001, 0.02]
    bounds = [(1e-5, 0.1), (1e-4, 0.1)]

    result = minimize(
        joint_loss_separable,
        initial,
        args=(list_c0_3d, list_c1_3d, list_brain_3d, Qx, Qy, Qz, Lx, Ly, Lz, dt, steps, n_jobs),
        bounds=bounds,
        method="L-BFGS-B"
    )
    return result.x, result.fun, (Qx, Qy, Qz), (Lx, Ly, Lz)


# =========================
# Metrics
# =========================
def dice_score(a, b, eps=1e-8):
    """
    Dice coefficient for binary masks.
    a, b: {0,1} arrays
    """
    inter = np.sum(a * b)
    return float((2.0 * inter + eps) / (np.sum(a) + np.sum(b) + eps))


# =========================
# Main
# =========================
def main():
    # 1) Load dataset
    list_c0, list_c1, list_brain, case_ids, ref_t1_paths = build_dataset_from_registered(
        brain_dir=BRAIN_DIR,
        tumor_dir=TUMOR_DIR,
        tumor_is_prob=TUMOR_IS_PROB
    )

    if len(list_c0) == 0:
        raise RuntimeError("No cases loaded. Please check directories and filename patterns.")

    # 2) Split: fixed test set
    is_test = np.array([cid in TEST_SET for cid in case_ids])

    train_c0 = list_c0[~is_test]
    train_c1 = list_c1[~is_test]
    train_brain = list_brain[~is_test]

    test_c0 = list_c0[is_test]
    test_c1 = list_c1[is_test]
    test_brain = list_brain[is_test]
    test_ids = case_ids[is_test]
    test_refs = ref_t1_paths[is_test]

    print(f"Train N = {len(train_c0)} | Test N = {len(test_c0)} | Test IDs = {list(test_ids)}")

    if len(test_c0) == 0:
        raise RuntimeError("No test cases found. Check TEST_SET names vs extracted case_ids.")

    # 3) Fit parameters on training set
    params, train_loss, Qs, Ls = fit_parameters_separable(
        train_c0, train_c1, train_brain,
        dx=DX, dt=DT, steps=STEPS, n_jobs=N_JOBS
    )

    D, rho = params
    Qx, Qy, Qz = Qs
    Lx, Ly, Lz = Ls

    print(f"Fitted parameters: D = {D:.6g}, rho = {rho:.6g}, train loss = {train_loss:.6g}")

    # Save fitted params
    np.save("params.npy", np.array([D, rho, train_loss], dtype=np.float32))
    np.save("Ls.npy", np.array(Ls, dtype=object), allow_pickle=True)
    np.save("Qs.npy", np.array(Qs, dtype=object), allow_pickle=True)

    # 4) Predict on test set, compute Dice, save NIfTI masks
    dice_list = []

    # Scale eigenvalues by D once (for speed)
    Lx_s, Ly_s, Lz_s = D * Lx, D * Ly, D * Lz

    for cid, c0, c1_true, brain, ref_path in zip(test_ids, test_c0, test_c1, test_brain, test_refs):
        c1_pred = simulate_N_steps(
            c0, Qx, Qy, Qz,
            Lx_s, Ly_s, Lz_s,
            DT, rho, STEPS,
            brain_mask=brain,
            verbose=False
        )

        # Dice on thresholded predictions
        pred_bin = (c1_pred > 0.5).astype(np.float32)
        true_bin = (c1_true > 0.5).astype(np.float32)
        d = dice_score(pred_bin, true_bin)
        dice_list.append(d)

        print(f"[TEST] {cid}: Dice = {d:.4f}")

        # Save predicted mask as NIfTI aligned to reference t1 tumor file
        out_path = os.path.join(OUT_DIR, f"{cid}_t1_pred_mask.nii.gz")
        save_pred_mask_as_nii(c1_pred, reference_nii_path=ref_path, out_path=out_path, threshold=0.5)
        print(f"Saved NIfTI: {out_path}")

    print(f"Mean Dice (TEST): {float(np.mean(dice_list)):.4f}")


if __name__ == "__main__":
    main()
