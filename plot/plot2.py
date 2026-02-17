import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion

def to_canonical(img):
    return nib.as_closest_canonical(img)

def resample_label_to_ref(label_img, ref_img):
    return resample_from_to(label_img, ref_img, order=0)

def robust_limits(img2d, p_low=1, p_high=99):
    vmin = np.percentile(img2d, p_low)
    vmax = np.percentile(img2d, p_high)
    return vmin, vmax

def outline(mask2d_bool):
    return mask2d_bool & (~binary_erosion(mask2d_bool))

def get_slice_2d(img3d, plane, index):
    data = img3d.get_fdata()
    sx, sy, sz = img3d.header.get_zooms()[:3]

    if plane == "axial":
        sl = data[:, :, index]
        aspect = sy / sx
    elif plane == "coronal":
        sl = data[:, index, :]
        aspect = sz / sx
    elif plane == "sagittal":
        sl = data[index, :, :]
        aspect = sz / sy
    else:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")

    sl = np.rot90(sl)
    sl = np.flipud(sl)
    return sl, aspect

def best_index_from_mask(mask_img, plane):
    m = mask_img.get_fdata() > 0.5
    if plane == "axial":
        counts = [m[:, :, k].sum() for k in range(m.shape[2])]
    elif plane == "coronal":
        counts = [m[:, j, :].sum() for j in range(m.shape[1])]
    elif plane == "sagittal":
        counts = [m[i, :, :].sum() for i in range(m.shape[0])]
    else:
        raise ValueError
    return int(np.argmax(counts))

PLANE = "coronal"
alpha_tumor = 0.55
draw_outline = True
flip_lr = False


cases = [
    {
        "name": "Case1",
        "t0_mri": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/mri/patient_25_pair1_t0.nii.gz",
        "t0_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/tumor_seg/patient_25_pair1_t0.nii.gz",
        "t1_mri": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/mri/patient_25_pair1_t1.nii.gz",
        "t1_true_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/tumor_seg/patient_25_pair1_t1.nii.gz",
        "t1_pred_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/predictions_test/patient_25_pair1_pred_t1.nii.gz",
        "plain": "axial",
        "idx": False,
    },
    {
        "name": "Case2",
        "t0_mri": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/mri/patient_31_pair1_t0.nii.gz",
        "t0_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/tumor_seg/patient_31_pair1_t0.nii.gz",
        "t1_mri": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/mri/patient_31_pair1_t1.nii.gz",
        "t1_true_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/registered/tumor_seg/patient_31_pair1_t1.nii.gz",
        "t1_pred_tumor": "I:/Isaac/MRI_data/Clinic/Simulation/data_T1/predictions_test/patient_31_pair1_pred_t1.nii.gz",
        "plain": "coronal",
        "idx": False,
    },
]




def prep_case(mri_path, tumor_path, plane):
    mri   = nib.load(mri_path)
    tumor = nib.load(tumor_path)

    tumor_rs = tumor

    mri_c   = to_canonical(mri)
    tumor_c = to_canonical(tumor_rs)

    idx = best_index_from_mask(tumor_c, plane)

    mri_sl, aspect = get_slice_2d(mri_c, plane, idx)
    tumor_sl, _    = get_slice_2d(tumor_c, plane, idx)

    tumor_sl = tumor_sl > 0.5

    if flip_lr:
        mri_sl   = np.fliplr(mri_sl)
        tumor_sl = np.fliplr(tumor_sl)

    vmin, vmax = robust_limits(mri_sl)
    return mri_sl, tumor_sl, aspect, (vmin, vmax), idx

def draw_panel(ax, mri_sl, tumor_sl, aspect, vmin, vmax):
    ax.imshow(
        mri_sl, cmap="gray",
        vmin=vmin, vmax=vmax,
        interpolation="bicubic",
        aspect=aspect, origin="lower"
    )

    ax.imshow(
        np.ma.masked_where(~tumor_sl, tumor_sl),
        cmap="Reds", alpha=alpha_tumor,
        interpolation="nearest",
        aspect=aspect, origin="lower"
    )

    if draw_outline:
        out = outline(tumor_sl)
        ax.imshow(
            np.ma.masked_where(~out, out),
            cmap="autumn", alpha=1.0,
            interpolation="nearest",
            aspect=aspect, origin="lower"
        )

    ax.axis("off")
    ax.set_facecolor("black")
  
PLANE = "coronal"
alpha_tumor = 0.55
draw_outline = True
flip_lr = False

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8),
    facecolor="black",
    gridspec_kw={"wspace": 0, "hspace": 0}
)


for r, case in enumerate(cases):

    # PLANE = case['plain']

    t0_mri  = to_canonical(nib.load(case["t0_mri"]))
    t0_msk  = to_canonical(nib.load(case["t0_tumor"]))
    t1_mri  = to_canonical(nib.load(case["t1_mri"]))
    t1_true = to_canonical(nib.load(case["t1_true_tumor"]))
    t1_pred = to_canonical(nib.load(case["t1_pred_tumor"]))

    if case['idx']:
        idx = case['idx']
    else:
        if (t1_true.get_fdata() > 0.5).any():
            idx = best_index_from_mask(t1_true, PLANE)
        elif (t1_pred.get_fdata() > 0.5).any():
            idx = best_index_from_mask(t1_pred, PLANE)
        else:
            idx = best_index_from_mask(t0_msk, PLANE)

    # ========== Col 1: t0 + t0 tumor ==========
    mri_sl, aspect = get_slice_2d(t0_mri, PLANE, idx)
    msk_sl, _      = get_slice_2d(t0_msk, PLANE, idx)
    msk_sl = msk_sl > 0.5
    if flip_lr:
        mri_sl = np.fliplr(mri_sl); msk_sl = np.fliplr(msk_sl)
    vmin, vmax = robust_limits(mri_sl)
    draw_panel(axes[r, 0], mri_sl, msk_sl, aspect, vmin, vmax)

    # ========== Col 2: t1 + true tumor ==========
    mri_sl, aspect = get_slice_2d(t1_mri, PLANE, idx)
    msk_sl, _      = get_slice_2d(t1_true, PLANE, idx)
    msk_sl = msk_sl > 0.5
    if flip_lr:
        mri_sl = np.fliplr(mri_sl); msk_sl = np.fliplr(msk_sl)
    vmin, vmax = robust_limits(mri_sl)
    draw_panel(axes[r, 1], mri_sl, msk_sl, aspect, vmin, vmax)

    # ========== Col 3: t1 + pred tumor ==========
    mri_sl, aspect = get_slice_2d(t1_mri, PLANE, idx)

    t0_sl, _   = get_slice_2d(t0_msk, PLANE, idx)
    pred_sl, _ = get_slice_2d(t1_pred, PLANE, idx)

    t0_sl   = t0_sl > 0.5
    pred_sl = pred_sl > 0.5

    growth_sl = pred_sl & (~t0_sl)

    if flip_lr:
        mri_sl = np.fliplr(mri_sl)
        growth_sl = np.fliplr(growth_sl)

    vmin, vmax = robust_limits(mri_sl)
    draw_panel(axes[r, 2], mri_sl, growth_sl, aspect, vmin, vmax)

# clean
for rr in range(2):
    for cc in range(3):
        axes[rr, cc].set_facecolor("black")
        axes[rr, cc].axis("off")

plt.subplots_adjust(left=0.06, right=1, top=0.95, bottom=0)
# plt.savefig("tumor_segmentation_2x3.png", dpi=600, facecolor="black")
plt.show()
