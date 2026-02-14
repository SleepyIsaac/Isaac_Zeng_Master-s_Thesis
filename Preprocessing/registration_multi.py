import SimpleITK as sitk
import os
from collections import defaultdict
import csv

# =========================
# User-configurable params
# =========================
CANON_SPACING = (1.0, 1.0, 1.0)        # isotropic spacing for PDE
CANON_SIZE    = (256, 256, 256)        # change to (128,128,128) if you want

# ---------------- Paths ----------------
mri_root   = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/original_corrected'
brain_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/brain_seg_corrected'
tumor_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/tumor_seg_nii'

out_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/registered'
os.makedirs(out_root, exist_ok=True)

out_mri   = os.path.join(out_root, "mri")
out_brain = os.path.join(out_root, "brain_seg")
out_tumor = os.path.join(out_root, "tumor_seg")
out_tfm   = os.path.join(out_root, "tfm")
for p in [out_mri, out_brain, out_tumor, out_tfm]:
    os.makedirs(p, exist_ok=True)


# ---------------- Helpers ----------------
def parse_base_and_time(fname: str):
    if not fname.endswith(".nii.gz"):
        return None, None
    if fname.endswith("_t0.nii.gz"):
        return fname[:-len("_t0.nii.gz")], "t0"
    if fname.endswith("_t1.nii.gz"):
        return fname[:-len("_t1.nii.gz")], "t1"
    return None, None


def build_pairs(folder):
    pairs = defaultdict(dict)
    for fn in os.listdir(folder):
        base, tt = parse_base_and_time(fn)
        if base is None:
            continue
        pairs[base][tt] = fn
    return pairs


def resample_to_spacing_and_size(
    img: sitk.Image,
    out_spacing=(1.0, 1.0, 1.0),
    out_size=(256, 256, 256),
    is_label=False,
    default_value=0
) -> sitk.Image:
    """
    Resample image to a fixed spacing + fixed size, preserving physical center.
    This makes all subjects comparable for PDE (same voxel size + same grid).
    """
    out_spacing = tuple(map(float, out_spacing))
    out_size    = tuple(map(int, out_size))

    in_spacing = img.GetSpacing()
    in_size    = img.GetSize()
    in_origin  = img.GetOrigin()
    in_dir     = img.GetDirection()

    # physical extent in each axis (approx)
    in_phys  = [in_spacing[i] * (in_size[i] - 1) for i in range(3)]
    out_phys = [out_spacing[i] * (out_size[i] - 1) for i in range(3)]

    # center-preserving origin shift
    out_origin = [in_origin[i] + 0.5 * (in_phys[i] - out_phys[i]) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetOutputDirection(in_dir)
    resampler.SetTransform(sitk.Transform())  # identity

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def register_affine_t1_to_t0(t0_path, t1_path, fixed_mask_path=None):
    """
    Returns:
      final_transform: affine mapping moving(t1) -> fixed(t0)
      fixed_img_original: original t0 image (native grid)
      t1_warped_to_t0: t1 resampled onto t0 native grid
    """
    fixed_img_original  = sitk.ReadImage(t0_path)
    moving_img_original = sitk.ReadImage(t1_path)

    fixed  = sitk.Cast(fixed_img_original, sitk.sitkFloat32)
    moving = sitk.Cast(moving_img_original, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)
    R.SetInterpolator(sitk.sitkLinear)

    # multi-resolution
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # optimizer
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8
    )
    R.SetOptimizerScalesFromPhysicalShift()

    if fixed_mask_path is not None and os.path.exists(fixed_mask_path):
        fm = sitk.ReadImage(fixed_mask_path)
        R.SetMetricFixedMask(sitk.Cast(fm > 0, sitk.sitkUInt8))

    R.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = R.Execute(fixed, moving)

    # warp t1 onto t0 native grid
    t1_warped_to_t0 = sitk.Resample(
        moving_img_original,
        fixed_img_original,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img_original.GetPixelID()
    )

    return final_transform, fixed_img_original, t1_warped_to_t0


def resample_label_t0_to_canon(label_path, canon_ref_img, out_path):
    """t0 label is already in t0 native grid; just resample to canonical grid (NN)."""
    lab = sitk.ReadImage(label_path)
    lab_canon = sitk.Resample(
        lab,
        canon_ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        lab.GetPixelID()
    )
    sitk.WriteImage(lab_canon, out_path)


def warp_label_t1_to_t0_then_canon(label_path, fixed_img_original, transform, canon_ref_img, out_path):
    """t1 label: warp -> t0 native grid, then resample -> canonical grid (NN)."""
    lab = sitk.ReadImage(label_path)

    warped_to_t0 = sitk.Resample(
        lab,
        fixed_img_original,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        lab.GetPixelID()
    )

    warped_canon = sitk.Resample(
        warped_to_t0,
        canon_ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        lab.GetPixelID()
    )
    sitk.WriteImage(warped_canon, out_path)


# ---------------- Build index ----------------
mri_pairs   = build_pairs(mri_root)
brain_pairs = build_pairs(brain_root)
tumor_pairs = build_pairs(tumor_root)

missing_mri_pair = []
missing_brain = []
missing_tumor = []
failed = []

qc_rows = []

for base, d in mri_pairs.items():
    if "t0" not in d or "t1" not in d:
        missing_mri_pair.append(base)
        continue

    mri_t0_fn = d["t0"]
    mri_t1_fn = d["t1"]

    mri_t0_path = os.path.join(mri_root, mri_t0_fn)
    mri_t1_path = os.path.join(mri_root, mri_t1_fn)

    fixed_mask_path = None
    if base in brain_pairs and "t0" in brain_pairs[base]:
        fixed_mask_path = os.path.join(brain_root, brain_pairs[base]["t0"])

    try:
        tx, fixed_img_original, t1_warped_to_t0 = register_affine_t1_to_t0(
            mri_t0_path, mri_t1_path, fixed_mask_path=fixed_mask_path
        )

        # ---- canonicalize MRI (both t0 and registered t1) ----
        t0_canon = resample_to_spacing_and_size(
            fixed_img_original, out_spacing=CANON_SPACING, out_size=CANON_SIZE, is_label=False, default_value=0
        )
        t1_canon = resample_to_spacing_and_size(
            t1_warped_to_t0, out_spacing=CANON_SPACING, out_size=CANON_SIZE, is_label=False, default_value=0
        )

        # ---- save MRI outputs ----
        sitk.WriteImage(t0_canon, os.path.join(out_mri, mri_t0_fn))
        sitk.WriteImage(t1_canon, os.path.join(out_mri, mri_t1_fn))
        sitk.WriteTransform(tx, os.path.join(out_tfm, base + "_t1_to_t0.tfm"))

        # ---- brain seg ----
        if base in brain_pairs and "t0" in brain_pairs[base]:
            brain_t0_fn = brain_pairs[base]["t0"]
            resample_label_t0_to_canon(
                os.path.join(brain_root, brain_t0_fn),
                t0_canon,
                os.path.join(out_brain, brain_t0_fn)
            )
        else:
            missing_brain.append(base + " (no t0)")

        if base in brain_pairs and "t1" in brain_pairs[base]:
            brain_t1_fn = brain_pairs[base]["t1"]
            warp_label_t1_to_t0_then_canon(
                os.path.join(brain_root, brain_t1_fn),
                fixed_img_original, tx, t0_canon,
                os.path.join(out_brain, brain_t1_fn)
            )
        else:
            missing_brain.append(base + " (no t1)")

        # ---- tumor seg ----
        if base in tumor_pairs and "t0" in tumor_pairs[base]:
            tumor_t0_fn = tumor_pairs[base]["t0"]
            resample_label_t0_to_canon(
                os.path.join(tumor_root, tumor_t0_fn),
                t0_canon,
                os.path.join(out_tumor, tumor_t0_fn)
            )
        else:
            missing_tumor.append(base + " (no t0)")

        if base in tumor_pairs and "t1" in tumor_pairs[base]:
            tumor_t1_fn = tumor_pairs[base]["t1"]
            warp_label_t1_to_t0_then_canon(
                os.path.join(tumor_root, tumor_t1_fn),
                fixed_img_original, tx, t0_canon,
                os.path.join(out_tumor, tumor_t1_fn)
            )
        else:
            missing_tumor.append(base + " (no t1)")

        # ---- QC row ----
        qc_rows.append({
            "base": base,
            "t0_native_size": fixed_img_original.GetSize(),
            "t0_native_spacing": fixed_img_original.GetSpacing(),
            "canon_size": t0_canon.GetSize(),
            "canon_spacing": t0_canon.GetSpacing(),
        })

    except Exception as e:
        failed.append((base, str(e)))


# ---------------- Logs ----------------
print("Missing MRI pairs:", len(missing_mri_pair))
print("Missing brain seg:", len(missing_brain))
print("Missing tumor seg:", len(missing_tumor))
print("Failed:", len(failed))

with open(os.path.join(out_root, "missing_mri_pairs.txt"), "w") as f:
    for b in missing_mri_pair:
        f.write(b + "\n")

with open(os.path.join(out_root, "missing_brain.txt"), "w") as f:
    for b in missing_brain:
        f.write(b + "\n")

with open(os.path.join(out_root, "missing_tumor.txt"), "w") as f:
    for b in missing_tumor:
        f.write(b + "\n")

with open(os.path.join(out_root, "failed.txt"), "w") as f:
    for b, err in failed:
        f.write(f"{b}\t{err}\n")

# QC summary CSV
qc_path = os.path.join(out_root, "qc.csv")
with open(qc_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["base","t0_native_size","t0_native_spacing","canon_size","canon_spacing"])
    writer.writeheader()
    for row in qc_rows:
        writer.writerow(row)

print("Wrote QC:", qc_path)
