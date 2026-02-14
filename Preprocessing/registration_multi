import SimpleITK as sitk
import os
import shutil
from collections import defaultdict

# ---------------- Paths ----------------
mri_root   = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/original_corrected'
brain_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/brain_seg_corrected'
tumor_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/tumor_seg_test'

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


def register_affine_t1_to_t0(t0_path, t1_path, fixed_mask_path=None):
    
    fixed_img  = sitk.ReadImage(t0_path)
    moving_img = sitk.ReadImage(t1_path)

    fixed  = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving = sitk.Cast(moving_img, sitk.sitkFloat32)

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

    # resample MRI t1 -> t0 space
    t1_to_t0 = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )
    return final_transform, fixed_img, t1_to_t0


def resample_label_to_fixed(label_path, fixed_img, transform, out_path):
    """
    warp label (t1 seg) -> t0 space (NearestNeighbor)
    """
    lab = sitk.ReadImage(label_path)
    warped = sitk.Resample(
        lab,
        fixed_img,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        lab.GetPixelID()
    )
    sitk.WriteImage(warped, out_path)


# ---------------- Build index ----------------
mri_pairs   = build_pairs(mri_root)
brain_pairs = build_pairs(brain_root)
tumor_pairs = build_pairs(tumor_root)

failed = []
missing_mri_pair = []
missing_brain = []
missing_tumor = []

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
        tx, fixed_img, mri_t1_to_t0 = register_affine_t1_to_t0(
            mri_t0_path, mri_t1_path, fixed_mask_path=fixed_mask_path
        )

        # ---- save MRI outputs ----
        # copy t0 MRI
        shutil.copy(mri_t0_path, os.path.join(out_mri, mri_t0_fn))
        # save registered t1 MRI
        sitk.WriteImage(mri_t1_to_t0, os.path.join(out_mri, mri_t1_fn))
        # save transform
        sitk.WriteTransform(tx, os.path.join(out_tfm, base + "_t1_to_t0.tfm"))

        # ---- brain seg: copy t0, warp t1 ----
        if base in brain_pairs and "t0" in brain_pairs[base]:
            brain_t0_fn = brain_pairs[base]["t0"]
            shutil.copy(os.path.join(brain_root, brain_t0_fn), os.path.join(out_brain, brain_t0_fn))
        else:
            missing_brain.append(base)

        if base in brain_pairs and "t1" in brain_pairs[base]:
            brain_t1_fn = brain_pairs[base]["t1"]
            resample_label_to_fixed(
                os.path.join(brain_root, brain_t1_fn),
                fixed_img,
                tx,
                os.path.join(out_brain, brain_t1_fn)
            )
        else:
            missing_brain.append(base + " (no t1)")

        # ---- tumor seg: copy t0, warp t1 ----
        if base in tumor_pairs and "t0" in tumor_pairs[base]:
            tumor_t0_fn = tumor_pairs[base]["t0"]
            shutil.copy(os.path.join(tumor_root, tumor_t0_fn), os.path.join(out_tumor, tumor_t0_fn))
        else:
            missing_tumor.append(base)

        if base in tumor_pairs and "t1" in tumor_pairs[base]:
            tumor_t1_fn = tumor_pairs[base]["t1"]
            resample_label_to_fixed(
                os.path.join(tumor_root, tumor_t1_fn),
                fixed_img,
                tx,
                os.path.join(out_tumor, tumor_t1_fn)
            )
        else:
            missing_tumor.append(base + " (no t1)")

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
