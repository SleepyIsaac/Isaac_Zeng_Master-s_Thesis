import SimpleITK as sitk
import os
import shutil
import numpy as np

root_path = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/all_scans_labeled'
file_names = os.listdir(root_path)

file_names = [f.split('_')[0] + '_' + f.split('_')[1] + '_' + f.split('_')[2] for f in file_names]
file_names = np.unique(file_names)


output_root = '/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/registered'
os.makedirs(output_root, exist_ok=True)

def register_affine(t0, t1, input_root):

    t0_path = os.path.join(input_root, t0)
    t1_path = os.path.join(input_root, t1)

    fixed = sitk.ReadImage(t0_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(t1_path, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=200, relaxationFactor=0.5
    )
    registration_method.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)

    moving_resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    sitk.WriteImage(moving_resampled, os.path.join(output_root, t1))

    shutil.copy(t0_path, os.path.join(output_root, t0))

failed_list = []

for file in file_names:
    t0 = file + '_t0.nii.gz'
    t1 = file + '_t1.nii.gz'
    
    try:
        register_affine(t0, t1, root_path)
    except:
        failed_list.append(file)

print(failed_list)
with open("failed.txt", "w") as f:
    for item in failed_list:
        f.write(item + "\n")



