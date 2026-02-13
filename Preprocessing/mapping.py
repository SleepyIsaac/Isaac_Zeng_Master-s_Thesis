import pandas as pd
import shutil
import os

long_unique = pd.read_csv('pair_long.csv')
check_sheet = pd.read_csv('Meningioma Seg Check.csv')

check_sheet = check_sheet.iloc[:51]
check_sheet['Original File'] = check_sheet['Original File'].str.extract(r'(.*?\d{14}_\d+)', expand=False)
check_sheet['Original File'] = check_sheet['Original File'] + ".nii.gz"
check_sheet = check_sheet[~check_sheet['Comment'].str.lower().str.contains('delete|do not include', na=False)]

long_unique['file_name'] = long_unique['file_name'].str.extract(r'(.*?\d{14}_\d+)', expand=False)
long_unique['file_name'] = long_unique['file_name'] + ".nii.gz"

pair_check = check_sheet['Pair'].tolist()
ori_check = check_sheet['Original File'].tolist()
pair_long = long_unique['pair'].tolist()
ori_long = long_unique['file_name'].tolist()

ori_long = [str(i).replace('_out_orig', '') for i in ori_long]
ori_long = [str(i).replace('_out', '') for i in ori_long]

pair_long = [i+'.nii.gz' for i in pair_long]
pair_check = [i+'.nii.gz' for i in pair_check]

from collections import defaultdict

long_map = defaultdict(list)

for file_long, pair_id in zip(ori_long, pair_long):
    long_map[file_long].append(pair_id)

map_dict = defaultdict(list)

for file_check, pair_id_check in zip(ori_check, pair_check):
    if file_check in long_map:
        map_dict[pair_id_check].extend(long_map[file_check])


for ori_file in map_dict:
    for file in map_dict[ori_file]:
        src = f"/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/original/{ori_file}"
        dst = f"/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/original_corrected/{file}"
        shutil.copy(src, dst)

dst = "/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data_T1/original_corrected/"
files = os.listdir(dst)


for file in files:
    if file.endswith("_t0.nii.gz"):
        base = file.replace("_t0.nii.gz", "")
        if base + "_t1.nii.gz" not in files:
            os.remove(os.path.join(dst, file))

    elif file.endswith("_t1.nii.gz"):
        base = file.replace("_t1.nii.gz", "")
        if base + "_t0.nii.gz" not in files:
            os.remove(os.path.join(dst, file))

