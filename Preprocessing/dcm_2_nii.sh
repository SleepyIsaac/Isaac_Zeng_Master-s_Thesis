#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zengy2@ccf.org
#SBATCH --job-name=move_dcm
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -n 1
#SBATCH --mem 50G
#SBATCH -o dcm_%A_%a.out
#SBATCH -e dcm_%A_%a.err

INPUT_ROOT=/home/zengy2/isilon/MRI-1-30-2025
OUTPUT_ROOT=/home/zengy2/isilon/Isaac/MRI_data/Clinic/all_scans_NIfTI

cd /home/zengy2/beegfs

for folder in "$INPUT_ROOT"/*; do
  if [ -d "$folder" ]; then
    patient=$(basename "$folder")
    echo "Processing $patient"
    ./dcm2niix -z y -o "$OUTPUT_ROOT" "$folder"
  fi
done
