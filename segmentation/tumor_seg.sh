#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zengy2@ccf.org
#SBATCH --job-name=meningioma_seg
#SBATCH -c 16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=100G
#SBATCH -o seg_%A_%a.out
#SBATCH -e seg_%A_%a.err

module load python/gpu/3.10.6
module load cuda11.8
module load singularity

INPUT_DIR="/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/registered"
OUTPUT_DIR="/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/meningioma_seg_new"

mkdir -p "$OUTPUT_DIR"

FILES=("$INPUT_DIR"/*.nii.gz)
TOTAL_FILES=${#FILES[@]}

COUNT=0

for file in "${FILES[@]}"; do
    ((COUNT++))

    filename=$(basename "$file" .nii.gz)

    echo "Processing ${COUNT}/${TOTAL_FILES}: ${filename}.nii.gz"
    
    if [[ -f "$INPUT_DIR/${filename}.nii.gz" ]]; then
    echo "Found file"
    fi

    singularity run --cleanenv \
    --bind /home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data:/mnt \
    -W /mnt --nv \
    ams_latest-gpu.sif \
    "/mnt/registered/${filename}.nii.gz" "/mnt/meningioma_seg_new/${filename}"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: $file (Exit Code: $EXIT_CODE)"
    fi
done

echo "Processing complete."
