#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zengy2@ccf.org
#SBATCH --job-name=brain_seg
#SBATCH -c 16
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem 100G
#SBATCH -o brain_seg_%A_%a.out
#SBATCH -e brain_seg_%A_%a.err

module load python/gpu/3.10.6

INPUT_DIR="/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/registered"
OUTPUT_DIR="/home/zengy2/isilon/Isaac/MRI_data/Clinic/Simulation/data/brain_seg_ori"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.nii.gz; do
    filename=$(basename "$file")
    echo "Processing $filename"
    hd-bet -i "$file" -o "$OUTPUT_DIR/$filename"
done
