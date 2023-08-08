#!/bin/bash -l

#SBATCH --job-name="prep"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLURMY

python \
src/main.py \
--output_dir="IGNORED"
