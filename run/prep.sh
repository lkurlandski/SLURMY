#!/bin/bash -l

#SBATCH --job-name="prep"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLURMY

python \
src/main.py \
--dataset_name="ag_news" \
--model_name="roberta-large"
