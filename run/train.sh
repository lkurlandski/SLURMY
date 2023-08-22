#!/bin/bash -l

#SBATCH --job-name="train"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLURMY

torchrun --nnodes=1 --nproc_per_node=1 \
src/main.py \
--dataset_name="trec" \
--model_name="prajjwal1/bert-tiny" \
--do_train \
--do_eval \
--do_anal \
--overwrite_output_dir \
--per_device_train_batch_size=512 \
--per_device_eval_batch_size=512 \
--num_train_epochs=10 \
--num_workers=16 \
--device="cuda"
