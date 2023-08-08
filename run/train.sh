#!/bin/bash -l

#SBATCH --job-name="train"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G

export CUDA_VISIBLE_DEVICES=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLURMY

torchrun --nproc-per-node 1 \
src/main.py \
--pretrained_model_name_or_path="gpt2" \
--scale=1.0 \
--output_dir="./output" \
--overwrite_output_dir=true \
--do_train \
--do_eval \
--num_train_epochs=1 \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=1024 \
--dataloader_num_workers=16 \
--optim="adamw_torch" \
--sharded_ddp="offload" \
--fp16=true
# --tf32=true \  # A100s only
