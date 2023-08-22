# SLURMY
Demonstration of using SLURM for large model training.

## Setup

Feel free to use pip instead.

```
conda create --name SLURMY python=3.11 pytorch=2.0 pytorch-cuda=11.8 transformers datasets tokenizers scikit-learn matplotlib pandas tqdm -c pytorch -c nvidia -c huggingface -c conda-forge
conda activate SLURMY
```

## Usage

Use `bash` for local computing without SLURM.

```
sbatch run/prep.sh
sbatch run/train.sh
```
