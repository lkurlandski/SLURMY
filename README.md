# SLURMY
Demonstration of using SLURM for large model training.

## RC

- Fill out a "Research Computing Project Request" (https://www.rit.edu/researchcomputing/services#high-performance-computing-hpc-access), then nstall Anaconda locally (https://www.anaconda.com/).

## Setup

conda is the most convenient way to manage packages on RC (in my opinion).

```
conda create --name SLURMY python=3.11 pytorch=2.0 pytorch-cuda=11.8 transformers datasets tokenizers scikit-learn matplotlib pandas tqdm -c pytorch -c nvidia -c huggingface -c conda-forge
conda activate SLURMY
```

## Usage

Use `bash` for local computing without SLURM (obviously leave out the --account flag).

```
sbatch --account={ACCOUNT} run/prep.sh
sbatch --account={ACCOUNT} run/train.sh
```
