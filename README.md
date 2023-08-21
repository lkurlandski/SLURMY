# SLURMY
Demonstration of using SLURM for large model training.

## Setup

```
conda create --name SLURMY python=3.11
conda activate SLURMY
pip install torch torchvision torchaudio torchtext transformers accelerate evaluate datasets tokenizers scikit-learn matplotlib
```

## Usage

```
bash run/prep.sh
bash run/train.sh
``````
