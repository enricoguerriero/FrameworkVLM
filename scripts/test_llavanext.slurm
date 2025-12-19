#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --job-name=test_llavanext
#SBATCH --output=outputs/test_llavanext_%j.out

set -e

mkdir -p outputs
mkdir -p models
mkdir -p checkpoints

echo "Job started on $(hostname) at $(date)"

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv || conda activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=.
python -m src.test --model LLaVANeXT --debug --model_path checkpoints/_LLaVANeXT_epoch1_trainloss0.3433.pt

echo "--- THE END ---"
