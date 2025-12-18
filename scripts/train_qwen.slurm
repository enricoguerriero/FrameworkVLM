#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=train_qwen
#SBATCH --output=outputs/train_qwen_$(date +%Y%m%d_%H%M%S).out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=.
python -m src.training --model Qwen3VL --debug

echo "--- THE END ---"
