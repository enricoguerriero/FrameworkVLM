#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --job-name=optim_qwen
#SBATCH --output=outputs/optim_qwen_%j.out

echo "Job started on $(hostname) at $(date)"

uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

source activate NewbornEnv || conda activate NewbornEnv
echo "virtual environment activated"

export PYTHONPATH=.
python -m src.hyperparam_optim.py --model Qwen3VL --storage sqlite:///optuna_hpo.db 

echo "--- THE END ---"