#!/bin/bash
#SBATCH --job-name=cnn_training
#SBATCH --output=logs/train_%j.out        # stdout log
#SBATCH --error=logs/train_%j.err         # stderr log
#SBATCH --partition=bmeph                   # adjust if your cluster has a specific name
#SBATCH --gres=gpu:1                      # request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00                   # max time (HH:MM:SS)

# Load Python (only if your cluster uses modules)
#module load python/3.10

# Activate your virtual environment
source /scratch/tgoedietdoebe/AI-project/venv/bin/activate

# Create output dir if it doesn't exist
mkdir -p results

# Run training
python /scratch/tgoedietdoebe/AI-project/src/CNN_Base_SLURM.py \
    --epochs 1 \
    --batch_size 8 \
   # --preprocessed_path /scratch/tgoedietdoebe/AI-project/data/prmocessed/ \
  #--save_model_path results/cnn_model.pt
