#!/bin/bash
#SBATCH --job-name=gru_train
#SBATCH --output=logs/%x_%j.out       # Save output to logs folder, %x=jobname, %j=job ID
#SBATCH --error=logs/%x_%j.err        # Save errors
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00               # Adjust as needed
#SBATCH --partition=gpu               # Change if your cluster uses another partition name

# Activate your environment
source ~/MRI/frankenstein/venv/bin/activate

# Navigate to the correct directory
cd ~/MRI/frankenstein/src

# Run your Python script with config
python main.py -config ./configs/config_GRU.yaml
