#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=47:59:00
#SBATCH --job-name=minicpm_test
#SBATCH --output=slurm/minicpm_test_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --error=slurm/minicpm_test_%j.err
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:1


wandb agent julian-yuya-caspary-university-of-mannheim/universal-vlm-jailbreak/acsmgg0k