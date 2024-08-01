#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100

python run_long_term.py