#!/bin/bash

#SBATCH --job-name interpolate
#SBATCH --time=24:00:00
#SBATCH --mem 100G
#SBATCH --cpus-per-task=64
#SBATCH -o interpolate.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate insar
python3 run_interp.py
conda deactivate
