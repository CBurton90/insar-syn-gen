#!/bin/bash

#SBATCH --job-name variogram
#SBATCH --time=48:00:00
#SBATCH --mem 100G
#SBATCH --cpus-per-task=64
#SBATCH -o variogram.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate insar
python3 spatial_variogram.py
conda deactivate
