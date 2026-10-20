#!/bin/bash
#SBATCH --job-name=cryo-unet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail

# Adapt these two lines to the cluster. Keep real data on project/scratch storage.
source .venv/bin/activate
python src/train.py --config config.yaml

