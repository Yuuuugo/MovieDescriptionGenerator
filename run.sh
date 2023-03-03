#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE

module purge
source $WORKDIR/MovieDescriptionGenerator/env.sh
cd $WORKDIR/MovieDescriptionGenerator
python3 train.py 