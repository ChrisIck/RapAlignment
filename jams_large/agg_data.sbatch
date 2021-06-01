#!/bin/bash

#SBATCH --job-name=agfmalarge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="o_agg_large-%j.out"

module purge
module load anaconda3/2020.07

source activate rap_env

python /home/ci411/RapAlignment/jams_large/agg_large.py