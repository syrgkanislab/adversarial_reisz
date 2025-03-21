#!/bin/bash 

#SBATCH --job-name=advriesz_gcv
#SBATCH -o <PATH OF OUTPUT LOG>
#SBATCH -e <PATH OF ERROR LOG>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<EMAIL>

#SBATCH -t <TIME LIMIT>
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --partition=<PARTITION>
#SBATCH -N 1

module load <PATH OF PYTHON INSTALLATION>
source venv/bin/activate
python Results_GCV.py