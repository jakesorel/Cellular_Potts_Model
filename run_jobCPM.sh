#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=64   # number of processor cores (i.e. tasks)
#SBATCH -J "cell_sorting_jakecs"   # job name
#SBATCH --output=output.out
#SBATCH --error=output.out   
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jakecs@caltech.edu

source activate apical_domain

python CPM/CPM_ND_cluster.py "$1" "$2" "$3" "$4"
