#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -J "cell_sorting_jakecs"   # job name
#SBATCH --output=output.out
#SBATCH --error=output.out   
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jakecs@caltech.edu

source activate apical_domain

python CPM_2_cell_jamming_time_tension.py "$1" "$2" "$3"