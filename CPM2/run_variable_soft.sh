#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=4:30:00   # walltime
#SBATCH -J "cpm_variable_soft"   # job name
#SBATCH --output=bash_out/output_variable_soft.out
#SBATCH --error=bash_out/error_variable_soft.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph

((i_iter =  ${SLURM_ARRAY_TASK_ID} + "$1"*"$2"))

python run_softstiff_simulation3.py "$i_iter"
python run_analysis_variable_soft.py "$i_iter"

#python run_cluster_bootstrap.py "$1" "$2"

#
#((jp1 = "$1" + 1))
#
#if [[ "$1" -lt "$2" ]]
#then
#    sbatch run_simulation.sh "$jp1" "$2"
#fi
