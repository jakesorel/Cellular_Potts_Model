#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:30:00   # walltime
#SBATCH -J "cpm_softstiff"   # job name
#SBATCH --output=bash_out/output_softstiff.out
#SBATCH --error=bash_out/error_softstiff.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph

((i_iter =  ${SLURM_ARRAY_TASK_ID} + "$1"*"$2"))

python run_softstiff_simulation.py "$i_iter"
python run_analysis.py "$i_iter"

#python run_cluster_bootstrap.py "$1" "$2"

#
#((jp1 = "$1" + 1))
#
#if [[ "$1" -lt "$2" ]]
#then
#    sbatch run_simulation.sh "$jp1" "$2"
#fi
