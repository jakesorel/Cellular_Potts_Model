#!/bin/bash                                                                                                                           

n_iter=$1
n_param_step=$2
N_job=$3

for i in $(seq 0 $(($3-1)))
do
    sbatch run_jobCPM.sh "$1" "$2" "$3" "$i"
done

