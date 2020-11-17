#!/bin/bash                                                                                                                           
#$1 is number of parameter steps, #2 is the number of repeats. Each parameter combo is a sepearate job

for i in $(seq 0 $(($(($1*$1*2))-1)))
do
    sbatch run_job_polrobust.sh "$i" "$1" "$2"
done


