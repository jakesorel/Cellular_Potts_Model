#!/bin/bash                                                                                                                           


for i in $(seq 0 $(($1-1)))
do
    sbatch run_job_dynamic_tension_p0.sh "$i" "$1" "$2"
done


