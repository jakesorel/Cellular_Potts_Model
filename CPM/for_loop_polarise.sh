#!/bin/bash                                                                                                                           


for i in $(seq 0 $(($1-1)))
do
    sbatch run_job_polarise.sh "$i" "$1" "$2"
done


