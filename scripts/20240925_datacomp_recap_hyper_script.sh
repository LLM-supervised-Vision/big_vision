#!/bin/bash

if [ -d "/home/austinwang/tensorflow_datasets" ]; then
    rm -r /home/austinwang/tensorflow_datasets
fi

config=100
num_jobs=2

for i in $(seq 0 $((num_jobs-1)))
do
    echo "Starting job $((i))"
    python /home/austinwang/austin_big_vision/scripts/20240924_datacomp_recap_construction.py --config $config --job_id $i --num_jobs $num_jobs
    # sleep 1
done