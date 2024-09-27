#!/bin/bash

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in merge_segments.sh. Exiting gracefully."
    # Kill all child processes
    pkill -P $$
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT

if [ -d "/home/austinwang/tensorflow_datasets" ]; then
    rm -r /home/austinwang/tensorflow_datasets
fi

config=10M
num_samples=10000000
num_samples_per_job=20000
num_jobs_per_split=16

num_jobs=$(( (num_samples + num_samples_per_job - 1) / num_samples_per_job ))
num_splits=$(( (num_jobs + num_jobs_per_split - 1) / num_jobs_per_split ))

echo "num_jobs $num_jobs"
echo "num_splits $num_splits"

for i in $(seq 0 $((num_splits-1)))
do
    for j in $(seq 0 $((num_jobs_per_split-1)))
    do
        job_id=$((i * num_jobs_per_split + j))
        if [ "$job_id" -lt "$num_jobs" ]; then
            echo "Starting split $i, job $j, job_id $job_id, start_sample $((job_id * num_samples_per_job))"
            python /home/austinwang/austin_big_vision/scripts/20240924_datacomp_recap_construction.py --config $config --job_id $job_id --num_jobs $num_jobs --gcs_tfds True &
            sleep 0.5
        fi
    done
    wait
    echo "Split $i done"
    sleep 3
done