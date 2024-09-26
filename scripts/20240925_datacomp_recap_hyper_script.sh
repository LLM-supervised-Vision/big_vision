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

config=1M
num_jobs=10

for i in $(seq 0 $((num_jobs-1)))
do
    echo "Starting job $((i))"
    python /home/austinwang/austin_big_vision/scripts/20240924_datacomp_recap_construction.py --config $config --job_id $i --num_jobs $num_jobs --gcs_tfds True &
    sleep 1
done