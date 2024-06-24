#!/bin/bash

num_shards=62917
real_start=1923

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected. Stopping the script."
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT


for i in {0..10}; do
    start_idx=$((real_start + 6000 * i))
    # echo "Start index: $start_idx"
    end_idx=$(( real_start + 6000 * (i + 1) ))
    end_idx=$((end_idx < num_shards ? end_idx : num_shards))
    # echo "End index: $end_idx"

    bash /home/austinwang/renaming_shards.sh $start_idx $end_idx $i &
    sleep 1
done

wait