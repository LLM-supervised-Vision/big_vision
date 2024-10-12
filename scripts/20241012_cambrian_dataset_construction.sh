#!/bin/bash

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in cambrian_dataset_construction.sh. Exiting gracefully."
    # Kill all child processes
    pkill -P $$
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT

if [ -d "/home/austinwang/tensorflow_datasets" ]; then
    rm -r /home/austinwang/tensorflow_datasets
fi

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_config>"
    echo "dataset_config options: 737k, 10M"
    exit 1
fi

dataset_config=$1

case $dataset_config in
    "737k")
        num_samples=736936
        ;;
    "10M")
        num_samples=9784500 # uncertain for now
        ;;
    *)
        echo "Invalid dataset_config. Options are: 737k, 10M"
        exit 1
        ;;
esac

num_samples_per_job=40000
num_jobs_per_split=16

num_jobs=$(( (num_samples + num_samples_per_job - 1) / num_samples_per_job ))
num_splits=$(( (num_jobs + num_jobs_per_split - 1) / num_jobs_per_split ))

echo "Dataset config: $dataset_config"
echo "num_jobs $num_jobs"
echo "num_splits $num_splits"

for i in $(seq 0 $((num_splits-1)))
do
    for j in $(seq 0 $((num_jobs_per_split-1)))
    do
        job_id=$((i * num_jobs_per_split + j))
        if [ "$job_id" -lt "$num_jobs" ]; then
            echo "Starting split $i, job $j, job_id $job_id, start_sample $((job_id * num_samples_per_job))"
            # python /home/austinwang/austin_big_vision/scripts/cambrian_dataset_construction.py --config $dataset_config --job_id $job_id --num_jobs $num_jobs --gcs_tfds True &
            # sleep 0.5
        fi
    done
    wait
    echo "Split $i done"
    # sleep 3
done

# # Merge dataset versions
# echo "Merging dataset versions..."
# python -c "
# import tensorflow_datasets as tfds
# import tensorflow as tf

# builder = tfds.builder('cambrian_dataset', config='$dataset_config', data_dir='gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets')
# builder.download_and_prepare(
#     download_config=tfds.download.DownloadConfig(
#         beam_options=tfds.core.BeamOptions(
#             runner='DirectRunner',
#             direct_num_workers=1,
#             direct_running_mode='multi_processing',
#         )
#     )
# )
# print('Dataset merged successfully')
# "