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
        num_samples=9784416 # uncertain for now
        ;;
    *)
        echo "Invalid dataset_config. Options are: 737k, 10M"
        exit 1
        ;;
esac

num_samples_per_job=40000
num_jobs_per_split=16
DEBUG=True
if [ "$DEBUG" = True ]; then
    num_samples_per_job=40000
    num_jobs_per_split=6
    # remove 1.0.{0,..,num_jobs_per_split-1} directories
    for i in $(seq 0 $((num_jobs_per_split-1)))
    do
        gcs_path="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/cambrian_dataset/$dataset_config/1.0.$i"
        # if gsutil ls $gcs_path; then
        #     gsutil -m rm -r $gcs_path
        # fi
    done
fi

num_jobs=$(( (num_samples + num_samples_per_job - 1) / num_samples_per_job ))
num_splits=$(( (num_jobs + num_jobs_per_split - 1) / num_jobs_per_split ))

echo "Dataset config: $dataset_config"
echo "Total samples: $num_samples"
echo "Samples per job: $num_samples_per_job"
echo "Jobs per split: $num_jobs_per_split"
echo "Number of jobs: $num_jobs"
echo "Number of splits: $num_splits"

# Example job for debugging
i=0
j=3
job_id=$((i * num_jobs_per_split + j))
echo "Starting example job: split $i, job $j, job_id $job_id"
python /home/austinwang/austin_big_vision/scripts/20241012_cambrian_dataset_construction.py \
    --config $dataset_config \
    --job_id $job_id \
    --num_jobs $num_jobs \
    --use_parallel True \
    --gcs_tfds True

# Uncomment the following loop for full execution
# for i in $(seq 0 $((num_splits-1)))
# do
#     for j in $(seq 0 $((num_jobs_per_split-1)))
#     do
#         job_id=$((i * num_jobs_per_split + j))
#         if [ "$job_id" -lt "$num_jobs" ]; then
#             echo "Starting split $i, job $j, job_id $job_id"
#             python /home/austinwang/austin_big_vision/scripts/20241012_cambrian_dataset_construction.py \
#                 --config $dataset_config \
#                 --job_id $job_id \
#                 --num_jobs $num_jobs \
#                 --use_parallel True \
#                 --gcs_tfds True \
#                 --local_data_dir "/home/austinwang/tensorflow_datasets" \
#                 --gcs_data_dir "gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets" &
#             sleep 0.5
#         fi
#     done
#     wait
#     echo "Split $i done"
#     if [ "$DEBUG" = True ]; then
#         exit 0
#     fi
#     sleep 3
# done

# echo "All splits completed. Dataset construction finished."