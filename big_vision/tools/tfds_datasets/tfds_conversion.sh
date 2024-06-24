#!/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets/

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in renaming_shards.sh. Exiting gracefully."
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT

total_num_shards=41408
num_workers=1 # 1 or 32
running_mode="in_memory" # multi_processing or multi_threading or in_memory
num_shards_per_job=20
num_jobs_per_split=16

# total_num_shards=400 # DEBUG
total_num_jobs=$(( (total_num_shards + num_shards_per_job - 1) / num_shards_per_job )) # 2071
total_num_split=$(( (total_num_jobs + num_jobs_per_split - 1) / num_jobs_per_split ))

echo "total_num_shards $total_num_shards"
echo "total_num_jobs $total_num_jobs"
echo "total_num_split $total_num_split"

for i in $(seq 0 $((total_num_split-1)))
do
    for j in $(seq 0 $((num_jobs_per_split-1)))
    do
        read_start=$((i * num_jobs_per_split * num_shards_per_job + j * num_shards_per_job))
        if [ "$read_start" -lt "$total_num_shards" ]; then
            echo "split $i, job $j, read_start $read_start"
            python /home/austinwang/vlm/data_processing/tfds_conversion.py \
                --direct_running_mode $running_mode \
                --direct_num_workers $num_workers \
                --read_start $read_start &
            sleep 1
        fi
    done
    wait
    echo "split $i done"
    sleep 3
done

wait


# for i in $(seq 0 $((total_num_jobs-1))) # i from 0 to 2070
# do
#     read_start=$((i * num_shards_per_job))
#     python /home/austinwang/vlm/data_processing/tfds_conversion.py \
#         --direct_running_mode $running_mode \
#         --direct_num_workers $num_workers \
#         --read_start $read_start &
#     sleep 1
# done

# wait




# # use gsutil cp to copy parquet, json, and tar files from index 00000 to 00400
# # from gs://us-central2-storage/tensorflow_datasets/downloads/manual/ to gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/downloads/manual/
# for i in {60..400}
# do
#     source_idx=$(printf "%05d" $i)
#     gsutil cp gs://us-central2-storage/tensorflow_datasets/downloads/manual/$source_idx* gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/downloads/manual/
# done
