# !/bin/bash

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in merge_segments.sh. Exiting gracefully."
    # Kill all child processes
    pkill -P $$
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT

# gsutil -m mv -r gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/* gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/dataset_info.json gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/features.json gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/nsfw.labels.txt gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/

export TFDS_NAME=cc12m # laion400m/images
export SPLIT=train
export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets
export FINAL_VERSION_ID=6
export NUM_WORKERS=1

cd ~/austin_big_vision/big_vision/tools/tfds_datasets
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Running worker $i"
    python merge_segments.py \
        --tfds_name $TFDS_NAME \
        --split $SPLIT \
        --tfds_data_dir $TFDS_DATA_DIR \
        --final_version_id $FINAL_VERSION_ID \
        --num_workers $NUM_WORKERS \
        --worker_id $i \
        --debug True &
    sleep 1
done

wait