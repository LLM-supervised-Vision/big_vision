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

export TFDS_NAME=cc12m # laion400m/images
export SPLIT=train
export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets
export FINAL_VERSION_ID=6
export NUM_WORKERS=20

# Check if the source directory exists in gcloud storage
if [ $(gsutil ls $TFDS_DATA_DIR/$TFDS_NAME/1.0.0_source | wc -l) -eq 0 ]; then

    echo "Creating $TFDS_DATA_DIR/$TFDS_NAME/1.0.0_source"
    gsutil -m cp -r $TFDS_DATA_DIR/$TFDS_NAME/1.0.0/* $TFDS_DATA_DIR/$TFDS_NAME/1.0.0_source/
    gsutil -m rm $TFDS_DATA_DIR/$TFDS_NAME/1.0.0/*tfrecord*

    echo "Creating $TFDS_DATA_DIR/$TFDS_NAME/backup for backup"
    gsutil -m cp -r $TFDS_DATA_DIR/$TFDS_NAME/* $TFDS_DATA_DIR/$TFDS_NAME/backup/
fi

cd ~/austin_big_vision/big_vision/tools/tfds_datasets
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Running worker $i"
    python merge_segments.py \
        --tfds_name $TFDS_NAME \
        --split $SPLIT \
        --tfds_data_dir $TFDS_DATA_DIR \
        --final_version_id $FINAL_VERSION_ID \
        --num_workers $NUM_WORKERS \
        --worker_id $i &
    sleep 1
done

# wait