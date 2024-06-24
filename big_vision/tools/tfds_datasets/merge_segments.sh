# !/bin/bash

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in renaming_shards.sh. Exiting gracefully."
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT

# gsutil -m mv -r gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/* gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/dataset_info.json gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/features.json gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/
# gsutil cp gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0_source/nsfw.labels.txt gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/

num_workers=100
final_version_id=2070
for i in $(seq 0 $((num_workers - 1))); do
    echo "Running worker $i"
    python /home/austinwang/vlm/data_processing/merge_segments.py \
        --destination_dir gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/ \
        --final_version_id $final_version_id \
        --num_workers $num_workers \
        --worker_id $i &
    sleep 1
done

wait