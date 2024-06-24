#!/bin/bash

# chunks: 1923-7923, 7923-13923, 13923-19923, 19923-25923, 25923-31923, 31923-37923, 37923-43923, 43923-49923, 49923-55923, 55923-61923, 61923-62917 
# start_idx=62000
# end_idx=62917

# Function to handle Ctrl+C signal
function handle_ctrl_c {
    echo "Ctrl+C detected in renaming_shards.sh. Exiting gracefully."
    exit 1
}

# Trap Ctrl+C signal
trap handle_ctrl_c SIGINT


# Check if start and end indices are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start_idx> <end_idx> <script_idx>"
    exit 1
fi

# Assign command-line arguments to start_idx and end_idx variables
start_idx=$1
end_idx=$2
script_idx=$3

echo "Start index: $start_idx"
echo "End index: $end_idx"
echo "Script index: $script_idx"

# Counter variable
i=0

# Loop through files in the specified directory
gsutil ls gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/* | while read -r file; do
    if [[ "$file" == *"laion400m-train.tfrecord"* && "$file" == *"62917"* ]]; then
        echo "$script_idx: Found file: $file"
        ((i++))
    fi
    # Check if file name starts with "laion400m-train.tfrecord"
    if [[ "$file" == *"laion400m-train.tfrecord"* && "$file" == *"66256" ]]; then

        if [ $i -ge $start_idx ] && [ $i -lt $end_idx ]; then
            # Extract the file name
            filename=$(basename "$file")
            echo "$script_idx: $filename"
            # Generate 5-digit padded index
            padded_idx=$(printf "%05d" $i)
            echo "$script_idx: $padded_idx"
            # Rename the file
            new_filename="laion400m-train.tfrecord-$padded_idx-of-62917"
            echo "$script_idx: $new_filename"
            gsutil mv "$file" gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.0/$new_filename
        else
            echo "$script_idx: Skipping file: $file"
        fi
        ((i++))
    fi
done
