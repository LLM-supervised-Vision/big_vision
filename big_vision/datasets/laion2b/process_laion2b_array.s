#!/bin/bash
#SBATCH --job-name=laion
#SBATCH --output=/scratch/zy2091/austin_big_vision/big_vision/datasets/laion2b/logs/output_.out
#SBATCH --error=/scratch/zy2091/austin_big_vision/big_vision/datasets/laion2b/logs/error_.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=2:00:00

# example usage: sbatch --array=1-10 process_laion2b_array.sh

# check the pwd
pwd

# create the dataset builder script for the subset with the task ID
python python laion2b_dataset_builder_template.py --version 1.0.$SLURM_ARRAY_TASK_ID --task_id $SLURM_ARRAY_TASK_ID
# change the directory to the newly created dataset builder
cd laion2b_$SLURM_ARRAY_TASK_ID
# run the dataset builder
tfds build --overwrite --data_dir /scratch/zy2091/tensorflow_datasets/laion2b --manual_dir /scratch/work/public/ml-datasets/laion2B-en-data/ --download_config download_config.json

# upload the dataset to the google cloud bucket
python ../../../tools/tfds_datasets/upload_dir.py --bucket_name us-central2-storage --source_directory /scratch/zy2091/tensorflow_datasets/laion2b/laion2b_$SLURM_ARRAY_TASK_ID --destination_blob us-central2-storage/tensorflow_datasets/laion2b

# remove the dataset
rm -rf /scratch/zy2091/tensorflow_datasets/laion2b/laion2b_$SLURM_ARRAY_TASK_ID

# remove the dataset builder
cd ..
rm -rf laion2b_$SLURM_ARRAY_TASK_ID

