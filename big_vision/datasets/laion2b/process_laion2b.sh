#!/bin/bash
#SBATCH --job-name=test_laion
#SBATCH --output=/scratch/zy2091/austin_big_vision/big_vision/datasets/laion2b/output.log
#SBATCH --error=/scratch/zy2091/austin_big_vision/big_vision/datasets/laion2b/error.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=12:00:00

# check the pwd
pwd

singularity exec /scratch/zy2091/bv.simg bash -c "
source ~/.bashrc;
cd /scratch/zy2091/austin_big_vision/big_vision/datasets/laion2b;
tfds build --overwrite --max_examples_per_split 10000000 --data_dir gs://us-central2-storage/tensorflow_datasets --manual_dir /scratch/work/public/ml-datasets/laion2B-en-data/ --max_shard_size_mb 32768
"
