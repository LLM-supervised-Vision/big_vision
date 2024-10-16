import os
import logging
import tensorflow_datasets as tfds
import pyarrow.dataset as ds
import json
from tqdm import tqdm
import argparse
import concurrent.futures
import multiprocessing
import pyarrow.parquet as pq
from google.cloud import storage

class CambrianDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = None  # This will be set dynamically in __init__
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="737k", description="Cambrian dataset with 737k samples"),
        tfds.core.BuilderConfig(name="10M", description="Cambrian dataset with ~10M samples"),
    ]

    def __init__(self, job_id=0, num_jobs=1, num_samples_per_job=40000, *args, **kwargs):
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.num_samples_per_job = num_samples_per_job
        self.__class__.VERSION = tfds.core.Version(f'1.0.{job_id}')
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Cambrian {self.builder_config.name} dataset (Batch {self.job_id + 1} of {self.num_jobs})",
            features=tfds.features.FeaturesDict({
                'id': tfds.features.Text(),
                'image': tfds.features.Image(),
                'conversations': tfds.features.Sequence({
                    'from': tfds.features.Text(),
                    'value': tfds.features.Text(),
                }),
                'source': tfds.features.Text(),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        dataset_paths = {
            "737k": "/mnt/disks/storage/data/finetune_data/jsons/737k.jsonl",
            "10M": "gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets/cambrian_dataset/cambrian_dataset_10M.parquet",
        }
        dataset_path = dataset_paths[self.builder_config.name]
        image_base_path = "/mnt/disks/storage/data/finetune_data"

        start_sample = self.job_id * self.num_samples_per_job
        end_sample = start_sample + self.num_samples_per_job

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            futures = []
            for i, sample in enumerate(tqdm(self._load_samples(dataset_path, start_sample, end_sample), 
                                            total=self.num_samples_per_job, 
                                            desc=f"Processing samples (Batch {self.job_id + 1}/{self.num_jobs})")):
                future = executor.submit(self._process_sample, start_sample + i, sample, image_base_path)
                futures.append(future)

            processed_samples = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    yield result
                    processed_samples += 1

            logging.info(f"Processed {processed_samples} samples for job {self.job_id}")
            if processed_samples == 0:
                logging.warning(f"No samples were processed for job {self.job_id}")

    def _load_samples(self, dataset_path, start_sample, end_sample):
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if start_sample <= i < end_sample:
                        yield json.loads(line)
                    elif i >= end_sample:
                        break
        elif dataset_path.endswith('.parquet'):
            storage_client = storage.Client()
            bucket_name = dataset_path.split('/')[2]
            blob_name = '/'.join(dataset_path.split('/')[3:])
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            try:
                with blob.open("rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    row_groups_to_read = self._get_row_groups_to_read(parquet_file, start_sample, end_sample)

                    current_row = 0
                    for row_group in row_groups_to_read:
                        table = parquet_file.read_row_group(row_group)
                        for row in table.to_pylist():
                            if start_sample <= current_row < end_sample:
                                yield row
                            current_row += 1
                            if current_row >= end_sample:
                                return
            except Exception as e:
                logging.error(f"Error reading Parquet file: {e}")
                raise

    def _get_row_groups_to_read(self, parquet_file, start_sample, end_sample):
        row_groups_to_read = []
        current_row = 0
        for i in range(parquet_file.num_row_groups):
            row_group = parquet_file.metadata.row_group(i)
            next_row = current_row + row_group.num_rows
            if current_row <= end_sample and next_row > start_sample:
                row_groups_to_read.append(i)
            if next_row >= end_sample:
                break
            current_row = next_row
        return row_groups_to_read

    def _process_sample(self, index, sample, image_base_path):
        try:
            image_file = sample.get('image')
            if not image_file or image_file in ['', 'None', 'none', 'nan']:
                logging.warning(f"Invalid or missing image for sample {index}")
                return None

            image_path = os.path.join(image_base_path, image_file)
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            conversations = sample.get('conversations', [])
            if not isinstance(conversations, list) or len(conversations) < 2:
                logging.warning(f"Invalid conversations format for sample {index}")
                return None

            processed_conversations = []
            for i in range(0, len(conversations) - 1, 2):
                human = conversations[i]
                gpt = conversations[i + 1]

                if not isinstance(human, dict) or not isinstance(gpt, dict):
                    logging.warning(f"Invalid conversation entry format for sample {index}")
                    continue

                if 'from' not in human or 'value' not in human or 'from' not in gpt or 'value' not in gpt:
                    logging.warning(f"Missing 'from' or 'value' in conversation for sample {index}")
                    continue

                if human['from'].lower() != 'human' or gpt['from'].lower() != 'gpt':
                    logging.warning(f"Incorrect conversation order for sample {index}")
                    continue

                processed_conversations.append(human)
                processed_conversations.append(gpt)

            if not processed_conversations:
                logging.warning(f"No valid conversations for sample {index}")
                return None

            return index, {
                'id': sample.get('id', str(index)),
                'image': image_data,
                'conversations': processed_conversations,
                'source': sample.get('source', ''),
            }
        except Exception as e:
            logging.error(f"Error processing sample {index}: {e}")
            return None

def main(config, job_id, num_jobs, num_samples_per_job, local_data_dir, gcs_data_dir, gcs_tfds):
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    try:
        builder = CambrianDataset(
            config=config, 
            job_id=job_id, 
            num_jobs=num_jobs, 
            num_samples_per_job=num_samples_per_job, 
            version=f"1.0.{job_id}", 
            data_dir=data_dir
        )
        builder.download_and_prepare(
            download_config=tfds.download.DownloadConfig(num_shards=1),
        )
        logging.info(f"Dataset batch {job_id + 1}/{num_jobs} has been prepared and stored in {data_dir}")
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cambrian dataset")
    parser.add_argument("--config", type=str, choices=["737k", "10M"], required=True, help="Dataset configuration to use")
    parser.add_argument("--job_id", type=int, required=True, help="Job ID for this batch")
    parser.add_argument("--num_jobs", type=int, required=True, help="Total number of jobs")
    parser.add_argument("--num_samples_per_job", type=int, required=True, help="Number of samples per job")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", action="store_true", help="Store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.job_id, args.num_jobs, args.num_samples_per_job, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)