import os
import math
import json
import logging
import argparse
import multiprocessing
import concurrent.futures

from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from google.cloud import storage
import tensorflow_datasets as tfds


logging.basicConfig(level=logging.INFO)

class CambrianDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = None  # This will be set dynamically in __init__
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="737k", description="Cambrian dataset with 737k samples"),
        tfds.core.BuilderConfig(name="10M", description="Cambrian dataset with ~10M samples"),
    ]

    def __init__(self, job_id=0, num_jobs=1, use_parallel=True, *args, **kwargs):
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.use_parallel = use_parallel
        self.storage_client = storage.Client()
        self.__class__.VERSION = tfds.core.Version(f'1.0.{job_id}')
        super().__init__(*args, **kwargs)

        # Initialize paths
        self.base_path = "/mnt/disks/storage/data/finetune_data"
        self.image_base_path = self.base_path
        self.dataset_path_737k = os.path.join(self.base_path, "jsons/737k.jsonl")
        self.local_folder_10M = "/home/austinwang/manual_cambrian_dataset"
        self.gcs_folder_10M = "tensorflow_datasets/tensorflow_datasets/downloads/manual_cambrian_dataset"

        logging.info(f"Initialized CambrianDataset with job_id={job_id}, num_jobs={num_jobs}, version={self.VERSION}")
        logging.info(f"Base path: {self.base_path}")
        logging.info(f"Image base path: {self.image_base_path}")

    def _info(self) -> tfds.core.DatasetInfo:
        logging.info("Calling _info method")
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Cambrian {self.builder_config.name} dataset (Batch {self.job_id + 1} of {self.num_jobs})",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Sequence(tfds.features.Tensor(shape=(), dtype=tf.string)),
                'conversations': tfds.features.Sequence({
                    'from': tfds.features.Text(),
                    'value': tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        logging.info("Calling _split_generators method")
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        logging.info(f"Calling _generate_examples for {self.builder_config.name}")
        if self.builder_config.name == "737k":
            yield from self._generate_737k_examples()
        else:  # "10M"
            yield from self._generate_10M_examples()

    def _generate_737k_examples(self):
        logging.info("Starting _generate_737k_examples")
        total_samples = sum(1 for _ in open(self.dataset_path_737k))
        samples_per_job = math.ceil(total_samples / self.num_jobs)
        start_sample = self.job_id * samples_per_job
        end_sample = min(start_sample + samples_per_job, total_samples)

        logging.info(f"Processing samples from {start_sample} to {end_sample}")

        counter = 0
        with open(self.dataset_path_737k, 'r') as f:
            for i, line in enumerate(tqdm(f, total=total_samples, desc=f"Processing 737k samples (Batch {self.job_id + 1}/{self.num_jobs})")):
                if start_sample <= i < end_sample:
                    sample = json.loads(line)
                    processed_sample = self._process_sample(sample)
                    if processed_sample:
                        yield counter, processed_sample
                        counter += 1

    def _generate_10M_examples(self):
        logging.info("Starting _generate_10M_examples")
        
        if not os.path.exists(self.local_folder_10M):
            logging.info(f"Local folder {self.local_folder_10M} does not exist. Downloading from GCS.")
            self._download_from_gcs(self.gcs_folder_10M, self.local_folder_10M)

        parquet_files = sorted([f for f in os.listdir(self.local_folder_10M) if f.endswith('.parquet')])
        total_files = len(parquet_files)
        files_per_job = math.ceil(total_files / self.num_jobs)
        start_file = self.job_id * files_per_job
        end_file = min(start_file + files_per_job, total_files)

        logging.info(f"Processing {end_file - start_file} files from {start_file} to {end_file}")
        logging.info(f"Total files: {total_files}, Files per job: {files_per_job}")

        files_to_process = parquet_files[start_file:end_file]

        if self.use_parallel:
            logging.info("Using parallel processing")
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [executor.submit(self._process_parquet_file, os.path.join(self.local_folder_10M, file)) for file in files_to_process]
                for future in concurrent.futures.as_completed(futures):
                    yield from future.result()
        else:
            logging.info("Using sequential processing")
            for file in tqdm(files_to_process, desc=f"Processing 10M samples (Batch {self.job_id + 1}/{self.num_jobs})"):
                yield from self._process_parquet_file(os.path.join(self.local_folder_10M, file))

    def _download_from_gcs(self, gcs_folder, local_folder):
        logging.info(f"Downloading files from GCS: {gcs_folder} to {local_folder}")
        os.makedirs(local_folder, exist_ok=True)
        bucket = self.storage_client.bucket("us-central2-storage")
        blobs = bucket.list_blobs(prefix=gcs_folder)
        for blob in tqdm(blobs, desc="Downloading files from GCS"):
            if blob.name.endswith('.parquet'):
                local_file_path = os.path.join(local_folder, os.path.basename(blob.name))
                blob.download_to_filename(local_file_path)

    def _process_parquet_file(self, file_path):
        logging.info(f"Processing Parquet file: {file_path}")
        table = pq.read_table(file_path)
        df = table.to_pandas()
        file_id = file_path.split('part_')[1].split('.parquet')[0]
        for counter, (_, row) in enumerate(df.iterrows()):
            sample = row.to_dict()
            processed_sample = self._process_sample(sample)
            sample_id = f"{file_id}_{counter}"
            if processed_sample:
                yield sample_id, processed_sample
            else:
                logging.warning(f"Error processing sample {sample_id}: {sample}")
                exit()

    def _process_sample(self, sample):
        try:
            processed_sample = {'image': []}  # Initialize with an empty list
            
            image_file = sample.get('image', None)
            if image_file and image_file not in ['', 'None', 'none', 'nan']:
                image_path = os.path.join(self.image_base_path, image_file)
                try:
                    with open(image_path, 'rb') as image_file:
                        processed_sample['image'] = [image_file.read()]  # Wrap in a list
                except FileNotFoundError:
                    logging.warning(f"Image file not found: {image_path}")
                except Exception as e:
                    logging.error(f"Error reading image {image_path}: {e}")
            
            conversations = sample.get('conversations', [])
            processed_conversations = self._process_conversations(conversations)

            if not processed_conversations:
                logging.warning(f"No valid conversations for sample")
                return None

            processed_sample['conversations'] = processed_conversations
            return processed_sample
        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            return None
        
    def _process_conversations(self, conversations):
        if isinstance(conversations, np.ndarray):
            conversations = conversations.tolist()
        
        if not isinstance(conversations, list) or len(conversations) < 2:
            return []

        processed_conversations = []
        for i in range(0, len(conversations) - 1, 2):
            human = conversations[i]
            gpt = conversations[i + 1]

            if not isinstance(human, dict) or not isinstance(gpt, dict): continue
            if 'from' not in human or 'value' not in human or 'from' not in gpt or 'value' not in gpt: continue
            if human['from'].lower() != 'human' or gpt['from'].lower() != 'gpt': continue

            processed_conversations.append(human)
            processed_conversations.append(gpt)

        return processed_conversations

def main(config, job_id, num_jobs, use_parallel, local_data_dir, gcs_data_dir, gcs_tfds):
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir

    builder = CambrianDataset(
        config=config,
        job_id=job_id,
        num_jobs=num_jobs,
        use_parallel=use_parallel,
        version=f"1.0.{job_id}", # Create a separate version for each job
        data_dir=data_dir
    )
    
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(num_shards=16),
    )
    
    logging.info(f"Dataset batch {job_id + 1}/{num_jobs} (version {builder.VERSION}) has been prepared and stored in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cambrian dataset")
    parser.add_argument("--config", type=str, choices=["737k", "10M"], required=True, help="Dataset configuration to use")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for this batch")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs")
    parser.add_argument("--use_parallel", type=bool, default=True, help="Whether to use parallel processing")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.job_id, args.num_jobs, args.use_parallel, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)