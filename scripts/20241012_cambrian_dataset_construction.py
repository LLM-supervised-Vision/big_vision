import os
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from google.cloud import storage
import pyarrow.parquet as pq
import io
from PIL import Image
from tqdm import tqdm
import argparse
import concurrent.futures
import multiprocessing
import math
import logging

logging.basicConfig(level=logging.INFO)

class CambrianDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="737k", description="Cambrian dataset with 737k samples"),
        tfds.core.BuilderConfig(name="10M", description="Cambrian dataset with ~10M samples"),
    ]

    def __init__(self, job_id=0, num_jobs=1, use_parallel=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.use_parallel = use_parallel
        self.storage_client = storage.Client()
        logging.info(f"Initialized CambrianDataset with job_id={job_id}, num_jobs={num_jobs}")

    def _info(self) -> tfds.core.DatasetInfo:
        logging.info("Calling _info method")
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
        dataset_path = "/mnt/disks/storage/data/finetune_data/jsons/737k.jsonl"
        image_base_path = "/mnt/disks/storage/data/finetune_data"

        total_samples = sum(1 for _ in open(dataset_path))
        samples_per_job = total_samples // self.num_jobs
        start_sample = self.job_id * samples_per_job
        end_sample = start_sample + samples_per_job if self.job_id < self.num_jobs - 1 else total_samples

        logging.info(f"Processing samples from {start_sample} to {end_sample}")

        with open(dataset_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=total_samples, desc=f"Processing 737k samples (Batch {self.job_id + 1}/{self.num_jobs})")):
                if start_sample <= i < end_sample:
                    sample = json.loads(line)
                    processed_sample = self._process_sample(sample, image_base_path)
                    if processed_sample:
                        yield i, processed_sample

    def _generate_10M_examples(self):
        logging.info("Starting _generate_10M_examples")
        local_folder = "/home/austinwang/manual_cambrian_dataset"
        gcs_folder = "tensorflow_datasets/tensorflow_datasets/downloads/manual_cambrian_dataset"
        
        if not os.path.exists(local_folder):
            logging.info(f"Local folder {local_folder} does not exist. Downloading from GCS.")
            self._download_from_gcs(gcs_folder, local_folder)

        parquet_files = sorted([f for f in os.listdir(local_folder) if f.endswith('.parquet')])
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
                futures = [executor.submit(self._process_parquet_file, os.path.join(local_folder, file)) for file in files_to_process]
                for future in concurrent.futures.as_completed(futures):
                    yield from future.result()
        else:
            logging.info("Using sequential processing")
            for file in tqdm(files_to_process, desc=f"Processing 10M samples (Batch {self.job_id + 1}/{self.num_jobs})"):
                yield from self._process_parquet_file(os.path.join(local_folder, file))

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
        for _, row in df.iterrows():
            sample = row.to_dict()
            processed_sample = self._process_sample(sample)
            if processed_sample:
                yield sample['id'], processed_sample

    def _process_sample(self, sample, image_base_path=None):
        try:
            image_file = sample.get('image')
            if not image_file or image_file in ['', 'None', 'none', 'nan']:
                logging.warning(f"Invalid or missing image for sample {sample.get('id')}")
                return None

            if image_base_path:
                image_path = os.path.join(image_base_path, image_file)
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()
            else:
                image_data = self._get_image_data(image_file)

            conversations = sample.get('conversations', [])
            processed_conversations = self._process_conversations(conversations)

            if not processed_conversations:
                logging.warning(f"No valid conversations for sample {sample.get('id')}")
                return None

            return {
                'id': str(sample.get('id')),
                'image': image_data,
                'conversations': processed_conversations,
                'source': sample.get('source', ''),
            }
        except Exception as e:
            logging.error(f"Error processing sample {sample.get('id')}: {e}")
            return None

    def _get_image_data(self, image_path):
        bucket = self.storage_client.bucket("us-central2-storage")
        blob = bucket.blob(image_path)
        try:
            return blob.download_as_bytes()
        except Exception as e:
            logging.error(f"Error downloading image {image_path}: {e}")
            return None

    def _process_conversations(self, conversations):
        if not isinstance(conversations, list) or len(conversations) < 2:
            return []

        processed_conversations = []
        for i in range(0, len(conversations) - 1, 2):
            human = conversations[i]
            gpt = conversations[i + 1]

            if not isinstance(human, dict) or not isinstance(gpt, dict):
                continue

            if 'from' not in human or 'value' not in human or 'from' not in gpt or 'value' not in gpt:
                continue

            if human['from'].lower() != 'human' or gpt['from'].lower() != 'gpt':
                continue

            processed_conversations.append(human)
            processed_conversations.append(gpt)

        return processed_conversations

def main(config, job_id, num_jobs, use_parallel, local_data_dir, gcs_data_dir, gcs_tfds):
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    builder = CambrianDataset(config=config, job_id=job_id, num_jobs=num_jobs, use_parallel=use_parallel, data_dir=data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            num_shards=1,
            # download_mode=tfds.download.GenerateMode.FORCE_REDOWNLOAD,
        ),
        # download_and_prepare_kwargs={'force_prepare': True}
    )
    logging.info(f"Dataset batch {job_id + 1}/{num_jobs} has been prepared and stored in {data_dir}")

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