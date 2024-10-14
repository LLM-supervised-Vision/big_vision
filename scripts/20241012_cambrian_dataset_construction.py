import os
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from PIL import Image
from tqdm import tqdm
import argparse
import concurrent.futures
import multiprocessing
import time

class CambrianDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = None  # This will be set dynamically in __init__
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="737k", description="Cambrian dataset with 737k samples"),
        tfds.core.BuilderConfig(name="10M", description="Cambrian dataset with ~10M samples"),
    ]

    def __init__(self, job_id=0, num_jobs=1, *args, **kwargs):
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.__class__.VERSION = tfds.core.Version(f'1.0.{job_id}')
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Cambrian {self.builder_config.name} dataset (Batch {self.job_id + 1} of {self.num_jobs})",
            features=tfds.features.FeaturesDict({
                'id': tfds.features.Text(),
                'image': tfds.features.Image(),
                'conversations': tfds.features.Sequence(tfds.features.Text()),
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
            "10M": "/mnt/disks/storage/data/finetune_data/clean_9784k.json"
        }
        dataset_path = dataset_paths[self.builder_config.name]
        image_base_path = "/mnt/disks/storage/data/finetune_data"

        # Count total samples
        total_samples = self._count_samples(dataset_path)

        samples_per_job = total_samples // self.num_jobs
        start_sample = self.job_id * samples_per_job
        end_sample = start_sample + samples_per_job if self.job_id < self.num_jobs - 1 else total_samples

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            futures = []
            for i, sample in enumerate(tqdm(self._load_samples(dataset_path), total=total_samples, desc=f"Submitting tasks (Batch {self.job_id + 1}/{self.num_jobs})")):
                if start_sample <= i < end_sample:
                    future = executor.submit(self._process_sample, i, sample, image_base_path)
                    futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), 
                               desc=f"Processing samples (Batch {self.job_id + 1}/{self.num_jobs})"):
                result = future.result()
                if result is not None:
                    yield result

    def _count_samples(self, dataset_path):
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                return sum(1 for _ in f)
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                return len(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

    def _load_samples(self, dataset_path):
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                for line in f:
                    yield json.loads(line)
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                for sample in json.load(f):
                    yield sample
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

    def _process_sample(self, index, sample, image_base_path):
        try:
            image_file = sample.get('image')
            if not image_file or image_file in ['', 'None', 'none', 'nan']:
                print(f"Warning: Invalid or missing image for sample {index}")
                return None

            image_path = os.path.join(image_base_path, image_file)
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            conversations = sample.get('conversations', [])
            processed_conversations = []

            if not isinstance(conversations, list) or len(conversations) < 2:
                print(f"Warning: Invalid conversations format for sample {index}")
                return None

            for i in range(0, len(conversations) - 1, 2):
                human = conversations[i]
                gpt = conversations[i + 1]

                if not isinstance(human, dict) or not isinstance(gpt, dict):
                    print(f"Warning: Invalid conversation entry format for sample {index}")
                    continue

                if 'from' not in human or 'value' not in human or 'from' not in gpt or 'value' not in gpt:
                    print(f"Warning: Missing 'from' or 'value' in conversation for sample {index}")
                    continue

                if human['from'].lower() != 'human' or gpt['from'].lower() != 'gpt':
                    print(f"Warning: Incorrect conversation order for sample {index}")
                    continue

                human_value = human['value']
                gpt_value = gpt['value']

                if not isinstance(human_value, str) or not isinstance(gpt_value, str):
                    print(f"Warning: 'value' is not a string for sample {index}")
                    continue

                combined_conversation = f"human: {human_value} gpt: {gpt_value}"
                # print(f"index: {index}, combined_conversation: {combined_conversation}")
                processed_conversations.append(combined_conversation)

            if not processed_conversations:
                print(f"Warning: No valid conversations for sample {index}")
                return None

            return index, {
                'id': sample.get('id', str(index)),
                'image': image_data,
                'conversations': processed_conversations,
                'source': sample.get('source', ''),
            }
        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            return None

def main(config, job_id, num_jobs, local_data_dir, gcs_data_dir, gcs_tfds):
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    builder = CambrianDataset(config=config, job_id=job_id, num_jobs=num_jobs, version=f"1.0.{job_id}", data_dir=data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(num_shards=1),
    )
    print(f"Dataset batch {job_id + 1}/{num_jobs} has been prepared and stored in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cambrian dataset")
    parser.add_argument("--config", type=str, choices=["737k", "10M"], required=True, help="Dataset configuration to use")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for this batch")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.job_id, args.num_jobs, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)