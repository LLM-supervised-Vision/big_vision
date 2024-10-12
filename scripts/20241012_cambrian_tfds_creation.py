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

class Cambrian737k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, job_id=0, num_jobs=1, *args, **kwargs):
        self.job_id = job_id
        self.num_jobs = num_jobs
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Cambrian 737k dataset (Batch {self.job_id + 1} of {self.num_jobs})",
            features=tfds.features.FeaturesDict({
                'id': tfds.features.Text(),
                'image': tfds.features.Image(),
                'conversations': tfds.features.Sequence({
                    'from': tfds.features.Text(),
                    'value': tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        cambrian_737k_path = "/mnt/disks/storage/data/finetune_data/jsons/737k.jsonl"
        image_base_path = "/mnt/disks/storage/data/finetune_data"

        with open(cambrian_737k_path, 'r') as f:
            total_samples = sum(1 for _ in f)

        samples_per_job = total_samples // self.num_jobs
        start_sample = self.job_id * samples_per_job
        end_sample = start_sample + samples_per_job if self.job_id < self.num_jobs - 1 else total_samples

        with open(cambrian_737k_path, 'r') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
                futures = []
                for i, line in enumerate(tqdm(f, total=total_samples, desc=f"Submitting tasks (Batch {self.job_id + 1}/{self.num_jobs})")):
                    if start_sample <= i < end_sample:
                        sample = json.loads(line)
                        future = executor.submit(self._process_sample, i, sample, image_base_path)
                        futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), 
                                   desc=f"Processing samples (Batch {self.job_id + 1}/{self.num_jobs})"):
                    result = future.result()
                    if result is not None:
                        yield result

    def _process_sample(self, index, sample, image_base_path):
        try:
            image_path = os.path.join(image_base_path, sample['image'])
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            return index, {
                'id': sample['id'],
                'image': image_data,
                'conversations': sample['conversations'],
            }
        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            return None

def main(job_id, num_jobs, local_data_dir, gcs_data_dir, gcs_tfds):
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    builder = Cambrian737k(job_id=job_id, num_jobs=num_jobs, data_dir=data_dir)
    builder.download_and_prepare()
    print(f"Dataset batch {job_id + 1}/{num_jobs} has been prepared and stored in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cambrian 737k dataset")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for this batch")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.job_id, args.num_jobs, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)