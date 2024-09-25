import os
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import argparse
import concurrent.futures
import multiprocessing
import time
from huggingface_hub.utils import HfHubHTTPError

class DatacompRecap(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="10", description="Dataset with 10 samples"),
        tfds.core.BuilderConfig(name="100", description="Dataset with 100 samples"),
        tfds.core.BuilderConfig(name="1k", description="Dataset with 1,000 samples"),
        tfds.core.BuilderConfig(name="10k", description="Dataset with 10,000 samples"),
        tfds.core.BuilderConfig(name="1M", description="Dataset with 1,000,000 samples"),
        tfds.core.BuilderConfig(name="10M", description="Dataset with 10,000,000 samples"),
        tfds.core.BuilderConfig(name="100M", description="Dataset with 100,000,000 samples"),
    ]

    def __init__(self, job_id=0, num_jobs=1, *args, **kwargs):
        self.job_id = job_id
        self.num_jobs = num_jobs
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"DataComp-Recap-1B dataset with {self.builder_config.name} samples (Batch {self.job_id + 1} of {self.num_jobs})",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'url': tfds.features.Text(),
                're_caption': tfds.features.Text(),
                'org_caption': tfds.features.Text(),
                'width': tf.int64,
                'height': tf.int64,
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        config_to_samples = {
            "10": 10,
            "100": 100,
            "1k": 1000,
            "10k": 10000,
            "1M": 1000000,
            "10M": 10000000,
            "100M": 100000000,
        }
        total_samples = config_to_samples[self.builder_config.name]
        samples_per_job = total_samples // self.num_jobs
        start_sample = self.job_id * samples_per_job
        end_sample = start_sample + samples_per_job if self.job_id < self.num_jobs - 1 else total_samples

        ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            futures = []
            for i, sample in enumerate(tqdm(self._resilient_take(ds.skip(start_sample), end_sample - start_sample), 
                                            total=end_sample - start_sample, 
                                            desc=f"Submitting tasks (Batch {self.job_id + 1}/{self.num_jobs})")):
                future = executor.submit(self._process_sample, start_sample + i, sample)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), 
                               desc=f"Processing samples (Batch {self.job_id + 1}/{self.num_jobs})"):
                result = future.result()
                if result is not None:
                    yield result

    def _resilient_take(self, dataset, n):
        """Resilient version of dataset.take() that handles API errors."""
        count = 0
        while count < n:
            try:
                for item in dataset:
                    yield item
                    count += 1
                    if count >= n:
                        return
            except HfHubHTTPError as e:
                print(f"Encountered API error: {e}. Retrying in 5 seconds...")
                time.sleep(5)

    def _process_sample(self, index, sample):
        image_data, width, height = self._download_image(sample['url'], index)
        if image_data is not None:
            return index, {
                'image': image_data,
                'url': sample['url'],
                're_caption': sample['re_caption'],
                'org_caption': sample['org_caption'],
                'width': width,
                'height': height,
            }
        return None

    def _download_image(self, url, index):
        max_retries = 2
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Convert to RGB to ensure consistency
                width, height = img.size
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG')
                return img_byte_arr.getvalue(), width, height
            except Exception as e:
                if attempt < max_retries - 1:
                    # print(f"index {index} Error downloading {url}: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"index {index} Error downloading {url}: {e}. Max retries = {max_retries} exceeded.")
        return None, None, None

def main(config_name, job_id, num_jobs, local_data_dir, gcs_data_dir, gcs_tfds):
    # Create the builder
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    builder = DatacompRecap(config=config_name, job_id=job_id, num_jobs=num_jobs, 
                            version=f"1.0.{job_id}", data_dir=data_dir)

    # Prepare the dataset
    builder.download_and_prepare()
    print(f"Dataset batch {job_id + 1}/{num_jobs} has been prepared and stored in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DataComp-Recap-1B dataset")
    parser.add_argument("--config", type=str, choices=["10", "100", "1k", "10k", "1M", "10M", "100M"], default="10", help="Configuration to use")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for this batch")
    parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.job_id, args.num_jobs, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)