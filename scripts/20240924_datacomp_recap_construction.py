import os
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import argparse

class DatacompRecap(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="10", description="Dataset with 10 samples"),
        tfds.core.BuilderConfig(name="1k", description="Dataset with 1,000 samples"),
        tfds.core.BuilderConfig(name="10k", description="Dataset with 10,000 samples"),
        tfds.core.BuilderConfig(name="1M", description="Dataset with 1,000,000 samples"),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"DataComp-Recap-1B dataset with {self.builder_config.name} samples",
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
            "1k": 1000,
            "10k": 10000,
            "1M": 1000000,
        }
        num_samples = config_to_samples[self.builder_config.name]
        
        ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)
        
        for i, sample in enumerate(tqdm(ds.take(num_samples), total=num_samples, desc="Generating samples")):
            image_data, width, height = self._download_image(sample['url'])
            if image_data is not None:
                yield i, {
                    'image': image_data,
                    'url': sample['url'],
                    're_caption': sample['re_caption'],
                    'org_caption': sample['org_caption'],
                    'width': width,
                    'height': height,
                }

    def _download_image(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Convert to RGB to ensure consistency
            width, height = img.size
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue(), width, height
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None, None, None
        
def main(config_name, local_data_dir, gcs_data_dir, gcs_tfds):
    # Create the builder
    data_dir = gcs_data_dir if gcs_tfds else local_data_dir
    builder = DatacompRecap(config=config_name, data_dir=data_dir)

    # Prepare the dataset
    builder.download_and_prepare()
    print(f"Dataset has been prepared and stored in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DataComp-Recap-1B dataset")
    parser.add_argument("--config", type=str, choices=["10", "1k", "10k", "1M"], default="10", help="Configuration to use")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)