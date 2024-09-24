import os
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import argparse

def download_image(url):
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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_example(sample):
    image_data, width, height = download_image(sample['url'])
    if image_data is None:
        return None
    
    feature = {
        'image': _bytes_feature(image_data),
        'url': _bytes_feature(sample['url'].encode('utf-8')),
        're_caption': _bytes_feature(sample['re_caption'].encode('utf-8')),
        'org_caption': _bytes_feature(sample['org_caption'].encode('utf-8')),
        'width': _int64_feature(width),
        'height': _int64_feature(height),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecord(dataset, output_file, num_samples):
    # Ensure the directory exists
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping conversion.")
        return
    # Write to TFRecord file
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, sample in enumerate(tqdm(dataset.take(num_samples))):
            tf_example = create_example(sample)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
            if i >= num_samples - 1:
                break

class DatacompRecap(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="100", description="Dataset with 100 samples"),
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
            'train': self._generate_examples(os.path.join(dl_manager.manual_dir, f"datacomp_recap_{self.builder_config.name}.tfrecord")),
        }

    def _generate_examples(self, filepath):
        dataset = tf.data.TFRecordDataset(filepath)
        for i, raw_record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feature = example.features.feature
            yield i, {
                'image': feature['image'].bytes_list.value[0],
                'url': feature['url'].bytes_list.value[0].decode('utf-8'),
                're_caption': feature['re_caption'].bytes_list.value[0].decode('utf-8'),
                'org_caption': feature['org_caption'].bytes_list.value[0].decode('utf-8'),
                'width': feature['width'].int64_list.value[0],
                'height': feature['height'].int64_list.value[0],
            }

def main(config_name, local_data_dir, gcs_data_dir, gcs_tfds):
    # Map config names to number of samples
    config_to_samples = {
        "100": 100,
        "1k": 1000,
        "10k": 10000,
        "1M": 1000000,
    }
    num_samples = config_to_samples[config_name]

    # Step 1: Construct TFRecord files
    ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)
    tfrecord_file = os.path.join(local_data_dir, "downloads", "manual", f"datacomp_recap_{config_name}.tfrecord")
    convert_to_tfrecord(ds, tfrecord_file, num_samples)

    # Step 2: Construct TFDS dataset
    # Create the builder
    builder = DatacompRecap(config=config_name, data_dir=gcs_data_dir if gcs_tfds else local_data_dir)
    
    # Prepare the dataset
    builder.download_and_prepare()

    print(f"Dataset has been prepared and stored in {gcs_data_dir if gcs_tfds else local_data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DataComp-Recap-1B dataset")
    parser.add_argument("--config", type=str, choices=["100", "1k", "10k", "1M"], default="100", help="Configuration to use")
    parser.add_argument("--local_data_dir", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://us-central2-storage/tensorflow_datasets/tensorflow_datasets", help="GCS path")
    parser.add_argument("--gcs_tfds", type=bool, default=False, help="Whether to store the TFDS dataset in GCS")
    args = parser.parse_args()

    main(args.config, args.local_data_dir, args.gcs_data_dir, args.gcs_tfds)