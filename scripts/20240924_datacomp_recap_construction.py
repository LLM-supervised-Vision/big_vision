import os
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import argparse
# from google.cloud import storage

# Ensure you have the necessary permissions and have authenticated with Google Cloud

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

class DataCompRecap1B(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="DataComp-Recap-1B dataset",
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
        import pdb; pdb.set_trace()
        return {
            'train': self._generate_examples(self.tfrecord_file),
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

# def upload_to_gcs(local_path, bucket_name, gcs_path):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(gcs_path)
#     blob.upload_xfrom_filename(local_path)
#     print(f"File {local_path} uploaded to {gcs_path}.")

def main(num_samples, local_storage_path, gcs_bucket, gcs_path):
    # Step 1: Construct TFRecord files
    ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="train", streaming=True)
    tfrecord_file = os.path.join(local_storage_path, f"recap_datacomp_{num_samples}.tfrecord")
    convert_to_tfrecord(ds, tfrecord_file, num_samples)

    # Step 2: Construct TFDS dataset
    DataCompRecap1B.tfrecord_file = tfrecord_file  # Set the TFRecord file path
    builder = DataCompRecap1B(data_dir=local_storage_path)
    builder.download_and_prepare()

    # # Step 3: Upload to Google Cloud Storage
    # gcs_tfrecord_path = os.path.join(gcs_path, f"recap_datacomp_{num_samples}.tfrecord")
    # upload_to_gcs(tfrecord_file, gcs_bucket, gcs_tfrecord_path)

    # # Upload TFDS files
    # tfds_dir = os.path.join(local_storage_path, "datacomp_recap_1b")
    # for root, _, files in os.walk(tfds_dir):
    #     for file in files:
    #         local_file_path = os.path.join(root, file)
    #         relative_path = os.path.relpath(local_file_path, local_storage_path)
    #         gcs_file_path = os.path.join(gcs_path, relative_path)
    #         upload_to_gcs(local_file_path, gcs_bucket, gcs_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DataComp-Recap-1B dataset")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to process")
    parser.add_argument("--local_storage_path", type=str, default="/home/austinwang/tensorflow_datasets", help="Local storage path")
    parser.add_argument("--gcs_bucket", type=str, default="us-central2-storage", help="GCS bucket name")
    parser.add_argument("--gcs_path", type=str, default="tensorflow_datasets/tensorflow_datasets/datacomp_recap", help="GCS path")
    args = parser.parse_args()

    main(args.num_samples, args.local_storage_path, args.gcs_bucket, args.gcs_path)