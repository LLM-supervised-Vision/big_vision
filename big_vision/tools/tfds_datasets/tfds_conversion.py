import os
import json
from absl import app, flags, logging
from google.cloud import storage
import tensorflow_datasets as tfds
from apache_beam.options.pipeline_options import DirectOptions

FLAGS = flags.FLAGS
flags.DEFINE_string('direct_running_mode', None, 'Direct running mode')
flags.DEFINE_integer('direct_num_workers', None, 'Number of workers')
flags.DEFINE_integer('read_start', None, 'Read start')


def main(argv):
  logging.info(f"Direct running mode: {FLAGS.direct_running_mode}")
  logging.info(f"Number of workers: {FLAGS.direct_num_workers}")
  logging.info(f"Read start: {FLAGS.read_start}")

  # Download and prepare the dataset
  download_and_prepare_kwargs = {
    "download_dir": None,
    "download_config": tfds.download.DownloadConfig(
      beam_options=DirectOptions(
        runner="DirectRunner",
        direct_num_workers=FLAGS.direct_num_workers,
        direct_running_mode=FLAGS.direct_running_mode, # "multi_processing", "in_memory", "multi_threading"
      ),
      download_mode=tfds.core.download.GenerateMode.CONTINUE_DOWNLOAD,
      read_start=FLAGS.read_start,
    ),
    "file_format": None,
  }
  tfds.load(name="laion400m/images", download=True,download_and_prepare_kwargs=download_and_prepare_kwargs)

  version_number = int(FLAGS.read_start/20)
  # Read the metadata_info.json to get the total number of samples
  destination_dir = f"gs://us-central2-storage/tensorflow_datasets/laion400m/images/1.0.{version_number}/"
  # Get the bucket
  client = storage.Client()
  bucket_name = destination_dir.split("/")[2] # us-central2-storage
  logging.info(f"Bucket name: {bucket_name}") # DEBUG
  bucket = client.get_bucket(bucket_name)
  # Get blobs in the source directory
  prefix = "/".join(destination_dir.split("/")[3:]) # tensorflow_datasets/laion400m/images/1.0.0/
  logging.info(f"Prefix: {prefix}") # DEBUG

  # check the metadata in folder 1.0.{version_number}
  folder_blob = bucket.get_blob(f"{prefix}dataset_info.json")
  logging.info(f"Folder blob: {folder_blob}") # DEBUG
  if folder_blob:
      # get the total number of samples
      dataset_info = json.loads(folder_blob.download_as_string())
      total_num_samples = sum([int(shard_length) for shard_length in dataset_info["splits"][0]["shardLengths"]])
      logging.info(f"Total number of samples: {total_num_samples}")

      # calculate the success rate
      success_rate = total_num_samples/200000 * 100
      logging.info(f"version {version_number} success rate: {success_rate}%")

      # record the version number and its success rate in a dictionary file stored in local
      # modify the success rate if the version number already exists
      success_rate_dict = {}
      key = f"1.0.{version_number}"
      success_rate_dict[key] = success_rate
      success_rate_file = "success_rate.json"
      if os.path.exists(success_rate_file) and os.path.getsize(success_rate_file) > 0:
          with open(success_rate_file, "r") as f:
              success_rate_dict = json.load(f)
          if key in success_rate_dict:
              success_rate_dict[key] = success_rate
          else:
              success_rate_dict.update({key: success_rate})
      with open(success_rate_file, "w") as f:
         json.dump(success_rate_dict, f, indent=4)

if __name__ == "__main__":
  app.run(main)
