from string import Template
import argparse
import os

def generate_script(version, task_id):

  template_code = '''
"""laion2b dataset."""

from etils import epath

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets as tfds

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile

_HOMEPAGE = 'https://laion.ai/blog/laion-5b/'

_TASK_ID = ${task_id}
_NUM_SHARDS = 232320 # for testing, use 1. for full dataset, use 232320.
_MISSING_SIMILARITY_VALUE = -1.0
_NSFW_MISSING_TAG = 'UNTAGGED'
_NSFW_TAGS = ('UNLIKELY', 'UNSURE', 'NSFW', _NSFW_MISSING_TAG)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for laion2b dataset."""

  VERSION = tfds.core.Version('${version}')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
  Refer to "Download" section on {_HOMEPAGE}
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
          'image': tfds.features.Image(doc='image'),
          'caption': tfds.features.Text(doc='HTML alt-text attribute'),
          'nsfw': tfds.features.ClassLabel(
              names=_NSFW_TAGS,
              doc=(
                  'NSFW tag (detected with CLIP). Incohesive and missing tags'
                  f' are replaced with {_NSFW_MISSING_TAG}'
              ),
          ),
          'similarity': tfds.features.Scalar(
              tf.float64,
              doc=tfds.features.Documentation(
                  desc=(
                      'cosine similarity score between the text and image '
                      'embedding. Missing values default to '
                      f'{_MISSING_SIMILARITY_VALUE}'
                  ),
                  value_range='[0.0, 1.0]',
              ),
          ),
          'license': tfds.features.Text(
              doc='type of Creative Commons license (if applicable)'
          ),
          'url': tfds.features.Text(doc='image URL'),
          'original_width': tfds.features.Scalar(
              tf.int32, doc='original width of the image'
          ),
          'original_height': tfds.features.Scalar(
              tf.int32, doc='original height of the image'
          ),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage=_HOMEPAGE,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples(dl_manager),
    }

  def _generate_examples(
      self,
      dl_manager: tfds.download.DownloadManager,
  ):
    for shard_idx in range(_NUM_SHARDS):
      for key, example in self._generate_examples_one_shard(dl_manager, shard_idx):
        yield key, example
  
  def _generate_examples_one_shard(
      self,
      dl_manager: tfds.download.DownloadManager,
      shard_idx: int,
  ):
    pd = tfds.core.lazy_imports.pandas

    img_archive_path = dl_manager.manual_dir / f'{shard_idx:05d}.tar'
    metadata_path = dl_manager.manual_dir / f'{shard_idx:05d}.parquet'

    metadata_df = pd.read_parquet(metadata_path)
    # keep rows with "success" status
    metadata_df = metadata_df[metadata_df['status'] == 'success']
    key_list = metadata_df['key']

    num_processes = multiprocessing.cpu_count() - 1

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    reader = multiprocessing.Process(
        target=_read_images,
        args=(img_archive_path, key_list, input_queue, num_processes),
    )
    reader.start()

    processes = [
        multiprocessing.Process(
            target=_process_image,
            args=(input_queue, output_queue, metadata_df, shard_idx),
        )
        for _ in range(num_processes)
    ]

    for process in processes:
      process.start()

    # collect results and yield them
    processed_count = 0
    while processed_count < len(key_list):
      yield output_queue.get()
      processed_count += 1
    
    # Ensure all processes have finished
    reader.join()
    for process in processes:
      process.join()

def _read_images(img_archive_path, key_list, input_queue, num_processes):
  # Using ThreadPoolExecutor to read images in parallel
  with tarfile.open(img_archive_path, 'r') as tar:
    for key in key_list:
      input_queue.put((key, tar.extractfile(f'{key}.jpg').read()))
    
  for _ in range(num_processes):
    input_queue.put(None)

def _process_image(input_queue, output_queue, metadata_df, shard_idx):
  while True:
    item = input_queue.get()
    if item is None:
      break

    key, image = item
    key_idx = int(key)
    key_ = f'{shard_idx}-{key_idx}'
    example = {
      'image': image,
      **_get_example_metadata(metadata_df[metadata_df['key'] == key].iloc[0]),
    }

    output_queue.put((key_, example))


def _get_example_metadata(metadata_df_row):
  """Returns example metadata."""
  nsfw_tag = metadata_df_row['NSFW']
  if nsfw_tag not in _NSFW_TAGS:
    nsfw_tag = _NSFW_MISSING_TAG

  return {
      'caption': metadata_df_row['caption'] or '',
      'nsfw': nsfw_tag,
      'similarity': metadata_df_row['similarity'] or _MISSING_SIMILARITY_VALUE,
      'license': metadata_df_row['LICENSE'] or '',
      'url': metadata_df_row['url'],
      'original_width': metadata_df_row['original_width'],
      'original_height': metadata_df_row['original_height'],
  }

'''
  template = Template(template_code)

  filled_template = template.substitute(version=version, task_id=task_id)

  # make a directory
  os.makedirs(f"laion2b_{task_id}", exist_ok=True)
  with open(f"laion2b_{task_id}/laion2b_{task_id}_dataset_builder.py", "w") as f:
    f.write(filled_template)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a laion2b dataset builder script")
    parser.add_argument("--version", help="The version of the dataset", type=str, required=True)
    parser.add_argument("--task_id", help="The task id", type=str, required=True)

    args = parser.parse_args()

    generate_script(args.version, args.task_id)