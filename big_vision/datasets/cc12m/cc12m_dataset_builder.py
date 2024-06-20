"""cc12m dataset."""

import functools
import numpy as np
from etils import epath
from typing import Dict, Tuple
import tensorflow_datasets as tfds

_HOMEPAGE="https://github.com/google-research-datasets/conceptual-12m"
_DESCRIPTION = """
We introduce the Conceptual 12M (CC12M), a dataset with ~12 million image-text pairs 
meant to be used for vision-and-language pre-training. It is larger and covers a much 
more diverse set of visual concepts than the Conceptual Captions (CC3M), a dataset that 
is widely used for pre-training and end-to-end training of image captioning models. 
Check our paper for further details.
"""

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cc12m dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
  Refer to "Download" section on {_HOMEPAGE}
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = {
      # These are the features of your dataset like images, labels ...
      'image': tfds.features.Image(doc='image'),
      "caption": tfds.features.Text(),
      "url": tfds.features.Text(doc='image URL'),
      "original_width": tfds.features.Tensor(shape=(), dtype=np.int64),
      "original_height": tfds.features.Tensor(shape=(), dtype=np.int64),
    }
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        supervised_keys=None,
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
    beam = tfds.core.lazy_imports.apache_beam
    num_shards = 2
    return (
        'Generate shard indices'
        >> beam.Create(list(range(num_shards)))
        | 'Generate examples from a single shard'
        >> beam.FlatMap(
            functools.partial(
                self._generate_examples_one_shard,
                dl_manager,
            )
        )
        | 'Prevent fusion of transforms' >> beam.Reshuffle()
    )

  def _generate_examples_one_shard(
      self,
      dl_manager: tfds.download.DownloadManager,
      shard_idx: int,
  ):
    """Yields examples from a single shard."""
    pd = tfds.core.lazy_imports.pandas

    img_archive_path = dl_manager.manual_dir / f'{shard_idx:05d}.tar'
    metadata_path = dl_manager.manual_dir / f'{shard_idx:05d}.parquet'
    print(f"img_archive_path: {img_archive_path}")
    print(f"metadata_path: {metadata_path}")

    metadata_df = pd.read_parquet(metadata_path)
    # print(f"metadata_df: {metadata_df}")
    count = 3
    for file_name, file_obj in dl_manager.iter_archive(img_archive_path):
      file_path = epath.Path(file_name)
      if file_path.suffix in ('.json', '.txt'):
        continue
      # print(f"file_path: {file_path}")
      # row_idx = int(file_path.stem) % 10000
      # key = f'{shard_idx}_{row_idx}'
      key = (file_path.stem)
      # print(f"key: {key}")
      # key_idx = int(metadata_df['key'].iloc[row_idx])%10000
      key_idx = metadata_df[metadata_df['key'] == key].index[0]
      # print(f"key_idx: {key_idx}")

      # if count > 0:
      #   print(f"key: {key}")
      #   print(f"file_path: {file_path}")
      #   row_df = metadata_df.iloc[row_idx]
      #   print(f"row_df['caption']: {row_df['caption']}")
      #   print(f"row_df['url']: {row_df['url']}")
      #   print(f"row_df['key']: {row_df['key']}")
      #   key_df = metadata_df.iloc[key_idx]
      #   print(f"key_df['caption']: {key_df['caption']}")
      #   print(f"key_df['url']: {key_df['url']}")
      #   # save the image
      #   with open(f"/home/austinwang/austin_big_vision/big_vision/datasets/cc12m/downloads/{file_path.name}", 'wb') as f:
      #     f.write(file_obj.read())
      #   count -= 1

      example = {
          'image': file_obj.read(),
          **_get_example_metadata(metadata_df.iloc[key_idx]),
      }
      yield (key, example)

def _get_example_metadata(metadata_df_row):
  """Returns example metadata."""

  return {
      'caption': metadata_df_row['caption'] or '',
      'url': metadata_df_row['url'],
      'original_width': metadata_df_row['original_width'],
      'original_height': metadata_df_row['original_height'],
  }