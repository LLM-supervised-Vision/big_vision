"""cc12m dataset."""

import json
import functools
import numpy as np
from etils import epath
from typing import Dict, Tuple
import tensorflow_datasets as tfds

_START_SHARD = 0
_END_SHARD = 1244
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

    return (
        'Generate shard indices'
        >> beam.Create(list(range(_START_SHARD, _END_SHARD)))
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
    print(f"shard_idx = {shard_idx}")
    pd = tfds.core.lazy_imports.pandas

    img_archive_path = dl_manager.manual_dir / f'{shard_idx:05d}.tar'
    metadata_path = dl_manager.manual_dir / f'{shard_idx:05d}.parquet'

    # read (or construct) metadata from parquet (or json) files
    try:
      metadata_df = pd.read_parquet(metadata_path)
    except Exception as e:
      print(f"Constructing metadata_df due to error reading {metadata_path}: {e}")
      metadata_df = pd.DataFrame(columns=['key', 'caption', 'url', 'original_width', 'original_height'])

      iter_archive = dl_manager.iter_archive(img_archive_path)
      file_path, file_obj = _get_next_item(iter_archive, suffix='.json')
      while file_path is not None:
        file_obj.seek(0)
        json_dict = json.loads(file_obj.read())
        key, caption, url, original_width, original_height = json_dict['key'], json_dict['caption'], json_dict['url'], json_dict['original_width'], json_dict['original_height']
        metadata_df = pd.concat([
          metadata_df,
          pd.DataFrame([[key, caption, url, original_width, original_height]], columns=['key', 'caption', 'url', 'original_width', 'original_height'])
        ],ignore_index=True)
        file_path, file_obj = _get_next_item(iter_archive, suffix='.json')
      print(f"metadata_df.shape = {metadata_df.shape}")

    # yield examples
    iter_archive = dl_manager.iter_archive(img_archive_path)
    file_path, file_obj = _get_next_item(iter_archive, suffix='.jpg')
    while file_path is not None:
      key = (file_path.stem)
      if key in metadata_df['key'].values:
        key_idx = metadata_df[metadata_df['key'] == key].index[0]

        example = {
            'image': file_obj.read(),
            **_get_example_metadata(metadata_df.iloc[key_idx]),
        }
        yield (key, example)

      file_path, file_obj = _get_next_item(iter_archive, suffix='.jpg')

def _get_next_item(iter_archive, suffix):
  try:
    while True:
      file_name, file_obj = next(iter_archive)
      if epath.Path(file_name).suffix == suffix: break
    file_path = epath.Path(file_name)
    temp = file_obj.read()
    del temp
  except Exception as e:
    file_path, file_obj = None, None
  return file_path, file_obj

def _get_example_metadata(metadata_df_row):
  """Returns example metadata."""

  return {
      'caption': metadata_df_row['caption'] or '',
      'url': metadata_df_row['url'],
      'original_width': metadata_df_row['original_width'],
      'original_height': metadata_df_row['original_height'],
  }