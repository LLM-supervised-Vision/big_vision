"""laion2b dataset."""

from etils import epath

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets as tfds

_HOMEPAGE = 'https://laion.ai/blog/laion-5b/'

_NUM_SHARDS = 1
_MISSING_SIMILARITY_VALUE = -1.0
_NSFW_MISSING_TAG = 'UNTAGGED'
_NSFW_TAGS = ('UNLIKELY', 'UNSURE', 'NSFW', _NSFW_MISSING_TAG)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for laion2b dataset."""

  VERSION = tfds.core.Version('1.0.0')
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

    for file_name, file_obj in dl_manager.iter_archive(img_archive_path):
      file_path = epath.Path(file_name)
      if file_path.suffix in ('.json', '.txt'):
        continue

      key_idx = int(file_path.stem)

      key = f'{shard_idx}-{key_idx}'
      example = {
        'image': file_obj.read(),
        **_get_example_metadata(metadata_df[metadata_df['key'] == f"{key_idx:09d}"].iloc[0]),
      }

      yield (key, example)

def _get_example_metadata(metadata_df_row):
  """Returns example metadata."""
  nsfw_tag = metadata_df_row['NSFW']
  if nsfw_tag not in _NSFW_TAGS:
    nsfw_tag = _NSFW_MISSING_TAG

  return {
      'caption': metadata_df_row['caption'],
      'nsfw': nsfw_tag,
      'similarity': metadata_df_row['similarity'] or _MISSING_SIMILARITY_VALUE,
      'license': metadata_df_row['LICENSE'] or '',
      'url': metadata_df_row['url'],
      'original_width': metadata_df_row['original_width'],
      'original_height': metadata_df_row['original_height'],
  }