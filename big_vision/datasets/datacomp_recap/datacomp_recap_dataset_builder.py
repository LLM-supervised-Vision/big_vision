"""datacomp_recap dataset."""

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

_HOMEPAGE="https://github.com/UCSC-VLAA/Recap-DataComp-1B"
_DESCRIPTION = """
Recap-Datacomp-1B is a large-scale image-text dataset that has been recaptioned 
using an advanced LLaVA-1.5-LLaMA3-8B model to enhance the alignment and detail 
of textual descriptions.
"""
class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for datacomp_recap dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

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

  # def _generate_examples(self, path):
  #   """Yields examples."""
  #   # TODO(datacomp_recap): Yields (key, example) tuples from the dataset
  #   for f in path.glob('*.jpeg'):
  #     yield 'key', {
  #         'image': f,
  #         'label': 'yes',
  #     }

  def _generate_examples(
      self,
      dl_manager: tfds.download.DownloadManager,
  ):
    from datasets import load_dataset
    ds = load_dataset("UCSC-VLAA/Recap-DataComp-1B", split="preview",streaming=True)
    for i, example in enumerate(ds):
      url = example["url"]
      import pdb; pdb.set_trace()
      image = Image.open(dl_manager.download(url))
      yield i, {
          'image': image,
          'url': url,
          'org_caption': example["org_caption"],
          're_caption': example["re_caption"],
          'original_width': image.width,
          'original_height': image.height,
      }