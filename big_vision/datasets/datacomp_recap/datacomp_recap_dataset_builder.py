"""datacomp_recap dataset."""

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

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
        """Dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="DataComp-Recap-1B dataset",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Text(),  # URL of the image
                'url': tfds.features.Text(),
                're_caption': tfds.features.Text(),
                'org_caption': tfds.features.Text(),
                'original_width': tf.int64,
                'original_height': tf.int64,
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager._download_dir / 'datacomp_recap_train.tfrecord'
        return {
            'train': self._generate_examples(path),
        }

    def _generate_examples(self, filepath):
        """Yields examples."""
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
                'original_width': feature['width'].int64_list.value[0],
                'original_height': feature['height'].int64_list.value[0],
            }
