# Copyright 2023 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoring classifier.

This one is based on a generative perspective for image classification.
Here we input the image as well as all the tokenized labels to compute their
perplexity and select the one with minimum loss as the prediction.
"""
import functools
from big_vision.datasets.imagenet import class_names as imagenet_class_names
from big_vision.evaluators import mean
from big_vision.pp import builder as pp_builder
import jax.numpy as jnp
import numpy as np

import tensorflow as tf

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


CLASS_NAMES = {
    "imagenet2012": imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES,
}


# As a separate function to cache result across instances.
@functools.lru_cache(maxsize=None)
def get_classes(dataset_name, pp_txt):
  """Load the class label strings and tokenize them using pp_txt."""
  pp_fn = pp_builder.get_preprocess_fn(pp_txt, log_data=False)
  cls_tokens = {"_label_tokens": [], "mask_ar": []}
  for name in CLASS_NAMES[dataset_name]:
    out = pp_fn({"label": tf.constant(name)})
    cls_tokens["_label_tokens"].append(out["labels"])
    cls_tokens["mask_ar"].append(out["mask_ar"])
  cls_tokens["_label_tokens"] = np.array(cls_tokens["_label_tokens"])
  cls_tokens["mask_ar"] = np.array(cls_tokens["mask_ar"])
  return cls_tokens


def scoring(predict_fn, tokenized_labels):

  def _scoring_fn(train_state, batch, *a, **kw):
    batch = {**tokenized_labels, **batch}
    scores = predict_fn(train_state, batch, *a, **kw)
    predictions = jnp.argmax(scores, axis=-1)
    out = predictions == batch["label"]
    import logging
    logging.info(f"accuracy: {out}")
    return {"prec@1": out}

  return _scoring_fn


class Evaluator(mean.Evaluator):
  """Evaluator for classification accuracy based on scoring all classes."""

  def __init__(self, predict_fn, data, pp_fn, pp_txt, *a, **kw):
    cls_tokens = get_classes(data["name"], pp_txt)
    kw['pp_fn'] = pp_fn
    kw['data'] = data
    super().__init__(scoring(predict_fn, cls_tokens), *a, **kw)
