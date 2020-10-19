# Lint as: python3

# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFRecord Dataset utilities."""

import glob
import os
from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft

from tfrecorder import types


TRANSFORMED_METADATA_DIR = tft.TFTransformOutput.TRANSFORMED_METADATA_DIR
TRANSFORM_FN_DIR = tft.TFTransformOutput.TRANSFORM_FN_DIR

_FILE_EXT_TO_COMPRESSION_TYPE = {
    '.gz': 'GZIP',
    '.zlib': 'ZLIB',
}


def _validate_tfrecord_dir(tfrecord_dir: str):
  """Verifies that the TFRecord directory contains expected files."""

  # Check that input is a valid directory.
  if not os.path.isdir(tfrecord_dir):
    raise ValueError(f'Not a directory: {tfrecord_dir}')

  # Check that TensorFlow Transform directories are present.
  for dirname in [TRANSFORMED_METADATA_DIR, TRANSFORM_FN_DIR]:
    if not os.path.isdir(os.path.join(tfrecord_dir, dirname)):
      raise FileNotFoundError(f'Missing expected directory: {dirname}')


# TODO(cezequiel): Add support for GCS files.
def _get_tfrecord_files_per_split(tfrecord_dir: str):
  """Returns TFRecord files for each split.

  The TFRecord filenames should have a prefix based on lowercase versions of
  items in `types.SplitKey.allowed_split_values`. DISCARD split is
  not checked.
  """
  split_to_files = {}
  for split in types.SplitKey.allowed_values[:-1]:
    prefix = split.lower()
    files = glob.glob(os.path.join(tfrecord_dir, prefix + '*'))
    if files:
      split_to_files[split] = files

  if not split_to_files:
    raise FileNotFoundError('No TFRecord files found.')

  return split_to_files


def _infer_tfrecord_compression_type(filename: str):
  """"Infers compression type (e.g. GZIP) of a TFRecord file.

  If file is uncompressed, returns ''.
  """

  ext = os.path.splitext(filename)[-1]
  return _FILE_EXT_TO_COMPRESSION_TYPE.get(ext, '')


def load(tfrecord_dir: str) -> Dict[str, tf.data.Dataset]:
  """Returns TF Datasets loaded from TFRecord files.

  This returns a `dict` keyed by dataset split, e.g.
    ```
    {
      'TRAIN': <tf.data.Dataset>,
      'VALIDATION': <tf.data.Dataset>,
      'TEST': <tf.data.Dataset>,
    }
    ```

  The `tfrecord_dir` is expected to have the following structure,
  based on TFRecorder's `create_tfrecords` default output:
    ```
    tfrecord_dir/
        train-*.tfrecord.gz
        validation-*.tfrecords.gz
        test-*.tfrecord.gz
        transformed_metadata/
        transform_fn/
    ```
  where the TFRecord file prefixes (e.g. train) would be have to match the
  excepted split values in lower case (see `schema.allowed_split_values`).

  This function will only generate items for splits that exists in the
  directory.
  """

  _validate_tfrecord_dir(tfrecord_dir)

  tft_output = tft.TFTransformOutput(tfrecord_dir)
  feature_spec = tft_output.transformed_feature_spec()
  splits = _get_tfrecord_files_per_split(tfrecord_dir)
  split_to_dataset = {}
  for key, files in splits.items():
    if not files:
      continue

    # Assumes all files in split have the same compression type
    first_file = files[0]
    compression_type = _infer_tfrecord_compression_type(first_file)

    dataset = tf.data.TFRecordDataset(files, compression_type)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_spec))
    split_to_dataset[key] = dataset

  return split_to_dataset
