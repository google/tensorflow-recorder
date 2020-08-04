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

"""Utilities for checking content of TFRecord files."""

from typing import Dict, Optional, Sequence, Union

import csv
import os

import tensorflow as tf
import tensorflow_transform as tft

from tfrecorder import beam_image
from tfrecorder import constants
from tfrecorder import common

_OUT_IMAGE_TEMPLATE = 'image_{:0>3d}.png'


def _stringify(scalar: tf.Tensor) -> str:
  """Converts scalar tensor into a Python string."""

  val = scalar.numpy()
  return val.decode('utf-8') if isinstance(val, bytes) else str(val)


def _read_tfrecords(
    file_pattern: Union[str, Sequence[str]],
    tft_output_dir: Optional[str] = None,
    compression_type: str = 'GZIP') -> tf.data.Dataset:
  """Reads TFRecords files and outputs a TensorFlow Dataset.

  Currently supports Image CSV format only.
  """

  files = tf.io.gfile.glob(file_pattern)

  if not tft_output_dir:
    tft_output_dir = os.path.dirname(file_pattern)
  tf_transform_output = tft.TFTransformOutput(tft_output_dir)
  feature_spec = tf_transform_output.transformed_feature_spec()

  if set(feature_spec.keys()) != set(constants.RAW_FEATURE_SPEC):
    raise ValueError('Unsupported schema: {}'.format(feature_spec.keys()))

  dataset = tf.data.TFRecordDataset(files, compression_type)
  return dataset.map(lambda x: tf.io.parse_single_example(
      x, feature_spec))


def _save_image_from_record(record: Dict[str, tf.Tensor], outfile: str):
  """Extracts image data from parsed TFRecord and saves it to a file."""

  b64_image = record['image'].numpy()
  image = beam_image.decode(
      b64_image,
      record['image_width'], record['image_height'], record['image_channels'])
  image.save(outfile)


def check_tfrecords(
    file_pattern: str,
    num_records: int = 1,
    output_dir: str = 'output',
    compression_type: str = 'GZIP'):
  """Reads TFRecord files and outputs decoded contents to a temp directory."""

  dataset = _read_tfrecords(file_pattern, compression_type=compression_type)

  data_dir = os.path.join(
      output_dir, 'check-tfrecords-' + common.get_timestamp())
  os.makedirs(data_dir)

  csv_file = os.path.join(data_dir, 'data.csv')
  with open(csv_file, 'wt') as f:
    writer = csv.writer(f)

    # Write CSV header
    header = [k for k in constants.RAW_FEATURE_SPEC.keys() if k != 'image']
    writer.writerow(header)

    for r in dataset.take(num_records):
      # Save non-image bytes data to CSV.
      # This will save image metadata as well.
      row = [_stringify(r[k]) for k in header]
      writer.writerow(row)

      # Save image data to a file
      if 'image_name' in r:
        _, image_filename = os.path.split(_stringify(r['image_name']))
        image_path = os.path.join(data_dir, image_filename)
        _save_image_from_record(r, image_path)

    print('Output written to {}'.format(data_dir))

    return data_dir
