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

"""Tests `check.py`."""

import functools
import os
import tempfile
import unittest

import mock
import pandas as pd
from pandas import testing as pdt
import tensorflow as tf

from tfrecorder import beam_image
from tfrecorder import check
from tfrecorder import constants
from tfrecorder import test_utils


# pylint: disable=protected-access

class ReadTFRecordsTest(unittest.TestCase):
  """Tests `_read_tfrecords`."""

  def setUp(self):
    self.tfrecords_dir = os.path.join(test_utils.TEST_DIR, 'sample_tfrecords')

  def test_valid_compressed_gzip(self):
    """Tests valid case using GZIP compression."""

    # Use list of file pattern strings to maintain train, validation, test
    # order.
    file_pattern = [
        os.path.join(self.tfrecords_dir, '{}*.tfrecord.gz'.format(f))
        for f in ['train, validation, test']]

    compression_type = 'GZIP'
    actual = check._read_tfrecords(
        file_pattern, self.tfrecords_dir, compression_type)

    expected_csv = os.path.join(test_utils.TEST_DIR, 'data.csv')
    expected = tf.data.experimental.make_csv_dataset(
        expected_csv, batch_size=1, label_name=None, num_epochs=1,
        shuffle=False)

    for a, e in zip(actual, expected):
      self.assertCountEqual(a.keys(), constants.RAW_FEATURE_SPEC)
      for key in constants.IMAGE_CSV_FEATURE_SPEC:
        self.assertEqual(a[key], e[key])

  def test_error_invalid_file_pattern(self):
    """Tests error case where file pattern is invalid."""

    file_pattern = 'gs://path/to/memes/folder'
    with self.assertRaises(tf.errors.OpError):
      check._read_tfrecords(file_pattern)


class CheckTFRecordsTest(unittest.TestCase):
  """Tests `check_tfrecords`."""

  def setUp(self):
    """Test setup."""

    image_height = 40
    image_width = 30
    image_channels = 3
    image_fn = functools.partial(
        test_utils.make_random_image, image_height, image_width,
        image_channels)

    data = test_utils.get_test_data()
    num_records = len(data[constants.IMAGE_URI_KEY])
    image_uris = data.pop(constants.IMAGE_URI_KEY)
    data['image_name'] = [os.path.split(uri)[-1] for uri in image_uris]
    data.update({
        'image': [beam_image.encode(image_fn())
                  for _ in range(num_records)],
        'image_height': [image_height] * num_records,
        'image_width': [image_width] * num_records,
        'image_channels': [image_channels] * num_records,
    })
    self.num_records = num_records
    self.data = data
    self.dataset = tf.data.Dataset.from_tensor_slices(self.data)

  @mock.patch.object(check, '_read_tfrecords', autospec=True)
  def test_valid_records(self, mock_fn):
    """Tests valid case on reading multiple records."""

    file_pattern = 'gs://path/to/tfrecords/*'
    mock_fn.return_value = self.dataset
    num_records = len(self.data['image'])

    with tempfile.TemporaryDirectory(dir='/tmp') as dir_:
      actual_dir = check.check_tfrecords(
          file_pattern, num_records=num_records, output_dir=dir_)
      self.assertTrue('check-tfrecords-' in actual_dir)

      actual_csv = os.path.join(actual_dir, 'data.csv')
      self.assertTrue(os.path.exists(actual_csv))

      _ = self.data.pop('image')

      # Check output CSV
      actual_df = pd.read_csv(actual_csv)
      expected_df = pd.DataFrame(self.data)
      pdt.assert_frame_equal(actual_df, expected_df)

      # Check output images
      actual_image_files = [
          f for f in os.listdir(actual_dir) if f.endswith('.jpg')]
      expected_image_files = self.data['image_name']
      self.assertCountEqual(actual_image_files, expected_image_files)


if __name__ == '__main__':
  unittest.main()
