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

"""Tests `utils.py`."""

import functools
import os
import tempfile
import unittest

import mock
import pandas as pd
from pandas import testing as pdt
import tensorflow as tf

from tfrecorder import beam_image
from tfrecorder import constants
from tfrecorder import utils
from tfrecorder import test_utils
from tfrecorder import input_schema
from tfrecorder import dataset_loader


# pylint: disable=protected-access


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
    schema = input_schema.IMAGE_CSV_SCHEMA
    image_uri_key = schema.image_uri_key
    num_records = len(data[image_uri_key])
    image_uris = data.pop(image_uri_key)
    data['image_name'] = [os.path.split(uri)[-1] for uri in image_uris]
    data.update({
        'image': [beam_image.encode(image_fn())
                  for _ in range(num_records)],
        'image_height': [image_height] * num_records,
        'image_width': [image_width] * num_records,
        'image_channels': [image_channels] * num_records,
    })
    self.tfrecord_dir = 'gs://path/to/tfrecords/dir'
    self.split = 'TRAIN'
    self.num_records = num_records
    self.data = data
    self.dataset = tf.data.Dataset.from_tensor_slices(self.data)

  @mock.patch.object(dataset_loader, 'load', autospec=True)
  def test_valid_records(self, mock_fn):
    """Tests valid case on reading multiple records."""

    mock_fn.return_value = {self.split: self.dataset}
    num_records = len(self.data['image'])

    with tempfile.TemporaryDirectory(dir='/tmp') as dir_:
      actual_dir = utils.inspect(
          self.tfrecord_dir, split=self.split, num_records=num_records,
          output_dir=dir_)
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

  @mock.patch.object(dataset_loader, 'load', autospec=True)
  def test_no_data_for_split(self, mock_fn):
    """Check exception raised when data could not be loaded given `split`."""

    mock_fn.return_value = {}
    with self.assertRaisesRegex(ValueError, 'Could not load data for'):
      utils.inspect(self.tfrecord_dir, split='UNSUPPORTED')


if __name__ == '__main__':
  unittest.main()


class CopyLogTest(unittest.TestCase):
  """Misc tests for _copy_logfile_to_gcs."""

  def test_valid_copy(self):
    """Test valid file copy."""
    with tempfile.TemporaryDirectory() as tmpdirname:
      text = 'log test log test'
      infile = os.path.join(tmpdirname, 'foo.log')
      with open(infile, 'w') as f:
        f.write(text)
      utils.copy_logfile_to_gcs(infile, tmpdirname)

      outfile = os.path.join(tmpdirname, constants.LOGFILE)
      with open(outfile, 'r') as f:
        data = f.read()
        self.assertEqual(text, data)

  def test_invalid_copy(self):
    """Test invalid file copy."""
    with tempfile.TemporaryDirectory() as tmpdirname:
      infile = os.path.join(tmpdirname, 'foo.txt')
      with self.assertRaises(FileNotFoundError):
        utils.copy_logfile_to_gcs(infile, tmpdirname)
