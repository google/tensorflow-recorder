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

"""Tests for `dataset_loader.py`."""

import os
import tempfile
import unittest

from tfrecorder import dataset_loader
from tfrecorder import test_utils
from tfrecorder import types

# pylint: disable=protected-access

class ValidateTFRecordDirTest(unittest.TestCase):
  """Tests `_validate_tfrecord_dir`."""

  def setUp(self):
    self._temp_dir_obj = tempfile.TemporaryDirectory()
    self.temp_dir = self._temp_dir_obj.name

  def tearDown(self):
    self._temp_dir_obj.cleanup()

  def test_ok(self):
    """Checks that function works as expected when TFT dirs are present."""
    os.makedirs(
        os.path.join(self.temp_dir, dataset_loader.TRANSFORMED_METADATA_DIR))
    os.makedirs(os.path.join(self.temp_dir, dataset_loader.TRANSFORM_FN_DIR))
    dataset_loader._validate_tfrecord_dir(self.temp_dir)

  def test_missing_metadata_dir(self):
    """Check exception raised when metadata directory missing."""

    with self.assertRaises(FileNotFoundError):
      os.makedirs(os.path.join(self.temp_dir, dataset_loader.TRANSFORM_FN_DIR))
      dataset_loader._validate_tfrecord_dir(self.temp_dir)

  def test_missing_transform_fn_dir(self):
    """Check exception raised when transform_fn directory missing."""
    with self.assertRaises(FileNotFoundError):
      os.makedirs(
          os.path.join(self.temp_dir, dataset_loader.TRANSFORMED_METADATA_DIR))
      dataset_loader._validate_tfrecord_dir(self.temp_dir)

  def test_missing_tf_transform_dirs(self):
    """Check exception raised when both TFT transform directories missing."""
    with self.assertRaises(FileNotFoundError):
      dataset_loader._validate_tfrecord_dir(self.temp_dir)

  def test_not_dir(self):
    """Check exception raised when input is not a valid directory."""

    input_dir = '/some/non-existent/dir'
    with self.assertRaisesRegex(ValueError, 'Not a directory:'):
      dataset_loader._validate_tfrecord_dir(input_dir)


class LoadTest(unittest.TestCase):
  """Tests `load` function."""

  def setUp(self):
    self.tfrecord_dir = test_utils.TEST_TFRECORDS_DIR

  def test_load_all_splits(self):
    """Test case where all TFRecord splits can be loaded."""
    dataset_dict = dataset_loader.load(self.tfrecord_dir)
    self.assertEqual(len(dataset_dict), 3)
    self.assertCountEqual(
        list(dataset_dict.keys()), types.SplitKey.allowed_values[:-1])


if __name__ == '__main__':
  unittest.main()
