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

"""Tests for TFRecorder types."""

import unittest
import tensorflow_transform as tft

from tfrecorder import input_schema


class InputSchemaTest(unittest.TestCase):
  """Tests for type module."""

  def setUp(self):
    self.schema = input_schema.Schema(input_schema.image_csv_schema_map)

  def test_valid_get_input_coder(self):
    """Tests a valid call on get_input_coder."""
    converter = self.schema.get_input_coder()
    self.assertIsInstance(converter, tft.coders.CsvCoder)

  def test_valid_get_key(self):
    """Tests a valid split key."""
    self.assertEqual(self.schema.split_key, 'split')

  def test_no_get_split_key(self):
    """Tests no split key present."""
    test_schema_map = dict()
    for k, v in input_schema.IMAGE_CSV_SCHEMA.input_schema_map.items():
      # Brute force copy because OG is a FrozenOrderedDict.
      if k != 'split':
        test_schema_map[k] = v

    with self.assertRaises(AttributeError):
      _ = input_schema.Schema(test_schema_map)

  def test_get_raw_metadata(self):
    """Tests a valid call to get_raw_metadata."""
    pre_tft_metadata = self.schema.get_pre_tft_metadata()
    self.assertIsInstance(
        pre_tft_metadata,
        tft.tf_metadata.dataset_metadata.DatasetMetadata)

  def test_get_input_keys(self):
    """"Tests get_input_keys() function."""
    schema = input_schema.IMAGE_CSV_SCHEMA
    self.assertEqual(schema.input_schema_map.keys(), schema.get_input_keys())


if __name__ == '__main__':
  unittest.main()
