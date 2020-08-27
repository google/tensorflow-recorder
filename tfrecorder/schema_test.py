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

"""Tests for types."""

import unittest
import tensorflow_transform as tft

from tfrecorder import schema


class SchemaTest(unittest.TestCase):
  """Tests for type module."""
  def test_valid_get_tft_coder(self):
    """Tests a valid call on get_tft_coder."""
    columns = ['split', 'image_uri', 'label']
    converter = schema.get_tft_coder(columns, schema.image_csv_schema)
    self.assertIsInstance(converter, tft.coders.CsvCoder)

  def test_valid_get_split_key(self):
    """Tests a valid split key."""
    key = schema.get_split_key(schema.image_csv_schema)
    self.assertEqual(key, 'split')

  def test_no_get_split_key(self):
    """Tests no split key present."""
    test_schema = dict()
    for k, v in schema.image_csv_schema.items():
      # Brute force copy because OG is a FrozenOrderedDict.
      if k != 'split':
        test_schema[k] = v

    key = schema.get_split_key(test_schema)
    self.assertIsNone(key)

  def test_get_raw_metadata(self):
    """Tests a valid call to get_raw_metadata."""
    columns = ['split', 'image_uri', 'label']
    raw_metadata = schema.get_raw_metadata(columns, schema.image_csv_schema)
    self.assertIsInstance(
        raw_metadata, 
        tft.tf_metadata.dataset_metadata.DatasetMetadata)

        
if __name__ == '__main__':
  unittest.main()
