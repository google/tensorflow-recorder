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

from tfrecorder import types
import tensorflow_transform as tft


class TypeTest(unittest.TestCase):
  """Tests for type module."""
  def valid_get_tft_coder(self):
    """Tests a valid call on get_tft_coder."""
    columns = ['split', 'image_uri', 'label']
    converter = types.get_tft_coder(columns, types.schema_map)
    self.assertIsInstance(converter, tft.coders.CsvCoder)


if __name__ == '__main__':
  unittest.main()
