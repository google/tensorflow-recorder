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

"""Tests for Pandas accessor."""

import os
import unittest

# pylint: disable=unused-import
import tfrecorder

from tfrecorder import test_utils


class DataFrameAccessor(unittest.TestCase):
  """UnitTests For DataFrame Accessor."""

  def setUp(self):
    self.test_df = test_utils.get_test_df()
    self.output_dir = '/tmp/train'
    os.makedirs(self.output_dir, exist_ok=True)

  def test_accessor(self):
    """Tests pandas accessor."""

    r = self.test_df.tensorflow.to_tfr(
        runner='DirectRunner', output_dir=self.output_dir)
    self.assertTrue('metrics' in r)


if __name__ == '__main__':
  unittest.main()
