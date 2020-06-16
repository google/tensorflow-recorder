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

"""Tests for client."""

import unittest
import pandas as pd
from tfrutil import client

TEST_DATA = {"image": ["gs://foo/bar/1.jpg",
                       "gs://foo/bar/2.jpg",
                       "gs://foo/bar/3.jpg"],
             "label": [0, 0, 1]}


class ClientTest(unittest.TestCase):

  def setUp(self):
    self.test_df = pd.DataFrame.from_dict(TEST_DATA)

  def test_create_tfrecords(self):

    pid = client.create_tfrecords(self.test_df,
                                  runner="DirectRunner",
                                  output_path="/tmp/train")
    self.assertEqual(pid, "p1234")


class InputValidationTest(unittest.TestCase):

  def setUp(self):
    self.test_df = pd.DataFrame.from_dict(TEST_DATA)

  def test_valid_dataframe(self):
    client._validate_data(self.test_df,
                          image_col="image",
                          label_col="label")

  def test_invalid_image(self):
    with self.assertRaises(AttributeError):
      client._validate_data(self.test_df,
                            image_col="foo",
                            label_col="label")

  def test_invalid_label(self):
    with self.assertRaises(AttributeError):
      client._validate_data(self.test_df,
                            image_col="image",
                            label_col="foo")

  def test_valid_runner(self):
    client._validate_runner("DirectRunner")

  def test_invalid_runner(self):
    with self.assertRaises(AttributeError):
      client._validate_runner("FooRunner")


if __name__ == "__main__":
  unittest.main()
