# Lint as: python3
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
                                  runner="local",
                                  output_path="/tmp/train")
    self.assertEqual(pid, "p1234")


class DataValidationTest(unittest.TestCase):

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

if __name__ == "__main__":
  unittest.main()
