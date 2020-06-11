# Lint as: python3
"""Tests for client."""

import unittest
import pandas as pd
from tfrutil import accessor


class ClientAccessor(unittest.TestCase):

  def setUp(self):
    data = {
        "image": ["gs://foo/bar/1.jpg",
                  "gs://foo/bar/2.jpg",
                  "gs://foo/bar/3.jpg"],
        "label": [0, 0, 1]}
    self.test_df = pd.DataFrame.from_dict(data)

  def test_pandas_accessor(self):
    pid = self.test_df.tfrutil.to_tfr(runner="local",
                                      output_path="/tmp/train")
    self.assertEqual(pid, "p1234")


if __name__ == "__main__":
  unittest.main()
