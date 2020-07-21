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

"""Common utilites for writing tfrecorder tests."""

from typing import Any, Dict, List

import os

import numpy as np
from PIL import Image

from apache_beam.testing import test_pipeline
import pandas as pd


TEST_DIR = 'tfrecorder/test_data'

# TEST_DATA = collections.OrderedDict({
#     constants.SPLIT_KEY:['TRAIN', 'VALIDATION', 'TEST'],
#     constants.IMAGE_URI_KEY: [
#         'gs://foo/bar/1.jpg',
#         'gs://foo/bar/2.jpg',
#         'gs://foo/bar/3.jpg',
#     ],
#     constants.LABEL_KEY: [0, 0, 1]})


def get_test_df():
  """Gets a test dataframe that works with the data in test_data/."""
  return pd.read_csv(os.path.join(TEST_DIR, 'data.csv'))


def get_test_data() -> Dict[str, List[Any]]:
  """Returns test data in columnar format."""

  return get_test_df().to_dict(orient='list')


def get_test_pipeline():
  """Gets a test pipeline."""
  return test_pipeline.TestPipeline(runner='DirectRunner')


def make_random_image(height, width, channels):
  """Returns a random Numpy image."""

  return Image.fromarray(
      (np.random.random((height, width, channels)) * 255).astype(np.uint8))
