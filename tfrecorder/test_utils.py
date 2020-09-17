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

from tfrecorder import constants


TEST_DIR = 'tfrecorder/test_data'


def get_test_df() -> pd.DataFrame:
  """Gets a test dataframe that works with the data in test_data/."""
  return pd.read_csv(os.path.join(TEST_DIR, 'data.csv'))


def get_test_data() -> Dict[str, List[Any]]:
  """Returns test data in columnar format."""

  return get_test_df().to_dict(orient='list')


def get_raw_feature_df() -> pd.DataFrame:
  """Returns test dataframe having raw feature spec schema."""

  df = get_test_df()
  df.drop(constants.IMAGE_URI_KEY, axis=1, inplace=True)
  df['image_name'] = 'image_name'
  df['image'] = 'image'
  # Note: TF Transform parser expects string values in input. They will
  # be parsed based on the raw feature spec that is passed together with the
  # data
  df['image_height'] = '48'
  df['image_width'] = '48'
  df['image_channels'] = '3'
  df = df[constants.RAW_FEATURE_SPEC.keys()]

  return df


def get_test_pipeline():
  """Gets a test pipeline."""
  return test_pipeline.TestPipeline(runner='DirectRunner')


def make_random_image(height, width, channels):
  """Returns a random Numpy image."""

  return Image.fromarray(
      (np.random.random((height, width, channels)) * 255).astype(np.uint8))
