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

"""Common utilites for writing tfrutil tests."""

from apache_beam.testing import test_pipeline
import pandas as pd

def get_test_df():
  """Gets a test dataframe that works with the data in test_data/."""
  return pd.read_csv("tfrutil/test_data/data.csv")


def get_test_pipeline():
  """Gets a test pipeline."""
  return test_pipeline.TestPipeline(runner="DirectRunner")
