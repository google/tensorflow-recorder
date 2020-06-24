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

"""Provides a common interface for TFRUtil to DF Accessor and CLI.

client.py provides create_tfrecords() to upstream clients including
the Pandas DataFrame Accessor (accessor.py) and the CLI (cli.py).
"""

from typing import Union, Optional, Sequence

import pandas as pd
import tensorflow as tf

from tfrutil import constants
from tfrutil import beam_pipeline


def _validate_data(df):
  """ Verify required image csv columsn exist in data."""
  if constants.IMAGE_URI_KEY not in df.columns:
  # or label_col not in df.columns:
    raise AttributeError(
        "Dataframe must contain image_uri column {}.")
  if constants.LABEL_KEY not in df.columns:
    raise AttributeError(
        "Dataframe must contain label column.")
  if constants.SPLIT_KEY not in df.columns:
    raise AttributeError(
        "Dataframe must contain split column.")


def _validate_runner(runner):
  """Validates a supported beam runner is chosen."""
  if runner not in ["DataFlowRunner", "DirectRunner"]:
    raise AttributeError("Runner {} is not supported.".format(runner))


# def read_image_directory(dirpath) -> pd.DataFrame:
#   """Reads image data from a directory into a Pandas DataFrame."""
#
#   # TODO(cezequiel): Implement in phase 2.
#   _ = dirpath
#   raise NotImplementedError


def _is_directory(input_data) -> bool:
  """Returns True if `input_data` is a directory; False otherwise."""
  # TODO(cezequiel): Implement in phase 2.
  _ = input_data
  return False


def read_csv(
    csv_file: str,
    header: Optional[Union[str, int, Sequence]] = "infer",
    names: Optional[Sequence] = None) -> pd.DataFrame:
  """Returns a a Pandas DataFrame from a CSV file."""

  if header is None and not names:
    names = constants.IMAGE_CSV_COLUMNS

  with tf.io.gfile.GFile(csv_file) as f:
    return pd.read_csv(f, names=names, header=header)


def to_dataframe(
    input_data: Union[str, pd.DataFrame],
    header: Optional[Union[str, int, Sequence]] = "infer",
    names: Optional[Sequence] = None) -> pd.DataFrame:
  """Converts `input_data` to a Pandas DataFrame."""

  if isinstance(input_data, pd.DataFrame):
    df = input_data[names] if names else input_data

  elif isinstance(input_data, str) and input_data.endswith(".csv"):
    df = read_csv(input_data, header, names)

  elif isinstance(input_data, str) and _is_directory(input_data):
    # TODO(cezequiel): Implement in phase 2
    raise NotImplementedError

  else:
    raise ValueError("Unsupported `input_data`: {}".format(type(input_data)))

  return df

#pylint: disable=too-many-arguments

def create_tfrecords(
    input_data: Union[str, pd.DataFrame],
    output_path: str,
    header: Optional[Union[str, int, Sequence]] = "infer",
    names: Optional[Sequence] = None,
    runner: str = "DirectRunner",
    job_label: str = "create-tfrecords",
    compression: Union[str, None] = "gzip",
    num_shards: int = 0) -> str:
  """Generates TFRecord files from given input data.

  TFRUtil provides an easy interface to create image-based tensorflow records
  from a dataframe containing GCS locations of the images and labels.

  Usage:
    import tfrutil

    job_id = tfrutil.client.create_tfrecords(
        train_df,
        output_path="gcs://foo/bar/train",
        runner="DataFlowRunner)

  Args:
    input_data: Pandas DataFrame, CSV file or image directory path.
    output_path: Local directory or GCS Location to save TFRecords to.
    header: List of field names to use.
      If `input_data` is a CSV file (str) and header is `None`,
        defaults to using `constants.IMAGE_CSV_COLUMNS`.
      if `input_data` is a DataFrame and `header` is given,
        filter output DataFrame columns based on `header`.
    runner: Beam runner. Can be "local" or "DataFlow"
    job_label: User supplied description for the beam job name.
    compression: Can be "gzip" or None for no compression.
    num_shards: Number of shards to divide the TFRecords into. Default is
        0 = no sharding.
  Returns:
    job_id: Job ID of the DataFlow job or PID of the local runner.
  """

  df = to_dataframe(input_data, header, names)

  _validate_data(df)
  _validate_runner(runner)
  beam_pipeline.run_pipeline(
      df,
      job_label=job_label,
      runner=runner,
      output_path=output_path,
      compression=compression,
      num_shards=num_shards)

  job_id = "p1234"
  return job_id
