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
the Pandas DataFrame Accessor (accessor.py) and the CLI
(TODO(cezequeil: name file)).
"""
from typing import Union
import pandas as pd

from tfrutil import beam_pipeline


def _validate_data(df, image_col, label_col):
  # verify there is a column latitude and a column longitude
  if image_col not in df.columns:
  # or label_col not in df.columns:
    raise AttributeError(
        "Dataframe must contain specified image column {}.".format(image_col))
  if label_col not in df.columns:
    raise AttributeError(
        "Dataframe must contain specified label column {}.".format(label_col))


def _validate_runner(runner):
  """Validates a supported beam runner is chosen."""
  if runner not in ["DataFlowRunner", "DirectRunner"]:
    raise AttributeError("Runner {} is not supported.".format(runner))


def create_tfrecords(df: pd.DataFrame,
                     runner: str,
                     output_path: str,
                     job_label: str = "beam-job",
                     compression: Union[str, None] = "gzip",
                     num_shards: int = 0,
                     image_col: str = "image",
                     label_col: str = "label") -> str:
  """TFRUtil Python Client.

  TFRUtil provides an easy interface to create image-based tensorflow records
  from a dataframe containing GCS locations of the images and labels.

  Usage:
    import tfrutil

    job_id = tfrutil.create_tfrecords(train_df,
                                     runner="local",
                                     output_path="gcs://foo/bar/train",
                                     compression="gzip",
                                     num_shards=10,
                                     image_col="image",
                                     label_col="label)

  Args:
    df: Pandas DataFrame
    runner: Beam runner. Can be "local" or "DataFlow"
    output_path: Local directory or GCS Location to save TFRecords to.
    job_label: User supplied description for the beam job name.
    compression: Can be "gzip" or None for no compression.
    num_shards: Number of shards to divide the TFRecords into. Default is
        0 = no sharding.
    image_col: DataFrame column containing GCS path to image. Defaults to
      "image".
    label_col: DataFrame column containing image label. Defaults to "label".
  Returns:
    job_id: Job ID of the DataFlow job or PID of the local runner.
  """
  _validate_data(df, image_col, label_col)
  _validate_runner(runner)
  beam_pipeline.run_pipeline(df,
                             job_label=job_label,
                             runner=runner,
                             output_path=output_path,
                             compression=compression,
                             num_shards=num_shards,
                             image_col=image_col,
                             label_col=label_col)
  job_id = "p1234"
  return job_id
