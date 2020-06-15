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

"""TFRUtil Beam Pipeline.

This file implements the full beam pipeline for TFRUtil.
"""
import datetime
import os
from typing import Union

import apache_beam as beam
import pandas as pd
# import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam


def _get_job_name(job_label: str = None) -> str:
  """Returns Beam runner job name.

  Args:
    job_label: A user defined string that helps define the job.

  Returns:
    A job name compatible with apache beam runners, including a time stamp to
      insure uniqueness.
  """

  job_name = "tfrutil-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  if job_label:
    job_label = job_label.replace("_", "-")
    job_name += "-" + job_label

  return job_name


def _get_job_dir(output_path: str, job_name: str) -> str:
  """Returns Beam processing job directory."""

  return os.path.join(output_path, job_name)


def _get_pipeline_options(
    job_name: str, job_dir: str, **popts: Union[bool, str, float]
    ) -> beam.pipeline.PipelineOptions:
  """Returns Beam pipeline options."""

  options_dict = {
      "staging_location": os.path.join(job_dir, "staging"),
      "temp_location": os.path.join(job_dir, "tmp"),
      "job_name": job_name,
      "teardown_policy": "TEARDOWN_ALWAYS",
      "save_main_session": True,
      "pipeline_type_check": False,
  }
  options_dict.update(popts)
  return beam.pipeline.PipelineOptions(flags=[], **options_dict)

#pylint: disable=too-many-arguments

def run_pipeline(df: pd.DataFrame,
                 job_label: str,
                 runner: str,
                 output_path: str,
                 compression: str,
                 num_shards: int,
                 image_col: str,
                 label_col: str):
  """Runs TFRUtil Beam Pipeline.

  Args:
    df: Pandas Dataframe
    job_label: User description for the beam job.
    runner: Beam Runner: (e.g. DataFlowRunner, DirectRunner).
    output_path: GCS or Local Path for output.
    compression: gzip or None.
    num_shards: Number of shards.
    image_col: Image column name.
    label_col: Label column name.

  Note: These inputs must be validated upstrea (by client.create_tfrecord())
  """

  # TODO: make use of the following params
  _ = compression
  _ = num_shards
  _ = image_col
  _ = label_col

  job_name = _get_job_name(job_label)
  job_dir = _get_job_dir(output_path, job_name)
  popts = {}  # TODO(mikebernico): consider how/if to pass pipeline options.
  options = _get_pipeline_options(job_name, job_dir, **popts)

  with beam.Pipeline(runner, options=options) as p:
    with tft_beam.Context(temp_dir=os.path.join(job_dir, "tft_tmp")):

      raw_data = (
          p
          | "ReadFromDataFrame" >> beam.Create(df.values.tolist())
      )

      # TODO(mikebernico): Implement full pipeline
      _ = (
          raw_data
          | "Debugger" >> beam.io.WriteToText(
              os.path.join(job_dir, "debug"))
      )
