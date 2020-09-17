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

"""Provides a common interface for TFRecorder to DF Accessor and CLI.

client.py provides create_tfrecords() to upstream clients including
the Pandas DataFrame Accessor (accessor.py) and the CLI (cli.py).
"""
import logging
import os
from typing import Any, Dict, Union, Optional, Sequence

import apache_beam as beam
import pandas as pd
import tensorflow as tf

from tfrecorder import common
from tfrecorder import constants
from tfrecorder import beam_pipeline


def _validate_data(df):
  """ Verifies required image csv columsn exist in data."""
  if constants.IMAGE_URI_KEY not in df.columns:
  # or label_col not in df.columns:
    raise AttributeError(
        'DataFrame must contain image_uri column {}.')
  if constants.LABEL_KEY not in df.columns:
    raise AttributeError(
        'DataFrame must contain label column.')
  if constants.SPLIT_KEY not in df.columns:
    raise AttributeError(
        'DataFrame must contain split column.')
  if list(df.columns) != constants.IMAGE_CSV_COLUMNS:
    raise AttributeError(
        'DataFrame column order must be {}'.format(
            constants.IMAGE_CSV_COLUMNS))


def _validate_runner(
    df: pd.DataFrame,
    runner: str,
    project: str,
    region: str,
    tfrecorder_wheel: str):
  """Validates an appropriate beam runner is chosen."""
  if runner not in ['DataflowRunner', 'DirectRunner']:
    raise AttributeError('Runner {} is not supported.'.format(runner))

  # gcs_path is a bool, true if all image paths start with gs://
  gcs_path = df[constants.IMAGE_URI_KEY].str.startswith('gs://').all()
  if (runner == 'DataflowRunner') & (not gcs_path):
    raise AttributeError('DataflowRunner requires GCS image locations.')

  if (runner == 'DataflowRunner') & (
      any(not v for v in [project, region])):
    raise AttributeError(
        'DataflowRunner requires valid `project` and `region` to be specified.'
        'The `project` is {} and `region` is {}'.format(project, region))

  if (runner == 'DataflowRunner') & (not tfrecorder_wheel):
    raise AttributeError(
        'DataflowRunner requires a tfrecorder whl file for remote execution.')


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
    header: Optional[Union[str, int, Sequence]] = 'infer',
    names: Optional[Sequence] = None) -> pd.DataFrame:
  """Returns a a Pandas DataFrame from a CSV file."""

  if header is None and not names:
    names = constants.IMAGE_CSV_COLUMNS

  with tf.io.gfile.GFile(csv_file) as f:
    return pd.read_csv(f, names=names, header=header)


def to_dataframe(
    input_data: Union[str, pd.DataFrame],
    header: Optional[Union[str, int, Sequence]] = 'infer',
    names: Optional[Sequence] = None) -> pd.DataFrame:
  """Converts `input_data` to a Pandas DataFrame."""

  if isinstance(input_data, pd.DataFrame):
    df = input_data[names] if names else input_data

  elif isinstance(input_data, str) and input_data.endswith('.csv'):
    df = read_csv(input_data, header, names)

  elif isinstance(input_data, str) and _is_directory(input_data):
    # TODO(cezequiel): Implement in phase 2
    raise NotImplementedError

  else:
    raise ValueError('Unsupported `input_data`: {}'.format(type(input_data)))

  return df


def _get_beam_metric(
    metric_filter: beam.metrics.MetricsFilter,
    result: beam.runners.runner.PipelineResult,
    metric_type: str = 'counters') -> Optional[int]:
  """Queries a beam pipeline result for a specificed metric.

  Args:
    metric_filter: an instance of apache_beam.metrics.MetricsFilter()
    metric_type: A metric type (counters, distributions, etc.)

  Returns:
    Counter value or None
  """
  query_result = result.metrics().query(metric_filter)
  result_val = None
  if query_result[metric_type]:
    result_val = query_result[metric_type][0].result
  return result_val


def _configure_logging(logfile):
  """Configures logging options."""
  # Remove default handlers that TF set for us.
  logger = logging.getLogger('')
  logger.handlers = []
  handler = logging.FileHandler(logfile)
  logger.addHandler(handler)
  logger.setLevel(constants.LOGLEVEL)
  # This disables annoying Tensorflow and TFX info/warning messages on console.
  tf_logger = logging.getLogger('tensorflow')
  tf_logger.handlers = []
  tf_logger.addHandler(handler)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

def create_tfrecords(
    input_data: Union[str, pd.DataFrame],
    output_dir: str,
    header: Optional[Union[str, int, Sequence]] = 'infer',
    names: Optional[Sequence] = None,
    runner: str = 'DirectRunner',
    project: Optional[str] = None,
    region: Optional[str] = None,
    tfrecorder_wheel: Optional[str] = None,
    dataflow_options: Optional[Dict[str, Any]] = None,
    job_label: str = 'create-tfrecords',
    compression: Optional[str] = 'gzip',
    num_shards: int = 0) -> Dict[str, Any]:
  """Generates TFRecord files from given input data.

  TFRecorder provides an easy interface to create image-based tensorflow records
  from a dataframe containing GCS locations of the images and labels.

  Usage:
    import tfrecorder

    job_id = tfrecorder.client.create_tfrecords(
        train_df,
        output_dir='gcs://foo/bar/train',
        runner='DirectFlowRunner)

  Args:
    input_data: Pandas DataFrame, CSV file or image directory path.
    output_dir: Local directory or GCS Location to save TFRecords to.
    header: Indicates row/s to use as a header. Not used when `input_data` is
      a Pandas DataFrame.
      If 'infer' (default), header is taken from the first line of a CSV
    runner: Beam runner. Can be 'DirectRunner' or 'DataFlowRunner'
    project: GCP project name (Required if DataflowRunner)
    region: GCP region name (Required if DataflowRunner)
    tfrecorder_wheel: Required for GCP Runs, path to the tfrecorder whl.
    dataflow_options: Options dict for DataflowRunner
    job_label: User supplied description for the Beam job name.
    compression: Can be 'gzip' or None for no compression.
    num_shards: Number of shards to divide the TFRecords into. Default is
        0 = no sharding.

  Returns:
    job_results: Dict
      job_id: Dataflow Job ID or 'DirectRunner'
      metrics: (optional) Beam metrics. Only used for DirectRunner
      dataflow_url: (optional) Job URL for DataflowRunner
  """

  df = to_dataframe(input_data, header, names)

  _validate_data(df)
  _validate_runner(df, runner, project, region, tfrecorder_wheel)

  logfile = os.path.join('/tmp', constants.LOGFILE)
  _configure_logging(logfile)


  integer_label = pd.api.types.is_integer_dtype(df[constants.LABEL_KEY])
  p = beam_pipeline.build_pipeline(
      df,
      job_label=job_label,
      runner=runner,
      project=project,
      region=region,
      output_dir=output_dir,
      compression=compression,
      num_shards=num_shards,
      tfrecorder_wheel=tfrecorder_wheel,
      dataflow_options=dataflow_options,
      integer_label=integer_label)

  result = p.run()

  if runner == 'DirectRunner':
    logging.info('Using DirectRunner - blocking until job completes.')
    result.wait_until_finish()

    row_count_filter = beam.metrics.MetricsFilter().with_name('row_count')
    good_image_filter = beam.metrics.MetricsFilter().with_name('image_good')
    bad_image_filter = beam.metrics.MetricsFilter().with_name('image_bad')

    row_count = _get_beam_metric(row_count_filter, result)
    good_image_count = _get_beam_metric(good_image_filter, result)
    bad_image_count = _get_beam_metric(bad_image_filter, result)

    # TODO(mikebernico): Profile metric impact with larger dataset.
    metrics = {
        'rows': row_count,
        'good_images': good_image_count,
        'bad_images': bad_image_count,
    }

    job_result = {
        'job_id': 'DirectRunner',
        'metrics': metrics
    }
    logging.info("Job Complete.")

  else:
    logging.info("Using Dataflow Runner.")
    # Construct Dataflow URL

    job_id = result.job_id()

    url = (
        constants.CONSOLE_DATAFLOW_URI +
        region +
        '/' +
        job_id +
        '?project=' +
        project)
    job_result = {
        'job_id': job_id,
        'dataflow_url': url
    }

  logging.shutdown()

  if runner == 'DataflowRunner':
    # if this is a Dataflow job, copy the logfile to GCS
    common.copy_logfile_to_gcs(logfile, output_dir)

  return job_result
