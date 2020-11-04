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

converter.py provides create_tfrecords() to upstream clients including
the Pandas DataFrame Accessor (accessor.py) and the CLI (cli.py).
"""

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import apache_beam as beam
import pandas as pd
import tensorflow as tf

from tfrecorder import beam_pipeline
from tfrecorder import dataset_loader
from tfrecorder import constants
from tfrecorder import input_schema
from tfrecorder import types
from tfrecorder import utils


# TODO(mikebernico) Add test for only one split_key.
def _validate_data(df: pd.DataFrame, schema: input_schema.Schema):
  """Verifies data is consistent with schema."""

  for key, value in schema.input_schema_map.items():
    _ = value # TODO(mikebernico) Implement type checking.
    if key not in df.columns:
      schema_keys = list(schema.input_schema_map.keys())
      raise AttributeError(
          f'DataFrame does not contain expected column: {key}. '
          f'Ensure header matches schema keys: {schema_keys}.')


def _validate_runner(
    runner: str,
    project: str,
    region: str,
    tfrecorder_wheel: str):
  """Validates an appropriate beam runner is chosen."""
  if runner not in ['DataflowRunner', 'DirectRunner']:
    raise AttributeError('Runner {} is not supported.'.format(runner))

  if (runner == 'DataflowRunner') & (
      any(not v for v in [project, region])):
    raise AttributeError(
        'DataflowRunner requires valid `project` and `region` to be specified.'
        'The `project` is {} and `region` is {}'.format(project, region))

  if (runner == 'DataflowRunner') & (not tfrecorder_wheel):
    raise AttributeError(
        'DataflowRunner requires a tfrecorder whl file for remote execution.')


def _path_split(filepath: str) -> Tuple[str, str]:
  """Splits `filepath` into (head, tail) where `tail` part after last '/'.

  e.g.
    filepath = '/path/to/image/file.jpg'
    head, tail = _path_split(filepath)
    # head -> '/path/to/image'
    # tail -> 'file.jpg'

  Similar to `os.path.split` but supports GCS paths (prefix: gs://).
  """

  if filepath.startswith(constants.GCS_PREFIX):
    _, path = filepath.split(constants.GCS_PREFIX)
    head, tail = os.path.split(os.path.normpath(path))
    return constants.GCS_PREFIX + head, tail

  return os.path.split(filepath)


def _read_image_directory(image_dir: str) -> pd.DataFrame:
  """Reads image data from a directory into a Pandas DataFrame.

  Expected directory structure:
    image_dir/
      <dataset split>/
        <label>/
          <image file>

  Example expected directory structure:
    image_dir/
      TRAIN/
        label0/
          image_000.jpg
          image_001.jpg
          ...
        label1/
          image_100.jpg
          ...
      VALIDATION/
        ...

  Output will be based on `schema.image_csv_schema`.
  The subdirectories should only contain image files.
  See `beam_image.load` for supported image formats.
  """

  rows = []
  split_values = types.SplitKey.allowed_values
  for root, _, files in tf.io.gfile.walk(image_dir):
    if files:
      root_, label = _path_split(root)
      _, split = _path_split(root_)
      if split not in split_values:
        logging.warning('Unexpected split value: %s. Skipping %s',
                        split, root)
      # TODO(cezequiel): Add guard for non image files (e.g. .DS_Store)
      for f in files:
        image_uri = os.path.join(root, f)
        row = [split, image_uri, label]
        rows.append(row)

  return pd.DataFrame(
      rows, columns=input_schema.IMAGE_CSV_SCHEMA.get_input_keys())


def _is_directory(input_data) -> bool:
  """Returns True if `input_data` is a directory; False otherwise."""

  # Note: First check will flag if user has the necessary credentials
  # to access the directory (if it is in GCS)
  return tf.io.gfile.exists(input_data) and tf.io.gfile.isdir(input_data)


def _get_job_name(job_label: str = None) -> str:
  """Returns Beam runner job name.

  Args:
    job_label: A user defined string that helps define the job.

  Returns:
    A job name compatible with apache beam runners, including a time stamp to
      insure uniqueness.
  """

  job_name = 'tfrecorder-' + utils.get_timestamp()
  if job_label:
    job_label = job_label.replace('_', '-')
    job_name += '-' + job_label

  return job_name


def _get_job_dir(output_path: str, job_name: str) -> str:
  """Returns Beam processing job directory."""

  return os.path.join(output_path, job_name)


def _get_dataflow_url(job_id: str, project: str, region: str) -> str:
  """Returns Cloud DataFlow URL for Apache Beam job."""

  return f'{constants.CONSOLE_DATAFLOW_URI}{region}/{job_id}?=project={project}'


def read_csv(
    csv_file: str,
    header: Optional[Union[str, int, Sequence]] = 'infer',
    names: Optional[Sequence] = None) -> pd.DataFrame:
  """Returns a a Pandas DataFrame from a CSV file."""

  if header is None and not names:
    names = list(input_schema.IMAGE_CSV_SCHEMA.get_input_keys())

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
    df = _read_image_directory(input_data)

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

def convert(
    source: Union[str, pd.DataFrame],
    output_dir: str = './tfrecords',
    schema: input_schema.Schema = input_schema.IMAGE_CSV_SCHEMA,
    header: Optional[Union[str, int, Sequence]] = 'infer',
    names: Optional[Sequence] = None,
    runner: str = 'DirectRunner',
    project: Optional[str] = None,
    region: Optional[str] = None,
    tfrecorder_wheel: Optional[str] = None,
    dataflow_options: Optional[Dict[str, Any]] = None,
    job_label: str = 'convert',
    compression: Optional[str] = 'gzip',
    num_shards: int = 0) -> Dict[str, Any]:
  """Generates TFRecord files from given input data.

  TFRecorder provides an easy interface to create image-based tensorflow records
  from a dataframe containing GCS locations of the images and labels.

  Usage:
    import tfrecorder

    job_id = tfrecorder.convert(
        train_df,
        output_dir='gcs://foo/bar/train',
        runner='DirectRunner)

  Args:
    source: Pandas DataFrame, CSV file or image directory path.
    output_dir: Local directory or GCS Location to save TFRecords to.
    schema: An instance of input_schema.Schema.
    header: Indicates row/s to use as a header. Not used when `input_data` is
      a Pandas DataFrame.
      If 'infer' (default), header is taken from the first line of a CSV
    names: List of column names to use for CSV or DataFrame input.
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

  df = to_dataframe(source, header, names)

  _validate_data(df, schema)
  _validate_runner(runner, project, region, tfrecorder_wheel)

  logfile = os.path.join('/tmp', constants.LOGFILE)
  _configure_logging(logfile)

  job_name = _get_job_name(job_label)
  job_dir = _get_job_dir(output_dir, job_name)

  p = beam_pipeline.build_pipeline(
      df,
      job_dir=job_dir,
      runner=runner,
      project=project,
      region=region,
      compression=compression,
      num_shards=num_shards,
      schema=schema,
      tfrecorder_wheel=tfrecorder_wheel,
      dataflow_options=dataflow_options,
  )

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

  elif runner == 'DataflowRunner':
    logging.info("Using Dataflow Runner.")
    job_id = result.job_id()
    url = _get_dataflow_url(job_id, project, region)
    job_result = {
        'job_id': job_id,
        'dataflow_url': url,
    }
    # Copy the logfile to GCS output dir
    utils.copy_logfile_to_gcs(logfile, output_dir)

  else:
    raise ValueError(f'Unsupported runner: {runner}')

  job_result['tfrecord_dir'] = job_dir

  return job_result


def convert_and_load(*args, **kwargs):
  """Converts data into TFRecords and loads them as TF Datasets."""

  job_result = convert(*args, **kwargs)
  return dataset_loader.load(job_result['tfrecord_dir'])
