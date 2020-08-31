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

"""TFRecorder Beam Pipeline.

This file implements the full Beam pipeline for TFRecorder.
"""

from typing import Any, Dict, Generator, Union
import collections
import functools
import logging
import os

import apache_beam as beam
import pandas as pd
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam

from tfrecorder import beam_image
from tfrecorder import common
from tfrecorder import schema


def _get_setup_py_filepath() -> str:
  """Returns the file path to the setup.py file.

  The location of the setup.py file is needed to run Dataflow jobs.
  """

  return os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '..', 'setup.py')


def _get_job_name(job_label: str = None) -> str:
  """Returns Beam runner job name.

  Args:
    job_label: A user defined string that helps define the job.

  Returns:
    A job name compatible with apache beam runners, including a time stamp to
      insure uniqueness.
  """

  job_name = 'tfrecorder-' + common.get_timestamp()
  if job_label:
    job_label = job_label.replace('_', '-')
    job_name += '-' + job_label

  return job_name


def _get_job_dir(output_path: str, job_name: str) -> str:
  """Returns Beam processing job directory."""

  return os.path.join(output_path, job_name)


def _get_pipeline_options(
    runner: str,
    job_name: str,
    job_dir: str,
    project: str,
    region: str,
    dataflow_options: Union[Dict[str, Any], None]
    ) -> beam.pipeline.PipelineOptions:
  """Returns Beam pipeline options."""

  options_dict = {
      'runner': runner,
      'staging_location': os.path.join(job_dir, 'staging'),
      'temp_location': os.path.join(job_dir, 'tmp'),
      'job_name': job_name,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'save_main_session': True,
      'pipeline_type_check': False,
  }

  if project:
    options_dict['project'] = project
  if region:
    options_dict['region'] = region
  if runner == 'DataflowRunner':
    options_dict['setup_file'] = _get_setup_py_filepath()
  if dataflow_options:
    options_dict.update(dataflow_options)

  return beam.pipeline.PipelineOptions(flags=[], **options_dict)


def _partition_fn(
    element: Dict[str, str],
    unused_num_partitions: int = -1,
    split_key: str = 'split') -> int:
  """Returns index used to partition an element from a PCollection."""
  del unused_num_partitions
  dataset_type = element[split_key].decode('utf-8')
  try:
    index = schema.split_key.allowed_values.index(dataset_type)
  except ValueError as e:
    logging.warning('Unable to index dataset type %s: %s.',
                    dataset_type, str(e))
    index = schema.split_key.allowed_values.index('DISCARD')
  return index

def _get_write_to_tfrecord(output_dir: str,
                           prefix: str,
                           compress: bool = True,
                           num_shards: int = 0) \
                           -> beam.io.tfrecordio.WriteToTFRecord:
  """Returns `beam.io.tfrecordio.WriteToTFRecord` object.

  This configures a Beam sink to output TFRecord files.

  Args:
    output_dir: Directory to output TFRecord files.
    prefix: TFRecord file prefix.
    compress: If True, GZip compress TFRecord files.
    num_shards: Number of file shards to split the TFRecord data.
  """

  path = os.path.join(output_dir, prefix)
  suffix = '.tfrecord'
  if compress:
    compression_type = 'gzip'
    suffix += '.gz'
  else:
    compression_type = 'uncompressed'

  return beam.io.tfrecordio.WriteToTFRecord(
      path,
      file_name_suffix=suffix,
      compression_type=compression_type,
      num_shards=num_shards,
  )

def _preprocessing_fn(inputs, schema_map):
  """TensorFlow Transform preprocessing function."""

  outputs = {}
  for name, supported_type in schema_map.items():
    if supported_type.type_name == 'string_label':
      outputs[name] = tft.compute_and_apply_vocabulary(inputs[name])
    else:
      outputs[name] = inputs[name]
  return outputs


# pylint: disable=abstract-method

class ToCSVRows(beam.DoFn):
  """Adds image to PCollection."""

  def __init__(self):
    """Constructor."""
    super().__init__()
    self.row_count = beam.metrics.Metrics.counter(self.__class__, 'row_count')


  # pylint: disable=unused-argument
  # pylint: disable=arguments-differ
  def process(
      self,
      element: Dict[str, Any]
      ) -> Generator[Dict[str, Any], None, None]:
    """Converts a pandas dataframe flat, column seperated row."""
    element = ','.join([str(item) for item in element])
    self.row_count.inc()
    yield element


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def build_pipeline(
    df: pd.DataFrame,
    job_label: str,
    runner: str,
    project: str,
    region: str,
    output_dir: str,
    compression: str,
    num_shards: int,
    schema_map: Dict[str, collections.namedtuple],
    dataflow_options: dict) -> beam.Pipeline:
  """Runs TFRecorder Beam Pipeline.

  Args:
    df: Pandas DataFrame
    job_label: User description for the beam job.
    runner: Beam Runner: (e.g. DataflowRunner, DirectRunner).
    project: GCP project ID (if DataflowRunner)
    region: GCP compute region (if DataflowRunner)
    output_dir: GCS or Local Path for output.
    compression: gzip or None.
    num_shards: Number of shards.
    schema_map: A schema map used to derive the input and target schema.
    dataflow_options: Dataflow Runner Options (optional)

  Returns:
    beam.Pipeline

  Note: These inputs must be validated upstream (by client.create_tfrecord())
  """

  job_name = _get_job_name(job_label)
  job_dir = _get_job_dir(output_dir, job_name)
  options = _get_pipeline_options(
      runner,
      job_name,
      job_dir,
      project,
      region,
      dataflow_options)

  p = beam.Pipeline(options=options)
  with tft_beam.Context(temp_dir=os.path.join(job_dir, 'tft_tmp')):

    converter = schema.get_tft_coder(df.columns, schema_map)
    flatten_rows = ToCSVRows()

    # Each element in the data PCollection will be a dict
    # including the image_csv_columns and the image features created from
    # extract_images_fn.
    data = (
        p
        | 'ReadFromDataFrame' >> beam.Create(df.values.tolist())
        | 'ToCSVRows' >> beam.ParDo(flatten_rows)
        | 'DecodeCSV' >> beam.Map(converter.decode)
    )

    # Extract images if an image_uri key exists.
    image_uri_key = schema.get_key(type_name='image_uri', schema_map=schema_map)
    if image_uri_key:
      extract_images_fn = beam_image.ExtractImagesDoFn(image_uri_key)

      data = (
          data
          | 'ReadImage' >> beam.ParDo(extract_images_fn)
      )

    # If the schema contains a valid split key, we are partitioning the dataset.
    split_key = schema.get_key(type_name='split_key', schema_map=schema_map)

    # Raw metadata is the metadata after image insertion but before TFT
    raw_metadata = schema.get_raw_metadata(df.columns, schema_map)

    if split_key:
      # Split dataset into train, validation, test.
      partition_fn = functools.partial(_partition_fn, split_key=split_key)
      train_data, val_data, test_data, discard_data = (
          data | 'SplitDataset' >> beam.Partition(
              partition_fn, len(schema.split_key.allowed_values)))

      train_dataset = (train_data, raw_metadata)
      val_dataset = (val_data, raw_metadata)
      test_dataset = (test_data, raw_metadata)
    else:
      train_dataset = (data, raw_metadata)

    # TensorFlow Transform applied to all datasets.
    raw_schema_map = schema.get_raw_schema_map(schema_map=schema_map)
    preprocessing_fn = functools.partial(
        _preprocessing_fn,
        schema_map=raw_schema_map)
    transformed_train_dataset, transform_fn = (
        train_dataset
        | 'AnalyzeAndTransformTrain' >> tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn))

    transformed_train_data, transformed_metadata = transformed_train_dataset
    transformed_data_coder = tft.coders.ExampleProtoCoder(
        transformed_metadata.schema)

    transformed_val_data, _ = (
        (val_dataset, transform_fn)
        | 'TransformVal' >> tft_beam.TransformDataset()
    )

    transformed_test_data, _ = (
        (test_dataset, transform_fn)
        | 'TransformTest' >> tft_beam.TransformDataset()
    )

    # Sinks for TFRecords and metadata.
    tfr_writer = functools.partial(_get_write_to_tfrecord,
                                   output_dir=job_dir,
                                   compress=compression,
                                   num_shards=num_shards)

    _ = (
        transformed_train_data
        | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
        | 'WriteTrainData' >> tfr_writer(prefix='train'))

  # VAL and TEST aren only written if a split key is present.
  if split_key:
    _ = (
        transformed_val_data
        | 'EncodeValData' >> beam.Map(transformed_data_coder.encode)
        | 'WriteValData' >> tfr_writer(prefix='val'))

    _ = (
        transformed_test_data
        | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
        | 'WriteTestData' >> tfr_writer(prefix='test'))

    _ = (
        discard_data
        | 'DiscardDataWriter' >> beam.io.WriteToText(
            os.path.join(job_dir, 'discarded-data')))

  # Output transform function and metadata
  _ = (transform_fn | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
      job_dir))

  # Output metadata schema
  _ = (transformed_metadata | 'WriteMetadata' >> tft_beam.WriteMetadata(
      job_dir, pipeline=p))

  return p
