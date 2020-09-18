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

from typing import Any, Callable, Dict, Generator, List, Optional, Union

import functools
import logging
import os

import apache_beam as beam
from apache_beam import pvalue
import pandas as pd
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam

from tfrecorder import beam_image
from tfrecorder import common
from tfrecorder import constants
from tfrecorder import types


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
    tfrecorder_wheel: str,
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
    options_dict['extra_packages'] = [tfrecorder_wheel]
  if dataflow_options:
    options_dict.update(dataflow_options)

  return beam.pipeline.PipelineOptions(flags=[], **options_dict)


def _partition_fn(
    element: Dict[str, str],
    unused_num_partitions: int = -1) -> int:
  """Returns index used to partition an element from a PCollection."""
  del unused_num_partitions
  dataset_type = element[constants.SPLIT_KEY].decode('utf-8')
  try:
    index = constants.SPLIT_VALUES.index(dataset_type)
  except ValueError as e:
    logging.warning('Unable to index dataset type %s: %s.',
                    dataset_type, str(e))
    index = constants.DISCARD_INDEX
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

def _preprocessing_fn(inputs: Dict[str, Any], integer_label: bool = False):
  """TensorFlow Transform preprocessing function."""

  outputs = inputs.copy()

  if not integer_label:
    # Integerize string labels, if present.
    outputs[constants.LABEL_KEY] = tft.compute_and_apply_vocabulary(
        outputs[constants.LABEL_KEY])

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
      element: List[str],
      ) -> Generator[Dict[str, Any], None, None]:
    """Loads image and creates image features.

    This DoFn extracts an image being stored on local disk or GCS and
    yields a base64 encoded image, the image height, image width, and channels.
    """
    element = ','.join([str(item) for item in element])
    self.row_count.inc()
    yield element


def get_split_counts(df: pd.DataFrame):
  """Returns number of rows for each data split type given dataframe."""
  assert constants.SPLIT_KEY in df.columns
  return df[constants.SPLIT_KEY].value_counts().to_dict()


def _transform_and_write_tfr(
    dataset: pvalue.PCollection,
    tfr_writer: Callable = None,
    preprocessing_fn: Optional[Callable] = None,
    transform_fn: Optional[types.TransformFn] = None,
    label: str = 'data'):
  """Applies TF Transform to dataset and outputs it as TFRecords."""

  dataset_metadata = (dataset, constants.RAW_METADATA)

  if transform_fn:
    transformed_dataset, transformed_metadata = (
        (dataset_metadata, transform_fn)
        | f'Transform{label}' >> tft_beam.TransformDataset())
  else:
    if not preprocessing_fn:
      preprocessing_fn = lambda x: x
    (transformed_dataset, transformed_metadata), transform_fn = (
        dataset_metadata
        | f'AnalyzeAndTransform{label}' >>
        tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

  transformed_data_coder = tft.coders.ExampleProtoCoder(
      transformed_metadata.schema)
  _ = (
      transformed_dataset
      | f'Encode{label}' >> beam.Map(transformed_data_coder.encode)
      | f'Write{label}' >> tfr_writer(prefix=label.lower()))

  return transform_fn


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
    tfrecorder_wheel: str,
    dataflow_options: dict,
    integer_label: bool) -> beam.Pipeline:
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
    tfrecorder_wheel: Path to TFRecorder wheel for DataFlow
    dataflow_options: Dataflow Runner Options (optional)
    integer_label: Flags if label is already an integer.

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
      tfrecorder_wheel,
      dataflow_options)

  #with beam.Pipeline(runner, options=options) as p:
  p = beam.Pipeline(options=options)
  with tft_beam.Context(temp_dir=os.path.join(job_dir, 'tft_tmp')):

    converter = tft.coders.CsvCoder(constants.IMAGE_CSV_COLUMNS,
                                    constants.IMAGE_CSV_METADATA.schema)

    extract_images_fn = beam_image.ExtractImagesDoFn(constants.IMAGE_URI_KEY)
    flatten_rows = ToCSVRows()

    # Each element in the image_csv_data PCollection will be a dict
    # including the image_csv_columns and the image features created from
    # extract_images_fn.
    image_csv_data = (
        p
        | 'ReadFromDataFrame' >> beam.Create(df.values.tolist())
        | 'ToCSVRows' >> beam.ParDo(flatten_rows)
        | 'DecodeCSV' >> beam.Map(converter.decode)
        | 'ReadImage' >> beam.ParDo(extract_images_fn)
    )

    # Note: This will not always reflect actual number of samples per dataset
    # written as TFRecords. The succeeding `Partition` operation may mark
    # additional samples from other splits as discarded. If a split has all
    # its samples discarded, the pipeline will still generate a TFRecord
    # file for that split, albeit empty.
    split_counts = get_split_counts(df)

    # Require training set to be available in the input data. The transform_fn
    # and transformed_metadata will be generated from the training set and
    # applied to the other datasets, if any
    assert 'TRAIN' in split_counts

    train_data, val_data, test_data, discard_data = (
        image_csv_data | 'SplitDataset' >> beam.Partition(
            _partition_fn, len(constants.SPLIT_VALUES))
    )

    preprocessing_fn = functools.partial(
        _preprocessing_fn,
        integer_label=integer_label)

    tfr_writer = functools.partial(
        _get_write_to_tfrecord, output_dir=job_dir, compress=compression,
        num_shards=num_shards)
    transform_fn = _transform_and_write_tfr(
        train_data, tfr_writer, preprocessing_fn=preprocessing_fn,
        label='Train')

    if 'VALIDATION' in split_counts:
      _transform_and_write_tfr(
          val_data, tfr_writer, transform_fn=transform_fn, label='Validation')

    if 'TEST' in split_counts:
      _transform_and_write_tfr(
          test_data, tfr_writer, transform_fn=transform_fn, label='Test')

    _ = (
        discard_data
        | 'WriteDiscardedData' >> beam.io.WriteToText(
            os.path.join(job_dir, 'discarded-data')))

    # Note: `transform_fn` already contains the transformed metadata
    _ = (transform_fn | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
        job_dir))

  return p
