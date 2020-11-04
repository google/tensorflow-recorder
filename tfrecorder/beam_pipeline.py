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

import collections
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
from tfrecorder import input_schema
from tfrecorder import types


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
    unused_num_partitions: int = -1,
    split_key: str = 'split') -> int:
  """Returns index used to partition an element from a PCollection."""
  del unused_num_partitions
  dataset_type = element[split_key]
  if isinstance(dataset_type, bytes):
    dataset_type = element[split_key].decode('utf-8')
  try:
    index = types.SplitKey.allowed_values.index(dataset_type)
  except ValueError as e:
    logging.warning('Unable to index dataset type %s: %s.',
                    dataset_type, str(e))
    index = types.SplitKey.allowed_values.index('DISCARD')
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


def _preprocessing_fn(inputs: Dict[str, Any],
                      schema_map: Dict[str, collections.namedtuple]):
  """TensorFlow Transform preprocessing function."""

  outputs = {}
  for name, supported_type in schema_map.items():
    if supported_type == types.StringLabel:
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
      element: List[str],
      ) -> Generator[Dict[str, Any], None, None]:
    """Converts an input pandas DataFrame row (List of strings) into a flat
      column seperated row. This is necessary so that the input DataFrame
      can operate with TF Transform Coders."""
    element = ','.join([str(item) for item in element])
    self.row_count.inc()
    yield element


def get_split_counts(df: pd.DataFrame, split_key: str):
  """Returns number of rows for each data split type given dataframe."""
  assert split_key in df.columns
  return df[split_key].value_counts().to_dict()


def _transform_and_write_tfr(
    dataset: pvalue.PCollection,
    tfr_writer: Callable[[], beam.io.tfrecordio.WriteToTFRecord],
    metadata: types.BeamDatasetMetadata,
    preprocessing_fn: Optional[Callable] = None,
    transform_fn: Optional[types.TransformFn] = None,
    label: str = 'data'):
  """Applies TF Transform to dataset and outputs it as TFRecords."""

  dataset_metadata = (dataset, metadata)

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
    job_dir: str,
    runner: str,
    project: str,
    region: str,
    compression: str,
    num_shards: int,
    schema: input_schema.Schema,
    tfrecorder_wheel: str,
    dataflow_options: Dict[str, Any]) -> beam.Pipeline:
  """Runs TFRecorder Beam Pipeline.

  Args:
    df: Pandas DataFrame
    job_dir: GCS or Local Path for output.
    runner: Beam Runner: (e.g. DataflowRunner, DirectRunner).
    project: GCP project ID (if DataflowRunner)
    region: GCP compute region (if DataflowRunner)
    compression: gzip or None.
    num_shards: Number of shards.
    schema: A Schema object defining the input schema.
    tfrecorder_wheel: Path to TFRecorder wheel for DataFlow
    dataflow_options: Dataflow Runner Options (optional)

  Returns:
    beam.Pipeline

  Note: These inputs must be validated upstream (by client.create_tfrecord())
  """

  _, job_name = os.path.split(job_dir)
  options = _get_pipeline_options(
      runner,
      job_name,
      job_dir,
      project,
      region,
      tfrecorder_wheel,
      dataflow_options)

  p = beam.Pipeline(options=options)
  with tft_beam.Context(temp_dir=os.path.join(job_dir, 'tft_tmp')):

    converter = schema.get_input_coder()
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
    image_uri_key = schema.image_uri_key
    if image_uri_key:
      extract_images_fn = beam_image.ExtractImagesDoFn(image_uri_key)

      data = (
          data
          | 'ReadImage' >> beam.ParDo(extract_images_fn)
      )

    # Get the split key from schema.
    split_key = schema.split_key

    # Note: This will not always reflect actual number of samples per dataset
    # written as TFRecords. The succeeding `Partition` operation may mark
    # additional samples from other splits as discarded. If a split has all
    # its samples discarded, the pipeline will still generate a TFRecord
    # file for that split, albeit empty.
    split_counts = get_split_counts(df, split_key)

    # Require training set to be available in the input data. The transform_fn
    # and transformed_metadata will be generated from the training set and
    # applied to the other datasets, if any
    if 'TRAIN' not in split_counts:
      raise AttributeError('`TRAIN` set expected to be present in splits')

    # Split dataset into train, validation, test sets.
    partition_fn = functools.partial(_partition_fn, split_key=split_key)
    train_data, val_data, test_data, discard_data = (
        data | 'SplitDataset' >> beam.Partition(
            partition_fn, len(types.SplitKey.allowed_values)))

    preprocessing_fn = functools.partial(
        _preprocessing_fn,
        schema_map=schema.pre_tft_schema_map)

    tfr_writer = functools.partial(
        _get_write_to_tfrecord, output_dir=job_dir, compress=compression,
        num_shards=num_shards)

    pre_tft_metadata = schema.get_pre_tft_metadata()

    transform_fn = _transform_and_write_tfr(
        train_data, tfr_writer, preprocessing_fn=preprocessing_fn,
        metadata=pre_tft_metadata,
        label='Train')

    if 'VALIDATION' in split_counts:
      _transform_and_write_tfr(
          val_data, tfr_writer, transform_fn=transform_fn,
          metadata=pre_tft_metadata,
          label='Validation')

    if 'TEST' in split_counts:
      _transform_and_write_tfr(
          test_data, tfr_writer, transform_fn=transform_fn,
          metadata=pre_tft_metadata,
          label='Test')

    _ = (
        discard_data
        | 'WriteDiscardedData' >> beam.io.WriteToText(
            os.path.join(job_dir, 'discarded-data')))

    # Note: `transform_fn` already contains the transformed metadata
    _ = (transform_fn | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
        job_dir))

  return p
