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

""" Defines input types for tfrecorder's input schema. """

import collections
from typing import Dict, List, Union

import frozendict
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

# All supported types will be based on _supported_type.
_supported_type = collections.namedtuple(
    'tfrecordInputType',
    ['type_name', 'feature_spec', 'allowed_values'],
    defaults=[None, tf.io.FixedLenFeature([], tf.string), []])

# Supported type definitions here
image_uri = _supported_type('image_uri')

allowed_split_values = ['TRAIN', 'VALIDATION', 'TEST', 'DISCARD']
split_key = _supported_type('split_key',
                            tf.io.FixedLenFeature([], tf.string),
                            allowed_split_values)
#DISCARD_INDEX = SPLIT_VALUES.index('DISCARD')

integerized_label = _supported_type('integerized_label')

# Default schema supports the legacy image_csv format.
image_csv_schema = frozendict.FrozenOrderedDict({
    'split': split_key,
    'image_uri': image_uri,
    'label': integerized_label})

def get_tft_coder(columns: List[str],
                  schema_map: Dict[str, collections.namedtuple]
                  ) -> tft.coders.CsvCoder:
  """Gets a TFT CSV Coder.

  Args:
    columns: Ordered dataframe column names, from df.column.
    schema_map: Schema map used to infer the schema.

  Returns:
    tft.coders.CsvCoder
  """
  feature_spec = dict()

  # Because the DF column name order may not match the feature_spec order
  # This maps existing column names to their feature spec (req part of
  # namedtuple)
  for col in columns:
    feature_spec[col] = schema_map[col].feature_spec


  metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(feature_spec))
  return tft.coders.CsvCoder(columns,
                             metadata.schema)


def get_split_key(
    schema_map: Dict[str, collections.namedtuple]
    ) -> Union[str, None]:
  """Gets split key name if present, otherwise returns None"""
  for k, v in schema_map.items():
    if v.type_name == 'split_key':
      return k
  return None


def get_raw_feature_spec(columns: List[str],
                         schema_map: Dict[str, collections.namedtuple]
                         ) -> Dict[str, tf.io.FixedLenFeature]:
  """Get's RAW (pre TFT) feature spec."""

  feature_spec = dict()

  # Because the DF column name order may not match the feature_spec order
  # This maps existing column names to their feature spec (req part of
  # namedtuple)
  for col in columns:
    if schema_map[col].type_name == "image_uri":
      # Modify feature_spec for extracted image, don't include image_uri.
      # TODO(mikebernico) This only works in the case where the input has
      # ONLY 1 image. Generalize to multiple images someday?
      feature_spec['image_name'] = tf.io.FixedLenFeature([], tf.string)
      feature_spec['image'] = tf.io.FixedLenFeature([], tf.string)
      feature_spec['image_height'] = tf.io.FixedLenFeature([], tf.int64)
      feature_spec['image_width'] = tf.io.FixedLenFeature([], tf.int64)
      feature_spec['image_channels'] = tf.io.FixedLenFeature([], tf.int64)
    else:
      # Copy feature as/is.
      feature_spec[col] = schema_map[col].feature_spec
  return feature_spec


def get_raw_metadata(columns: List[str],
                     schema_map: Dict[str, collections.namedtuple]
                     ) -> dataset_metadata.DatasetMetadata:
  """Get's RAW (pre TFT) schema."""
  feature_spec = get_raw_feature_spec(columns, schema_map)
  return dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(feature_spec))
