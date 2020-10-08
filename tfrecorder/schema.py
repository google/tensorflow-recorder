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

"""Defines input types for TFRecorder's input schema."""

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import frozendict
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

@dataclass
class SupportedType:
  """Base type for TFRecorder Types."""
  feature_spec: tf.io.FixedLenFeature
  allowed_values: List(Any)

@dataclass
class ImageUriType(SupportedType, frozen=True):
  """Supports image uri columns."""
  feature_spec=tf.io.FixedLenFeature([], tf.string)
  allowed_values=[]

@dataclass
class SplitKeyType(SupportedType, frozen=True):
  """Supports split key columns."""
  feature_spec=tf.io.FixedLenFeature([], tf.string)
  allowed_split_values = ['TRAIN', 'VALIDATION', 'TEST', 'DISCARD']

@dataclass
class IntegerInputType(SupportedType, frozen=True):
  """Supports integer columns."""
  feature_spec=tf.io.FixedLenFeature([], tf.int64)
  allowed_values=[]

@dataclass
class FloatInputType (SupportedType, frozen=True):
  """Supports float columns."""
  feature_spec=tf.io.FixedLenFeature([], tf.float64)
  allowed_values=[]

#TODO(mikebernico): Implement in preprocess_fn
@dataclass
class StringInputType(SupportedType, frozen=True):
  """Supports string input columns."""
  feature_spec=tf.io.FixedLenFeature([], tf.string)
  allowed_values=[]

@dataclass
class IntegerLabelType(SupportedType, frozen=True):
  """Supports integer labels."""
  feature_spec=tf.io.FixedLenFeature([], tf.int64)
  allowed_values=[]

@dataclass
class StringLabelType(SupportedType, frozen=True):
  """Supports string labels."""
  feature_spec=tf.io.FixedLenFeature([], tf.string)
  allowed_values=[]

@dataclass
class ImageSupportStringType(SupportedType, frozen=True):
  """Supports generated image bytestrings."""
  feature_spec=tf.io.FixedLenFeature([], tf.string)
  allowed_values=[]

@dataclass
class ImageSupportIntType(SupportedType, frozen=True):
  """Supports generated image ints (height, width, channels)."""
  feature_spec=tf.io.FixedLenFeature([], tf.int64)
  allowed_values=[]

# TODO(mikebernico): Refactor schema_map to a container class.
# Default schema supports the legacy image_csv format.
SchemaMap = Dict[str, SupportedType]

image_csv_schema = frozendict.FrozenOrderedDict({
    'split': SplitKeyType,
    'image_uri': ImageUriType,
    'label': StringLabelType})


def get_raw_schema_map(
    schema_map: Dict[str, collections.namedtuple]
    ) -> Dict[str, collections.namedtuple]:
  """Converts a schema to a raw (pre TFT / post image extraction) schema."""
  raw_schema = {}
  for k, v in schema_map.items():
    if v.type_name == 'image_uri':
      raw_schema['image_name'] = ImageSupportStringType
      raw_schema['image'] = ImageSupportStringType
      raw_schema['image_height'] = ImageSupportIntType
      raw_schema['image_width'] = ImageSupportIntType
      raw_schema['image_channels'] = ImageSupportIntType
    else:
      raw_schema[k] = schema_map[k]
  return raw_schema


def get_tft_coder(
    columns: List[str],
    schema_map: Dict[str, collections.namedtuple]
    ) -> tft.coders.CsvCoder:
  """Gets a TFT CSV Coder.

  Args:
    columns: Ordered DataFrame column names, from df.column.
    schema_map: Schema map used to infer the schema.

  Returns:
    tft.coders.CsvCoder
  """
  feature_spec = {}
  # Because the DF column name order may not match the feature_spec order
  # This maps existing column names to their feature spec (required part of
  # namedtuple)
  for col in columns:
    feature_spec[col] = schema_map[col].feature_spec

  metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(feature_spec))

  return tft.coders.CsvCoder(columns, metadata.schema)


def get_key(
    type_: SupportedType,
    schema_map: Dict[str, collections.namedtuple]) -> Union[str, None]:
  """Gets first instance of key of type 'type_name' from schema map.

  Returns key name if present, otherwise returns None.
  """
  #TODO(mikebernico): Fix so that multiples of a key type work in future.
  for k, v in schema_map.items():
    if v.type_name == type_.type_name:
      return k
  return None


def get_raw_feature_spec(columns: List[str],
                         schema_map: Dict[str, collections.namedtuple]
                         ) -> Dict[str, tf.io.FixedLenFeature]:
  """Gets RAW (pre TFT) feature spec."""

  feature_spec = dict()

  # Because the DF column name order may not match the feature_spec order
  # this maps existing column names to their feature spec (req part of
  # namedtuple)
  for col in columns:
    if schema_map[col].type_name == 'image_uri':
      # Modify feature_spec for extracted image, don't include image_uri.
      # TODO(mikebernico) This only works in the case where the input has
      # ONLY 1 image. Generalize to multiple images someday?
      feature_spec['image_name'] = tf.io.FixedLenFeature([], tf.string)
      feature_spec['image'] = tf.io.FixedLenFeature([], tf.string)
      feature_spec['image_height'] = tf.io.FixedLenFeature([], tf.int64)
      feature_spec['image_width'] = tf.io.FixedLenFeature([], tf.int64)
      feature_spec['image_channels'] = tf.io.FixedLenFeature([], tf.int64)
    else:
      # Copy feature as-is.
      feature_spec[col] = schema_map[col].feature_spec
  return feature_spec


def get_raw_metadata(columns: List[str],
                     schema_map: Dict[str, collections.namedtuple]
                     ) -> dataset_metadata.DatasetMetadata:
  """Returns metadata prior to TF Transform preprocessing

  Note: takes base schema_map as input, not raw_schema_map.
  """
  feature_spec = get_raw_feature_spec(columns, schema_map)
  return dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(feature_spec))
