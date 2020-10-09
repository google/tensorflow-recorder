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
from typing import Any, Dict, List, Union, OrderedDict

import frozendict
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tfrecorder import types

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
  allowed_values = ['TRAIN', 'VALIDATION', 'TEST', 'DISCARD']

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

class Schema:
  """Defines a TFRecorder input schema."""
  def __init__(self,
               schema_map:OrderedDict[str, SupportedType]
               ) -> None:
    """Defines TFRecorder input schema.

    Args:
      schema_map: An ordered dictionary that maps input columns to
        TFRecorder supported types.
    """
    self.split_key = None
    self.image_uri_key = None
    self.input_schema_map=schema_map
    self.pre_tft_schema_map = {}

    for k, v in schema_map.items():
      if isinstance(v, SplitKeyType):
        self.split_key = k
      if isinstance(v, ImageUriType):
        self.image_uri_key = k

        # if an image key is present, add image features to pre tft schema
        self.pre_tft_schema_map['image_name'] = ImageSupportStringType()
        self.pre_tft_schema_map['image'] = ImageSupportStringType()
        self.pre_tft_schema_map['image_height'] = ImageSupportIntType()
        self.pre_tft_schema_map['image_width'] = ImageSupportIntType()
        self.pre_tft_schema_map['image_channels'] = ImageSupportIntType()
      else:
        self.pre_tft_schema_map[k] = self.schema_map[k]

    if not self.split_key:
      raise AttributeError("Schema must contain a split key.")

  def _get_feature_spec(
      self,
      schema_map:Dict[str, SupportedType])-> Dict[str, tf.io.FixedLenFeature]:
    """Gets map of column names to tf.io.FixedLenFeatures for TFT."""
    return {k:v.feature_spec for (k,v) in schema_map}

  def _get_metadata(
      self,
      feature_spec: Dict[str, tf.io.FixedLenFeature]
      ) -> types.BeamDatasetMetadata:
    """Gets DatasetMetadata."""
    return dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

  def get_pre_tft_metadata(self) -> types.BeamDatasetMetadata:
    """Gets pre tft metadata, used by TFT extern to this class."""
    feature_spec = self._get_feature_spec(self.pre_tft_schema_map)
    return self._get_metadata(feature_spec)

  def get_input_coder(self) -> tft.coders.CsvCoder:
    """Gets input schema TFT CSV Coder."""
    feature_spec = self._get_feature_spec(self.input_schema_map)
    metadata = self._get_metadata(feature_spec)
    return tft.coders.CsvCoder(list(self.schema_map.keys()), metadata.schema)

# Built in / Default schema map.
image_csv_schema_map = frozendict.FrozenOrderedDict({
    'split': SplitKeyType(),
    'image_uri': ImageUriType(),
    'label': StringLabelType()})

ImageCsvSchema = Schema(image_csv_schema_map)
