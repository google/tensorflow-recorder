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

from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tfrecorder import types



class Schema:
  """Defines a TFRecorder input schema."""
  def __init__(self, schema_map: Dict[str, types.SupportedType]) -> None:
    """Defines TFRecorder input schema.

    Args:
      schema_map: An ordered dictionary that maps input columns to
        TFRecorder supported types.
    """
    self.split_key = None
    self.image_uri_key = None
    self.label_key = None
    self.input_schema_map = schema_map
    self.pre_tft_schema_map = {}

    for k, v in schema_map.items():
      if v == types.SplitKey:
        self.split_key = k
      if 'Label' in v.__name__: # Matches any label type
        self.label_key = k

      if v == types.ImageUri:
        self.image_uri_key = k
        # if an image key is present, add image features to pre tft schema
        self.pre_tft_schema_map['image_name'] = types.ImageSupportString
        self.pre_tft_schema_map['image'] = types.ImageSupportString
        self.pre_tft_schema_map['image_height'] = types.ImageDim
        self.pre_tft_schema_map['image_width'] = types.ImageDim
        self.pre_tft_schema_map['image_channels'] = types.ImageDim
      else:
        self.pre_tft_schema_map[k] = schema_map[k]

    if not self.split_key:
      raise AttributeError("Schema must contain a split key.")

  @staticmethod
  def _get_feature_spec(
      schema_map: Dict[str, types.SupportedType]
      ) -> Dict[str, tf.io.FixedLenFeature]:
    """Gets map of column names to tf.io.FixedLenFeatures for TFT."""
    return {k: v.feature_spec for k, v in schema_map.items()}

  @staticmethod
  def _get_metadata(
      feature_spec: Dict[str, tf.io.FixedLenFeature]
      ) -> types.BeamDatasetMetadata:
    """Gets DatasetMetadata."""
    return dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

  def get_pre_tft_metadata(self) -> types.BeamDatasetMetadata:
    """Gets pre TFT metadata, used by TFT external to this class."""
    feature_spec = self._get_feature_spec(self.pre_tft_schema_map)
    return self._get_metadata(feature_spec)

  def get_input_coder(self) -> tft.coders.CsvCoder:
    """Gets input schema TFT CSV Coder."""
    feature_spec = self._get_feature_spec(self.input_schema_map)
    metadata = self._get_metadata(feature_spec)
    return tft.coders.CsvCoder(list(self.input_schema_map.keys()),
                               metadata.schema)

  def get_input_keys(self):
    """Returns keys for input_schema_map as list."""
    return self.input_schema_map.keys()

# Built in / Default schema map.
image_csv_schema_map = {
    'split': types.SplitKey,
    'image_uri': types.ImageUri,
    'label': types.StringLabel}

IMAGE_CSV_SCHEMA = Schema(image_csv_schema_map)
