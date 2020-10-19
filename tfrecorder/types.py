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

"""Custom types."""

import dataclasses
from typing import Tuple, List, Any

import tensorflow as tf
from apache_beam.pvalue import PCollection
from tensorflow_transform import beam as tft_beam

BeamDatasetMetadata = tft_beam.tft_beam_io.beam_metadata_io.BeamDatasetMetadata
TransformedMetadata = BeamDatasetMetadata
TransformFn = Tuple[PCollection, TransformedMetadata]


@dataclasses.dataclass
class SupportedType:
  """Base type for TFRecorder Types."""
  feature_spec: tf.io.FixedLenFeature
  allowed_values: List[Any]


@dataclasses.dataclass
class ImageUri(SupportedType):
  """Supports image uri columns."""
  feature_spec = tf.io.FixedLenFeature([], tf.string)
  allowed_values = []


@dataclasses.dataclass
class SplitKey(SupportedType):
  """Supports split key columns."""
  feature_spec = tf.io.FixedLenFeature([], tf.string)
  allowed_values = ['TRAIN', 'VALIDATION', 'TEST', 'DISCARD']


@dataclasses.dataclass
class IntegerInput(SupportedType):
  """Supports integer columns."""
  feature_spec = tf.io.FixedLenFeature([], tf.int64)
  allowed_values = []


@dataclasses.dataclass
class FloatInput(SupportedType):
  """Supports float columns."""
  feature_spec = tf.io.FixedLenFeature([], tf.float32)
  allowed_values = []


#TODO(mikebernico): Implement in preprocess_fn
@dataclasses.dataclass
class StringInput(SupportedType):
  """Supports string input columns."""
  feature_spec = tf.io.FixedLenFeature([], tf.string)
  allowed_values = []


@dataclasses.dataclass
class IntegerLabel(IntegerInput):
  """Supports integer labels."""


@dataclasses.dataclass
class StringLabel(StringInput):
  """Supports string labels."""


@dataclasses.dataclass
class ImageSupportString(StringInput):
  """Supports generated image bytestrings."""


@dataclasses.dataclass
class ImageDim(IntegerInput):
  """Supports generated image ints (height, width, channels)."""
