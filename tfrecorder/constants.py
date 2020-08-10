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

"""Global constants."""

import logging

import frozendict
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils


SPLIT_KEY = 'split'
SPLIT_VALUES = ['TRAIN', 'VALIDATION', 'TEST', 'DISCARD']
DISCARD_INDEX = SPLIT_VALUES.index('DISCARD')
IMAGE_URI_KEY = 'image_uri'
LABEL_KEY = 'label'
IMAGE_CSV_COLUMNS = [SPLIT_KEY, IMAGE_URI_KEY, LABEL_KEY]

IMAGE_CSV_FEATURE_SPEC = frozendict.FrozenOrderedDict({
    SPLIT_KEY: tf.io.FixedLenFeature([], tf.string),
    IMAGE_URI_KEY: tf.io.FixedLenFeature([], tf.string),
    LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
})

RAW_FEATURE_SPEC = frozendict.FrozenOrderedDict({
    SPLIT_KEY: tf.io.FixedLenFeature([], tf.string),
    LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_height': tf.io.FixedLenFeature([], tf.int64),
    'image_width': tf.io.FixedLenFeature([], tf.int64),
    'image_channels':  tf.io.FixedLenFeature([], tf.int64)
})

IMAGE_CSV_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(IMAGE_CSV_FEATURE_SPEC))

RAW_METADATA = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(RAW_FEATURE_SPEC))

LOGFILE = 'tfrecorder-beam.log'
LOGLEVEL = logging.INFO

CONSOLE_DATAFLOW_URI = "https://console.cloud.google.com/dataflow/jobs/"
