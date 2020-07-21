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

"""Tests for beam_pipeline."""

import os
import unittest
from unittest import mock

import apache_beam as beam
import tensorflow as tf

from tfrecorder import beam_pipeline


# pylint: disable=protected-access

class BeamPipelineTests(unittest.TestCase):
  """Tests for beam_pipeline.py"""

  def test_processing_fn_with_int_label(self):
    'Test preprocessing fn with integer label.'
    element = {
        'split': 'TRAIN',
        'image_uri': 'gs://foo/bar.jpg',
        'label': 1}
    result = beam_pipeline._preprocessing_fn(element, integer_label=True)
    self.assertEqual(element, result)

  @mock.patch('tfrecorder.beam_pipeline.tft')
  def test_processing_fn_with_string_label(self, mock_transform):
    'Test preprocessing fn with string label.'
    mock_transform.compute_and_apply_vocabulary.return_value = tf.constant(
        0, dtype=tf.int64)
    element = {
        'split': 'TRAIN',
        'image_uri': 'gs://foo/bar.jpg',
        'label': tf.constant('cat', dtype=tf.string)}
    result = beam_pipeline._preprocessing_fn(element, integer_label=False)
    result['label'] = result['label'].numpy()
    self.assertEqual(0, result['label'])

  def test_write_to_tfrecord(self):
    """Test _write_to_tfrecord() fn."""
    tfr_writer = beam_pipeline._get_write_to_tfrecord(
        output_dir='tmp',
        prefix='foo',
        compress=True,
        num_shards=2)
    self.assertIsInstance(tfr_writer, beam.io.tfrecordio.WriteToTFRecord)

  def test_partition_fn(self):
    """Test the partition function."""

    test_data = {
        'split': 'update_me',
        'image_uri': 'gs://foo/bar0.jpg',
        'label': 1}

    for i, part in enumerate(['TRAIN', 'VALIDATION', 'TEST', 'FOO']):
      test_data['split'] = part.encode('utf-8')
      index = beam_pipeline._partition_fn(test_data)

      self.assertEqual(
          index, i,
          '{} should be index {} but was index {}'.format(part, i, index))

  def test_get_setup_py_filepath(self):
    """Tests `_get_setup_py_filepath`."""
    filepath = beam_pipeline._get_setup_py_filepath()
    self.assertTrue(os.path.isfile(filepath))
    self.assertTrue(os.path.isabs(filepath))


if __name__ == '__main__':
  unittest.main()
