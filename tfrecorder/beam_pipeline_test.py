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

import functools
import glob
import os
import tempfile
import unittest
from unittest import mock

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam

from tfrecorder import beam_pipeline
from tfrecorder import constants
from tfrecorder import test_utils


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


class GetSplitCountsTest(unittest.TestCase):
  """Tests `get_split_counts` function."""

  def setUp(self):
    self.df = test_utils.get_test_df()

  def test_all_splits(self):
    """Tests case where train, validation and test data exists"""
    expected = {'TRAIN': 2, 'VALIDATION': 2, 'TEST': 2}
    actual = beam_pipeline.get_split_counts(self.df)
    self.assertEqual(actual, expected)

  def test_one_split(self):
    """Tests case where only one split (train) exists."""
    df = self.df[self.df.split == 'TRAIN']
    expected = {'TRAIN': 2}
    actual = beam_pipeline.get_split_counts(df)
    self.assertEqual(actual, expected)

  def test_error_no_split_key(self):
    """Tests case no split key/column exists."""
    df = self.df.drop(constants.SPLIT_KEY, axis=1)
    with self.assertRaises(AssertionError):
      beam_pipeline.get_split_counts(df)


class TransformAndWriteTfrTest(unittest.TestCase):
  """Tests `_transform_and_write_tfr` function."""

  def setUp(self):
    self.pipeline = test_utils.get_test_pipeline()
    self.raw_df = test_utils.get_raw_feature_df()
    self.temp_dir_obj = tempfile.TemporaryDirectory(dir='/tmp', prefix='test-')
    self.test_dir = self.temp_dir_obj.name
    self.tfr_writer = functools.partial(
        beam_pipeline._get_write_to_tfrecord, output_dir=self.test_dir,
        compress='gzip', num_shards=2)
    self.converter = tft.coders.CsvCoder(
        constants.RAW_FEATURE_SPEC.keys(), constants.RAW_METADATA.schema)
    self.transform_fn_path = ('./tfrecorder/test_data/sample_tfrecords')

  def tearDown(self):
    self.temp_dir_obj.cleanup()

  def _get_dataset(self, pipeline, df):
    """Returns dataset `PCollection`."""
    return (pipeline
            | beam.Create(df.values.tolist())
            | beam.ParDo(beam_pipeline.ToCSVRows())
            | beam.Map(self.converter.decode))

  def test_train(self):
    """Tests case where training data is passed."""

    with self.pipeline as p:
      with tft_beam.Context(temp_dir=os.path.join(self.test_dir, 'tmp')):
        df = self.raw_df[self.raw_df.split == 'TRAIN']
        dataset = self._get_dataset(p, df)
        transform_fn = (
            beam_pipeline._transform_and_write_tfr(
                dataset, self.tfr_writer, label='Train'))
        _ = transform_fn | tft_beam.WriteTransformFn(self.test_dir)

    self.assertTrue(
        os.path.isdir(os.path.join(self.test_dir, 'transform_fn')))
    self.assertTrue(
        os.path.isdir(os.path.join(self.test_dir, 'transformed_metadata')))
    self.assertTrue(glob.glob(os.path.join(self.test_dir, 'train*.gz')))
    self.assertFalse(glob.glob(os.path.join(self.test_dir, 'validation*.gz')))
    self.assertFalse(glob.glob(os.path.join(self.test_dir, 'test*.gz')))

  def test_non_training(self):
    """Tests case where dataset contains non-training (e.g. test) data."""

    with self.pipeline as p:
      with tft_beam.Context(temp_dir=os.path.join(self.test_dir, 'tmp')):

        df = self.raw_df[self.raw_df.split == 'TEST']
        dataset = self._get_dataset(p, df)
        transform_fn = p | tft_beam.ReadTransformFn(self.transform_fn_path)
        beam_pipeline._transform_and_write_tfr(
            dataset, self.tfr_writer, transform_fn=transform_fn,
            label='Test')

    self.assertFalse(glob.glob(os.path.join(self.test_dir, 'train*.gz')))
    self.assertFalse(glob.glob(os.path.join(self.test_dir, 'validation*.gz')))
    self.assertTrue(glob.glob(os.path.join(self.test_dir, 'test*.gz')))


if __name__ == '__main__':
  unittest.main()
