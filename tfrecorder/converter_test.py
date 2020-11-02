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

"""Tests for client."""

import os
import re
from typing import List

import csv
import tempfile
import unittest

import mock
import pandas as pd
import tensorflow as tf

from tfrecorder import beam_pipeline
from tfrecorder import converter
from tfrecorder import dataset_loader
from tfrecorder import test_utils
from tfrecorder import input_schema


# pylint: disable=protected-access


class IsDirectoryTest(unittest.TestCase):
  """Tests `_is_directory`."""

  def test_local_ok(self):
    """Test function returns True on local directory."""

    with tempfile.TemporaryDirectory() as dirname:
      self.assertTrue(converter._is_directory(dirname))

  def test_local_exists_but_not_dir(self):
    """Test function returns False on local (non-directory) file."""

    with tempfile.NamedTemporaryFile(prefix='test_', dir='/tmp') as f:
      self.assertFalse(converter._is_directory(f.name))


# TODO(cezequiel): Refactor to per-function test case classes
class MiscTest(unittest.TestCase):
  """Misc tests for `client` module."""

  def setUp(self):
    self.test_df = test_utils.get_test_df()
    self.test_region = 'us-central1'
    self.test_project = 'foo'
    self.test_wheel = '/my/path/wheel.whl'

  @mock.patch.object(beam_pipeline, 'build_pipeline', autospec=True)
  def test_create_tfrecords_direct_runner(self, _):
    """Tests `create_tfrecords` Direct case."""
    r = converter.convert(
        self.test_df,
        runner='DirectRunner',
        output_dir='/tmp/direct_runner')
    self.assertCountEqual(r.keys(), ['job_id', 'metrics', 'tfrecord_dir'])
    self.assertCountEqual(
        r['metrics'].keys(), ['rows', 'good_images', 'bad_images'])

  @mock.patch.object(converter, '_get_dataflow_url')
  @mock.patch.object(beam_pipeline, 'build_pipeline')
  def test_create_tfrecords_dataflow_runner(self, mock_pipeline, mock_url):
    """Tests `create_tfrecords` Dataflow case."""
    job_id = 'foo_id'
    dataflow_url = 'http://some/job/url'
    mock_pipeline().run().job_id.return_value = job_id
    mock_url.return_value = dataflow_url
    df2 = self.test_df.copy()
    df2['image_uri'] = 'gs://' + df2['image_uri']

    outdir = '/tmp/dataflow_runner'
    os.makedirs(outdir, exist_ok=True)
    r = converter.convert(
        df2,
        runner='DataflowRunner',
        output_dir=outdir,
        region=self.test_region,
        project=self.test_project,
        tfrecorder_wheel=self.test_wheel)

    self.assertCountEqual(r.keys(), ['job_id', 'dataflow_url', 'tfrecord_dir'])
    self.assertEqual(r['job_id'], job_id)
    self.assertEqual(r['dataflow_url'], dataflow_url)
    self.assertRegex(r['tfrecord_dir'], fr'{outdir}/tfrecorder-.+-?.*')

  def test_path_split(self):
    """Tests `_path_split`."""

    filename = 'image_file.jpg'
    dirpaths = ['/path/to/image/dir/', 'gs://path/to/image/dir/']
    for dir_ in dirpaths:
      filepath = os.path.join(dir_, filename)
      act_dirpath, act_filename = converter._path_split(filepath)
      self.assertEqual(act_dirpath, dir_.rsplit('/', 1)[0])
      self.assertEqual(act_filename, filename)


class InputValidationTest(unittest.TestCase):
  """'Tests for validation input data."""

  def setUp(self):
    self.test_df = test_utils.get_test_df()
    self.test_region = 'us-central1'
    self.test_project = 'foo'
    self.test_wheel = '/my/path/wheel.whl'
    self.test_schema = input_schema.IMAGE_CSV_SCHEMA

  def test_valid_dataframe(self):
    """Tests valid DataFrame input."""
    self.assertIsNone(converter._validate_data(self.test_df, self.test_schema))

  def test_missing_image(self):
    """Tests missing image column."""
    with self.assertRaises(AttributeError):
      df2 = self.test_df.copy()
      df2.drop('image_uri', inplace=True, axis=1)
      converter._validate_data(df2, self.test_schema)

  def test_missing_label(self):
    """Tests missing label column."""
    with self.assertRaises(AttributeError):
      df2 = self.test_df.copy()
      df2.drop('label', inplace=True, axis=1)
      converter._validate_data(df2, self.test_schema)

  def test_missing_split(self):
    """Tests missing split column."""
    split_key = 'split'
    schema_keys = re.escape(
        str(list(self.test_schema.input_schema_map.keys())))
    regex = fr'^.+column: {split_key}.+keys: {schema_keys}.$'
    with self.assertRaisesRegex(AttributeError, regex):
      df2 = self.test_df.copy()
      df2.drop(split_key, inplace=True, axis=1)
      converter._validate_data(df2, self.test_schema)

  def test_valid_runner(self):
    """Tests valid runner."""
    self.assertIsNone(converter._validate_runner(
        runner='DirectRunner',
        project=self.test_project,
        region=self.test_region,
        tfrecorder_wheel=None))

  def test_invalid_runner(self):
    """Tests invalid runner."""
    with self.assertRaises(AttributeError):
      converter._validate_runner(
          runner='FooRunner',
          project=self.test_project,
          region=self.test_region,
          tfrecorder_wheel=None)


  def test_gcs_path_with_dataflow_runner_missing_param(self):
    """Tests DataflowRunner with missing required parameter."""
    for p, r in [
        (None, self.test_region), (self.test_project, None), (None, None)]:
      with self.assertRaises(AttributeError) as context:
        converter._validate_runner(
            runner='DataflowRunner',
            project=p,
            region=r,
            tfrecorder_wheel=self.test_wheel)
      self.assertTrue('DataflowRunner requires valid `project` and `region`'
                      in repr(context.exception))


  def test_gcs_path_with_dataflow_runner_missing_wheel(self):
    """Tests DataflowRunner with missing required whl path."""
    with self.assertRaises(AttributeError) as context:
      converter._validate_runner(
          runner='DataflowRunner',
          project=self.test_project,
          region=self.test_region,
          tfrecorder_wheel=None)
      self.assertTrue('requires a tfrecorder whl file for remote execution.'
                      in repr(context.exception))


def _make_csv_tempfile(data: List[List[str]]) -> tempfile.NamedTemporaryFile:
  """Returns `NamedTemporaryFile` representing an image CSV."""

  f = tempfile.NamedTemporaryFile(mode='w+t', suffix='.csv')
  writer = csv.writer(f, delimiter=',')
  for row in data:
    writer.writerow(row)
  f.seek(0)
  return f


def get_sample_image_csv_data() -> List[List[str]]:
  """Returns sample CSV data in Image CSV format."""

  data = test_utils.get_test_data()
  header = list(data.keys())
  content = [list(row) for row in zip(*data.values())]
  return [header] + content


class ReadImageDirectoryTest(unittest.TestCase):
  """Tests `_read_image_directory`."""

  def setUp(self):
    self.image_data = test_utils.get_test_df()
    self.tempfiles = []
    self.tempdir = None
    self.schema = input_schema.Schema(
        input_schema.IMAGE_CSV_SCHEMA.input_schema_map)

  def tearDown(self):
    for fp in self.tempfiles:
      fp.close()
    self.tempdir.cleanup()

  def test_normal(self):
    """Tests conversion of expected directory structure on local machine."""

    g = self.image_data.groupby([self.schema.split_key, self.schema.label_key])

    self.tempdir = tempfile.TemporaryDirectory()
    rows = []
    for (split, label), indices in g.groups.items():
      dir_ = os.path.join(self.tempdir.name, split, label)
      os.makedirs(dir_)
      for f in list(self.image_data.loc[indices, self.schema.image_uri_key]):
        _, name = os.path.split(f)
        fp = tempfile.NamedTemporaryFile(
            dir=dir_, suffix='.jpg', prefix=name)
        self.tempfiles.append(fp)
        rows.append([split, fp.name, label])

    columns = list(input_schema.IMAGE_CSV_SCHEMA.get_input_keys())
    actual = converter._read_image_directory(self.tempdir.name)
    actual.sort_values(by=columns, inplace=True)
    actual.reset_index(drop=True, inplace=True)
    expected = pd.DataFrame(rows, columns=columns)
    expected.sort_values(by=columns, inplace=True)
    expected.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(actual, expected)


class ReadCSVTest(unittest.TestCase):
  """Tests `read_csv`."""

  def setUp(self):
    data = get_sample_image_csv_data()
    self.header = data.pop(0)
    self.sample_data = data

  def test_valid_csv_no_header_no_names_specified(self):
    """Tests a valid CSV without a header and no header names given."""
    f = _make_csv_tempfile(self.sample_data)
    actual = converter.read_csv(f.name, header=None)
    self.assertEqual(
        list(actual.columns),
        list(input_schema.IMAGE_CSV_SCHEMA.get_input_keys()))
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_no_header_names_specified(self):
    """Tests valid CSV without a header, but header names are given."""
    f = _make_csv_tempfile(self.sample_data)
    actual = converter.read_csv(f.name, header=None, names=self.header)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_with_header_no_names_specified(self):
    """Tests valid CSV with header, and no header names given (inferred)."""

    f = _make_csv_tempfile([self.header] + self.sample_data)
    actual = converter.read_csv(f.name)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_with_header_names_specified(self):
    """Tests valid CSV with header, and header names given (override)."""

    f = _make_csv_tempfile([self.header] + self.sample_data)
    actual = converter.read_csv(f.name, names=self.header, header=0)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)


class ToDataFrameTest(unittest.TestCase):
  """Tests `to_dataframe`."""

  def setUp(self) -> None:
    sample_data = get_sample_image_csv_data()
    columns = sample_data.pop(0)
    self.input_df = pd.DataFrame(sample_data, columns=columns)

  @mock.patch.object(converter, 'read_csv', autospec=True)
  def test_input_csv(self, read_csv):
    """Tests valid input CSV file."""
    expected = self.input_df
    read_csv.return_value = expected
    f = _make_csv_tempfile(get_sample_image_csv_data())
    actual = converter.to_dataframe(f.name)
    pd.testing.assert_frame_equal(actual, expected)

  def test_input_dataframe_no_names_specified(self):
    """Tests valid input dataframe with no header names specified."""
    actual = converter.to_dataframe(self.input_df)
    pd.testing.assert_frame_equal(actual, self.input_df)

  def test_input_dataframe_with_header(self):
    """Tests valid input dataframe with header specified."""
    names = list(self.input_df.columns[0:-1])
    actual = converter.to_dataframe(self.input_df, names=names)
    pd.testing.assert_frame_equal(actual, self.input_df[names])

  @mock.patch.object(converter, '_read_image_directory')
  def test_input_image_dir(self, mock_fn):
    """Tests valid input image directory."""

    mock_fn.return_value = self.input_df

    with tempfile.TemporaryDirectory() as input_data:
      actual = converter.to_dataframe(input_data)
      pd.testing.assert_frame_equal(actual, self.input_df)

  def test_error_invalid_inputs(self):
    """Tests error handling with different invalid inputs."""
    inputs = [0, 'not_a_csv_file', list(), dict()]
    for input_data in inputs:
      with self.assertRaises(ValueError):
        converter.to_dataframe(input_data)


class ConvertAndLoadTest(unittest.TestCase):
  """Tests `convert_and_load`."""

  def setUp(self):
    self.tfrecord_dir = '/path/to/tfrecords'
    self.dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    self.datasets = {
        'TRAIN': self.dataset,
        'VALIDATION': self.dataset,
        'TEST': self.dataset,
    }

  @mock.patch.object(dataset_loader, 'load', autospec=True)
  @mock.patch.object(converter, 'convert', autospec=True)
  def test_convert_and_load_normal(self, convert_fn, load_fn):
    """Tests normal case."""
    convert_fn.return_value = dict(tfrecord_dir=self.tfrecord_dir)
    load_fn.return_value = self.datasets
    source = '/path/to/data.csv'
    datasets = converter.convert_and_load(source)
    self.assertEqual(datasets, self.datasets)


if __name__ == '__main__':
  unittest.main()
