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
from typing import List

import csv
import tempfile
import unittest

import mock
import pandas as pd

from tfrecorder import client
from tfrecorder import test_utils
from tfrecorder import input_schema


# pylint: disable=protected-access


class ClientTest(unittest.TestCase):
  """Misc tests for `client` module."""

  def setUp(self):
    self.test_df = test_utils.get_test_df()
    self.test_region = 'us-central1'
    self.test_project = 'foo'
    self.test_wheel = '/my/path/wheel.whl'

  @mock.patch('tfrecorder.client.beam_pipeline')
  def test_create_tfrecords_direct_runner(self, mock_beam):
    """Tests `create_tfrecords` Direct case."""
    mock_beam.build_pipeline().run().wait_until_finished.return_value = {
        'rows':6}
    r = client.create_tfrecords(
        self.test_df,
        runner='DirectRunner',
        output_dir='/tmp/direct_runner')
    self.assertTrue('metrics' in r)

  @mock.patch('tfrecorder.client.beam_pipeline')
  def test_create_tfrecords_dataflow_runner(self, mock_beam):
    """Tests `create_tfrecords` Dataflow case."""
    mock_beam.build_pipeline().run().job_id.return_value = 'foo_id'

    df2 = self.test_df.copy()
    df2['image_uri'] = 'gs://' + df2['image_uri']

    outdir = '/tmp/dataflow_runner'

    expected = {
        'job_id': 'foo_id',
        'dataflow_url': 'https://console.cloud.google.com/dataflow/jobs/' +
                        'us-central1/foo_id?project=foo'}

    os.makedirs(outdir, exist_ok=True)
    r = client.create_tfrecords(
        df2,
        runner='DataflowRunner',
        output_dir=outdir,
        region=self.test_region,
        project=self.test_project,
        tfrecorder_wheel=self.test_wheel)
    self.assertEqual(r, expected)

  def test_path_split(self):
    """Tests `_path_split`."""

    filename = 'image_file.jpg'
    dirpaths = ['/path/to/image/dir', 'gs://path/to/image/dir']
    for dir_ in dirpaths:
      filepath = os.path.join(dir_, filename)
      act_dirpath, act_filename = client._path_split(filepath)
      self.assertEqual(act_dirpath, dir_)
      self.assertEqual(act_filename, filename)


class InputValidationTest(unittest.TestCase):
  """'Tests for validation input data."""

  def setUp(self):
    self.test_df = test_utils.get_test_df()
    self.test_region = 'us-central1'
    self.test_project = 'foo'
    self.test_wheel = '/my/path/wheel.whl'
    self.test_schema_map = input_schema.IMAGE_CSV_SCHEMA.input_schema_map

  def test_valid_runner(self):
    """Tests valid runner."""
    self.assertIsNone(client._validate_runner(
        runner='DirectRunner',
        project=self.test_project,
        region=self.test_region,
        tfrecorder_wheel=None))

  def test_invalid_runner(self):
    """Tests invalid runner."""
    with self.assertRaises(AttributeError):
      client._validate_runner(
          runner='FooRunner',
          project=self.test_project,
          region=self.test_region,
          tfrecorder_wheel=None)


  def test_gcs_path_with_dataflow_runner_missing_param(self):
    """Tests DataflowRunner with missing required parameter."""
    for p, r in [
        (None, self.test_region), (self.test_project, None), (None, None)]:
      with self.assertRaises(AttributeError) as context:
        client._validate_runner(
            runner='DataflowRunner',
            project=p,
            region=r,
            tfrecorder_wheel=self.test_wheel)
      self.assertTrue('DataflowRunner requires valid `project` and `region`'
                      in repr(context.exception))


  def test_gcs_path_with_dataflow_runner_missing_wheel(self):
    """Tests DataflowRunner with missing required whl path."""
    with self.assertRaises(AttributeError) as context:
      client._validate_runner(
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
    actual = client._read_image_directory(self.tempdir.name)
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
    actual = client.read_csv(f.name, header=None)
    self.assertEqual(
        list(actual.columns),
        list(input_schema.IMAGE_CSV_SCHEMA.get_input_keys()))
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_no_header_names_specified(self):
    """Tests valid CSV without a header, but header names are given."""
    f = _make_csv_tempfile(self.sample_data)
    actual = client.read_csv(f.name, header=None, names=self.header)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_with_header_no_names_specified(self):
    """Tests valid CSV with header, and no header names given (inferred)."""

    f = _make_csv_tempfile([self.header] + self.sample_data)
    actual = client.read_csv(f.name)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)

  def test_valid_csv_with_header_names_specified(self):
    """Tests valid CSV with header, and header names given (override)."""

    f = _make_csv_tempfile([self.header] + self.sample_data)
    actual = client.read_csv(f.name, names=self.header, header=0)
    self.assertEqual(list(actual.columns), self.header)
    self.assertEqual(actual.values.tolist(), self.sample_data)


class ToDataFrameTest(unittest.TestCase):
  """Tests `to_dataframe`."""

  def setUp(self) -> None:
    sample_data = get_sample_image_csv_data()
    columns = sample_data.pop(0)
    self.input_df = pd.DataFrame(sample_data, columns=columns)

  @mock.patch.object(client, 'read_csv', autospec=True)
  def test_input_csv(self, read_csv):
    """Tests valid input CSV file."""
    expected = self.input_df
    read_csv.return_value = expected
    f = _make_csv_tempfile(get_sample_image_csv_data())
    actual = client.to_dataframe(f.name)
    pd.testing.assert_frame_equal(actual, expected)

  def test_input_dataframe_no_names_specified(self):
    """Tests valid input dataframe with no header names specified."""
    actual = client.to_dataframe(self.input_df)
    pd.testing.assert_frame_equal(actual, self.input_df)

  def test_input_dataframe_with_header(self):
    """Tests valid input dataframe with header specified."""
    names = list(self.input_df.columns[0:-1])
    actual = client.to_dataframe(self.input_df, names=names)
    pd.testing.assert_frame_equal(actual, self.input_df[names])

  @mock.patch.object(client, '_read_image_directory')
  def test_input_image_dir(self, mock_fn):
    """Tests valid input image directory."""

    mock_fn.return_value = self.input_df

    with tempfile.TemporaryDirectory() as input_data:
      actual = client.to_dataframe(input_data)
      pd.testing.assert_frame_equal(actual, self.input_df)

  def test_error_invalid_inputs(self):
    """Tests error handling with different invalid inputs."""
    inputs = [0, 'not_a_csv_file', list(), dict()]
    for input_data in inputs:
      with self.assertRaises(ValueError):
        client.to_dataframe(input_data)


if __name__ == '__main__':
  unittest.main()
