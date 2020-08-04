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

"""Tests for beam_image."""

import base64
import unittest

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import PIL
from PIL import Image
import tensorflow_transform as tft

from tfrecorder import beam_image
from tfrecorder import constants
from tfrecorder import test_utils


class BeamImageTests(unittest.TestCase):
  """Tests for beam_image.py"""

  def setUp(self):
    self.pipeline = test_utils.get_test_pipeline()
    self.df = test_utils.get_test_df()
    self.image_file = 'tfrecorder/test_data/images/cat/cat-640x853-1.jpg'

  def test_load(self):
    """Tests the image loading function."""
    img = beam_image.load(self.image_file)
    self.assertIsInstance(img, PIL.JpegImagePlugin.JpegImageFile)

  def test_file_not_found_load(self):
    """Test loading an image that doesn't exist."""
    with self.assertRaises(OSError):
      _ = beam_image.load('tfrecorder/test_data/images/cat/food.jpg')

  def test_mode_to_channel(self):
    """Tests `mode_to_channel`."""

    actual = [beam_image.mode_to_channel(mode)
              for mode in ('L', 'RGB', 'whatever')]
    self.assertEqual(actual, [1, 3, 3])

  def test_channel_to_mode(self):
    """Tests `channel_to_mode`."""

    actual = [beam_image.channel_to_mode(channel) for channel in (1, 2, 3)]
    self.assertEqual(actual, ['L', 'RGB', 'RGB'])

  def test_base64_encode(self):
    """Tests encode function."""
    img = beam_image.load(self.image_file)
    enc = beam_image.encode(img)
    decode = base64.b64decode(enc, altchars=beam_image.BASE64_ALTCHARS)
    self.assertEqual(img.tobytes(), decode)

  def test_base64_decode(self):
    """Tests `decode` function."""

    image = Image.open(self.image_file)
    b64_bytes = base64.b64encode(
        image.tobytes(), altchars=beam_image.BASE64_ALTCHARS)
    width, height = image.size
    actual = beam_image.decode(b64_bytes, width, height, 3)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(image))

  def test_extract_image_dofn(self):
    """Tests ExtractImageDoFn."""

    with self.pipeline as p:

      converter = tft.coders.CsvCoder(constants.IMAGE_CSV_COLUMNS,
                                      constants.IMAGE_CSV_METADATA.schema)

      extract_images_fn = beam_image.ExtractImagesDoFn(constants.IMAGE_URI_KEY)

      data = (
          p
          | 'ReadFromDataFrame' >> beam.Create(self.df.values.tolist())
          | 'FlattenDataFrame' >> beam.Map(
              lambda x: ','.join([str(item) for item in x]))
          | 'DecodeCSV' >> beam.Map(converter.decode)
          | 'ExtractImage' >> beam.ParDo(extract_images_fn)
      )

      def key_matcher(expected_keys):
        """Custom Beam Test Matcher that tests a list of keys exist."""
        def _equal(actual):
          """ _equal raises a BeamAssertException when an element in the
              PCollection doesn't contain the image extraction keys."""
          expected_keys_ = set(expected_keys)
          for element in actual:
            actual_keys = set(element.keys())
            if actual_keys != expected_keys_:
              raise util.BeamAssertException(
                  'PCollection key match failed. Actual ({}) vs. expected ({})'
                  .format(actual_keys, expected_keys_))
        return _equal

      expected_keys = ['image_name', 'label', 'split', 'image',
                       'image_height', 'image_width', 'image_channels']
      util.assert_that(data, key_matcher(expected_keys))
