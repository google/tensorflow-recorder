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
""" Functions and beam DoFn for reading and encoding images."""

import base64
import logging
import os
from typing import Any, Dict, Generator, Tuple

import apache_beam as beam
from apache_beam.metrics import Metrics
import tensorflow as tf
from PIL import Image


BASE64_ALTCHARS = b'-_'


def mode_to_channel(mode: str) -> int:
  """Returns number of channels depending on PIL image `mode`s."""

  return 1 if 'L' in mode else 3


def channel_to_mode(channels: int) -> str:
  """Returns PIL image mode depending on number of `channels`."""

  return 'L' if channels == 1 else 'RGB'


def encode(image: Image):
  """Returns base64-encoded image data.

  Args:
    image: PIL image.
  """

  return base64.b64encode(image.tobytes(), altchars=BASE64_ALTCHARS)


def decode(b64_bytes, width, height, channels) -> Image:
  """Decodes an image from base64-encoded data."""

  image_bytes = base64.b64decode(b64_bytes, altchars=BASE64_ALTCHARS)
  mode = channel_to_mode(channels)
  return Image.frombytes(mode, (width, height), image_bytes)


def load(image_uri):
  """Loads an image."""

  try:
    with tf.io.gfile.GFile(image_uri, 'rb') as f:
      return Image.open(f)
  except tf.python.framework.errors_impl.NotFoundError as e:
    raise OSError('File {} was not found.'.format(image_uri)) from e


# pylint: disable=abstract-method

class ExtractImagesDoFn(beam.DoFn):
  """Adds image to PCollection."""

  def __init__(self, image_uri_key: str):
    """Constructor."""
    super().__init__()
    self.image_uri_key = image_uri_key
    self.image_good_counter = Metrics.counter(self.__class__, 'image_good')
    self.image_bad_counter = Metrics.counter(self.__class__, 'image_bad')


  # pylint: disable=unused-argument
  def process(
      self,
      element: Dict[str, Any],
      *args: Tuple[Any, ...],
      **kwargs: Dict) -> Generator[Dict[str, Any], None, None]:
    """Loads image and creates image features.

    This DoFn extracts an image being stored on local disk or GCS and
    yields a base64 encoded image, the image height, image width, and channels.
    """
    d = {}

    try:
      image_uri = element.pop(self.image_uri_key)
      image = load(image_uri)
      element['image_name'] = os.path.split(image_uri)[-1]
      d['image'] = encode(image)
      d['image_width'], d['image_height'] = image.size
      d['image_channels'] = mode_to_channel(image.mode)
      self.image_good_counter.inc()

    # pylint: disable=broad-except
    except Exception as e:
      logging.warning('Could not load image: %s', image_uri)
      logging.error('Exception was: %s', str(e))
      self.image_bad_counter.inc()
      d['split'] = 'DISCARD'

    element.update(d)
    yield element
