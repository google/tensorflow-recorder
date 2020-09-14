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

"""Common utility functions."""

from datetime import datetime
import os

import tensorflow as tf

from tfrecorder import constants


def get_timestamp() -> str:
  """Returns current date and time as formatted string."""
  return datetime.now().strftime('%Y%m%d-%H%M%S')


def copy_logfile_to_gcs(logfile: str, output_dir: str):
  """Copies a logfile from local to gcs storage."""
  try:
    with open(logfile, 'r') as log_reader:
      out_log = os.path.join(output_dir, constants.LOGFILE)
      with tf.io.gfile.GFile(out_log, 'w') as gcs_logfile:
        log = log_reader.read()
        gcs_logfile.write(log)
  except FileNotFoundError as e:
    raise FileNotFoundError("Unable to copy log file {} to gcs.".format(
        e.filename)) from e
