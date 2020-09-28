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

"""Creates a pandas DataFrame accessor for TFRecorder.

accessor.py contains TFRecorderAccessor which provides a pandas DataFrame
accessor.  This accessor allows us to inject the to_tfr() function into
pandas DataFrames.
"""

from typing import Any, Dict, Optional, Union
import pandas as pd
from IPython.core import display

from tfrecorder import client
from tfrecorder import constants
from tfrecorder import schema


@pd.api.extensions.register_dataframe_accessor('tensorflow')
class TFRecorderAccessor:
  """DataFrame Accessor class for TFRecorder."""

  def __init__(self, pandas_obj):
    self._df = pandas_obj

  # pylint: disable=too-many-arguments
  def to_tfr(
      self,
      output_dir: str,
      schema_map: Dict[str, schema.SchemaMap] = schema.image_csv_schema,
      runner: str = 'DirectRunner',
      project: Optional[str] = None,
      region: Optional[str] = None,
      tfrecorder_wheel: Optional[str] = None,
      dataflow_options: Union[Dict[str, Any], None] = None,
      job_label: str = 'to-tfr',
      compression: Optional[str] = 'gzip',
      num_shards: int = 0) -> Dict[str, Any]:
    """TFRecorder Pandas Accessor.

    TFRecorder provides an easy interface to create image-based tensorflow
    records from a dataframe containing GCS locations of the images and labels.

    Usage:
      import tfrecorder

      df.tfrecorder.to_tfr(
          output_dir='gcs://foo/bar/train',
          runner='DirectRunner',
          compression='gzip',
          num_shards=10)

    Args:
      schema_map: A dict mapping column names to supported types.
      output_dir: Local directory or GCS Location to save TFRecords to.
        Note: GCS required for DataflowRunner
      runner: Beam runner. Can be DirectRunner or  DataflowRunner.
      project: GCP project name (Required if DataflowRunner).
      region: GCP region name (Required if DataflowRunner).
      tfrecorder_wheel: Path to the tfrecorder wheel Dataflow will run.
        (create with 'python setup.py sdist' or
        'pip download tfrecorder --no-deps')
      dataflow_options: Optional dictionary containing Dataflow options.
      job_label: User supplied description for the beam job name.
      compression: Can be 'gzip' or None for no compression.
      num_shards: Number of shards to divide the TFRecords into. Default is
          0 = no sharding.
    Returns:
      job_results: A dictionary of job results.
    """
    display.display(
        display.HTML(
            '<b>Logging output to /tmp/{} </b>'.format(constants.LOGFILE)))

    r = client.create_tfrecords(
        self._df,
        output_dir=output_dir,
        schema_map=schema_map,
        runner=runner,
        project=project,
        region=region,
        tfrecorder_wheel=tfrecorder_wheel,
        dataflow_options=dataflow_options,
        job_label=job_label,
        compression=compression,
        num_shards=num_shards)
    return r
