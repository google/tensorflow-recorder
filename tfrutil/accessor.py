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

"""Creates a pandas DataFrame accessor for TFRUtil.

accessor.py contains TFRUtilAccessor which provides a pandas DataFrame
accessor.  This accessor allows us to inject the to_tfr() function into
pandas DataFrames.
"""
from typing import Union
import pandas as pd

from tfrutil import client


@pd.api.extensions.register_dataframe_accessor("tensorflow")
class TFRUtilAccessor:
  """DataFrame Accessor class for TFRUtil."""

  def __init__(self, pandas_obj):
    self._df = pandas_obj

  def to_tfr(self,
             runner: str,
             output_path: str,
             job_label: str = "to-tfr",
             compression: Union[str, None] = "gzip",
             num_shards: int = 0,
             image_col: str = "image",
             label_col: str = "label") -> str:
    """TFRUtil Pandas Accessor.

    TFRUtil provides an easy interface to create image-based tensorflow records
    from a dataframe containing GCS locations of the images and labels.

    Usage:
      import tfrutil

      df.tfrutil.to_tfr(runner="local",
                        output_path="gcs://foo/bar/train",
                        compression="gzip",
                        num_shards=10,
                        image_col="image",
                        label_col="label)

    Args:
      runner: Beam runner. Can be "local" or "DataFlow"
      output_path: Local directory or GCS Location to save TFRecords to.
      job_label: User supplied description for the beam job name.
      compression: Can be "gzip" or None for no compression.
      num_shards: Number of shards to divide the TFRecords into. Default is
          0 = no sharding.
      image_col: DataFrame column containing GCS path to image. Defaults to
        "image".
      label_col: DataFrame column containing image label. Defaults to "label".
    Returns:
      job_id: Job ID of the DataFlow job or PID of the local runner.
    """
    job_id = client.create_tfrecords(df=self._df,
                                     runner=runner,
                                     output_path=output_path,
                                     job_label=job_label,
                                     compression=compression,
                                     num_shards=num_shards,
                                     image_col=image_col,
                                     label_col=label_col)
    return job_id
