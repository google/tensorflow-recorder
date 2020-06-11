# python3

# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================
"""Creates a pandas DataFrame accessor for TFRUtil.

accessor.py contains TFRUtilAccessor which provides a pandas DataFrame
accessor.  This accessor allows us to inject the to_tfr() function into
pandas DataFrames.
"""
from typing import Union
import pandas as pd

from tfrutil import client


@pd.api.extensions.register_dataframe_accessor("tfrutil")
class TFRUtilAccessor:
  """DataFrame Accessor class for TFRUtil."""

  def __init__(self, pandas_obj):
    self._df = pandas_obj

  def to_tfr(self,
             runner: str,
             output_path: str,
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
                                     compression=compression,
                                     num_shards=num_shards,
                                     image_col=image_col,
                                     label_col=label_col)
    return job_id
