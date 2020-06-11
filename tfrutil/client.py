# Lint as: python3
"""Provides a common interface for TFRUtil to DF Accessor and CLI.

client.py provides create_tfrecords() to upstream clients including
the Pandas DataFrame Accessor (accessor.py) and the CLI (TODO(cezequeil: name file)).
"""
from typing import Union
import pandas as pd


def _validate_data(df, image_col, label_col):
  # verify there is a column latitude and a column longitude
  if image_col not in df.columns:
  # or label_col not in df.columns:
    raise AttributeError(
        "Dataframe must contain specified image column {}.".format(image_col))
  if label_col not in df.columns:
    raise AttributeError(
        "Dataframe must contain specified label column {}.".format(label_col))


def create_tfrecords(df: pd.DataFrame,
                     runner: str,
                     output_path: str,
                     compression: Union[str, None] = "gzip",
                     num_shards: int = 0,
                     image_col: str = "image",
                     label_col: str = "label") -> str:
  """TFRUtil Python Client.

  TFRUtil provides an easy interface to create image-based tensorflow records
  from a dataframe containing GCS locations of the images and labels.

  Usage:
    import tfrutil

    job_id = tfrutil.create_tfrecords(train_df,
                                     runner="local",
                                     output_location="gcs://foo/bar/train",
                                     compression="gzip",
                                     num_shards=10,
                                     image_col="image",
                                     label_col="label)

  Args:
    df: Pandas DataFrame
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
  _validate_data(df, image_col, label_col)
  job_id = _do_magic(df)
  return job_id


def _do_magic(df: pd.DataFrame):
  pid = "p1234"
  return pid

