# TFRecord Conversion Utilities

## Installing

1. Clone this repo.

2. Run the command `python3 setup.py`

## What is TFRUtil
TFRUtil makes it easy to create TFRecords from images and labels using Pandas DataFrames or CSVs. 
Today, TFRUtil supports data stored in 'image csv format' similar to GCP AutoML Vision. In the
future TFRUtil will support converting any dataframe or CSV file into TFRecords. 

## Using TFRUtil to create TFRecords

### Image CSV Format
TFRUtil currently expects data to be in the same format as [AutoML Vision](https://cloud.google.com/vision/automl/docs/prepare).  This format looks like a pandas dataframe or CSV formatted as:

| split | image_uri               | label |
|-------|-------------------------|-------|
| TRAIN | gs://foo/bar/image1.jpg | cat   |

Where:
* split can take on the values TRAIN, VALIDATION, and TEST
* image_uri specifies a local or google cloud storage location for the image file. 
* label can be either a text based label that will be integerized or integer

### Pandas API
TODO

### Python API
TODO

### CSV File
TODO

## Using TFRutil to inspect TFRecords
TODO
