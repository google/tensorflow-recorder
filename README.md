# TFRecorder

TFRecorder makes it easy to create TFRecords from images and labels in 
Pandas DataFrames or CSV files.
Today, TFRecorder supports data stored in 'image csv format' similar to 
GCP AutoML Vision. 
In the future TFRecorder will support converting any Pandas DataFrame or CSV 
file into TFRecords. 

!["TFRecorder CI/CD Badge"](https://github.com/google/tensorflow-recorder/workflows/TFRecord%20CICD/badge.svg)

## Installation

From the top directory of the repo, run the following command:

```bash
pip install .
```

## Usage

### IPython/Jupyter

#### Pandas DataFrame Conversion

```bash
import pandas as pd
import tfrecorder
df = pd.read_csv(...)
df.tensorflow.to_tfrecord(output_dir="gs://my/bucket")
```

#### Using Cloud Dataflow

```bash
df.tensorflow.to_tfrecord(
    output_dir="gs://my/bucket",
    runner="DataFlowRunner",
    project="my-project",
    region="us-central1)
```

### Command-line interface

```bash
tfrecorder create-tfrecords --output_dir="gs://my/bucket" data.csv
```

## Input format

TFRecorder currently expects data to be in the same format as 
[AutoML Vision](https://cloud.google.com/vision/automl/docs/prepare).  
This format looks like a pandas dataframe or CSV formatted as:

| split | image_uri                 | label |
|-------|---------------------------|-------|
| TRAIN | gs://my/bucket/image1.jpg | cat   |

Where:
* `split` can take on the values TRAIN, VALIDATION, and TEST
* `image_uri` specifies a local or google cloud storage location for the image file. 
* `label` can be either a text based label that will be integerized or integer

## Contributing

Pull requests are welcome. 
