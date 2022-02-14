# TFRecorder

TFRecorder makes it easy to create [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) from [Pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) or CSV Files. TFRecord reads data, transforms it using [TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started), stores it in the TFRecord format using [Apache Beam](https://beam.apache.org/) and optionally [Google Cloud Dataflow](https://cloud.google.com/dataflow). Most importantly, TFRecorder does this without requiring the user to write an Apache Beam pipeline or TensorFlow Transform code.

TFRecorder can convert any Pandas DataFrame or CSV file into TFRecords. If your data includes images TFRecorder can also serialize those into TFRecords. By default, TFRecorder expects your DataFrame or CSV file to be in the same ['Image CSV'](https://cloud.google.com/vision/automl/docs/prepare) format that Google Cloud Platform's AutoML Vision product uses, however you can also specify an input data schema using TFRecorder's flexible schema system.

!['TFRecorder CI/CD Badge'](https://github.com/google/tensorflow-recorder/workflows/TFRecord%20CICD/badge.svg)

[Release Notes](RELEASE.md)

## Why TFRecorder?
Using the TFRecord storage format is important for optimal machine learning pipelines and getting the most from your hardware (in cloud or on prem). The TFRecorder project started inside [Google Cloud AI Services](https://cloud.google.com/consulting) when we realized we were writing TFRecord conversion code over and over again.

When to use TFRecords:
* Your model is input bound (reading data is impacting training time).
* Anytime you want to use tf.Dataset
* When your dataset can't fit into memory


## Installation

### Install from Github

1. Clone this repo.

```bash
git clone https://github.com/google/tensorflow-recorder.git
```

For "bleeding edge" changes, check out the `dev` branch.

2. From the top directory of the repo, run the following command:

```bash
python setup.py install
```

#### Docker

Docker support is available through the `dev` branch

1. Run the following command to build the image. Specify the image name and tag:

```bash
docker build -t <image_name>:<tag> .
```

2. Run the following command to create and start the image. Include `bash` to use an interactive shell

```bash
docker run -it --rm -v $(pwd):/<working_directory> <image_name>:<tag> [bash]
```

### Install from PyPi
```bash
pip install tfrecorder
````

## Usage

### Generating TFRecords

You can generate TFRecords from a Pandas DataFrame, CSV file or
a directory containing images.

#### From Pandas DataFrame

TFRecorder has an accessor which enables creation of TFRecord files through
the Pandas DataFrame object.

Make sure the DataFrame contains a header identifying each of the columns.
In particular, the `split` column needs to be specified so that TFRecorder
would know how to split the data into train, test and validation sets.

##### Running on a local machine

```python
import pandas as pd
import tfrecorder

csv_file = '/path/to/images.csv'
df = pd.read_csv(csv_file, names=['split', 'image_uri', 'label'])
df.tensorflow.to_tfr(output_dir='/my/output/path')
```

##### Running on Cloud Dataflow

Google Cloud Platform Dataflow workers need to be supplied with the tfrecorder
package that you would like to run remotely.  To do so first download or build
the package (a python wheel file) and then specify the path the file when
tfrecorder is called.

Step 1: Download or create the wheel file.

To download the wheel from pip:
`pip download tfrecorder --no-deps`

To build from source/git:
`python setup.py sdist`

Step 2:
Specify the project, region, and path to the tfrecorder wheel for remote execution.

*Cloud Dataflow Requirements*
* The output_dir must be a Google Cloud Storage location.
* The image files specified in an image_uri column must be located in Google Cloud Storage.
* If being run from your local machine, the user must be [authenticated to use Google Cloud.](https://cloud.google.com/docs/authentication/getting-started)

```python
import pandas as pd
import tfrecorder

df = pd.read_csv(...)
df.tensorflow.to_tfr(
    output_dir='gs://my/bucket',
    runner='DataflowRunner',
    project='my-project',
    region='us-central1',
    tfrecorder_wheel='/path/to/my/tfrecorder.whl')
```

#### From CSV

Using Python interpreter:
```python
import tfrecorder

tfrecorder.convert(
    source='/path/to/data.csv',
    output_dir='gs://my/bucket')
```

Using the command line:
```bash
tfrecorder create-tfrecords \
    --input_data=/path/to/data.csv \
    --output_dir=gs://my/bucket
```

#### From an image directory

```python
import tfrecorder

tfrecorder.convert(
    source='/path/to/image_dir',
    output_dir='gs://my/bucket')
```

The image directory should have the following general structure:

```
image_dir/
  <dataset split>/
    <label>/
      <image file>
```

Example:
```
images/
  TRAIN/
    cat/
      cat001.jpg
    dog/
      dog001.jpg
  VALIDATION/
    cat/
      cat002.jpg
    dog/
      dog002.jpg
  ...
```

### Loading a TF Dataset from TFRecord files

You can load a TensorFlow dataset from TFRecord files generated by TFRecorder
on your local machine.

```python
import tfrecorder

dataset_dict = tfrecorder.load('/path/to/tfrecord_dir')
train = dataset_dict['TRAIN']
```

### Verifying data in TFRecords generated by TFRecorder

Using Python interpreter:
```python
import tfrecorder

tfrecorder.inspect(
    tfrecord_dir='/path/to/tfrecords/',
    split='TRAIN',
    num_records=5,
    output_dir='/tmp/output')
```

This will generate a CSV file containing structured data and image files
representing the images encoded into TFRecords.

Using the command line:

```bash
tfrecorder inspect \
    --tfrecord-dir=/path/to/tfrecords/ \
    --split='TRAIN' \
    --num_records=5 \
    --output_dir=/tmp/output
```

### Creating an image CSV file

You may want to create an image CSV file from an [image directory](#from-an-image-directory) 
for analysis, so that you can select image samples to convert to TFRecords using 
TFRecorder. Additionally, you can use the generated image CSV file to create 
datasets on [Cloud AutoML Vision](https://cloud.google.com/vision/overview/docs#automl-vision).

The image CSV file will adhere to the [Default Schema](#default-schema).

Uisng Python interpreter:

```python
from tfrecorder import utils

utils.create_image_csv('path/to/image/dir', 'out.csv')
```

Using the command line:

```bash
tfrecorder create-image-csv path/to/image/dir out.csv
```

## Default Schema

If you don't specify an input schema, TFRecorder expects data to be in the same format as
[AutoML Vision input](https://cloud.google.com/vision/automl/docs/prepare).
This format looks like a Pandas DataFrame or CSV formatted as:

| split | image_uri                 | label |
|-------|---------------------------|-------|
| TRAIN | gs://my/bucket/image1.jpg | cat   |

where:
* `split` can take on the values TRAIN, VALIDATION, and TEST
* `image_uri` specifies a local or Google Cloud Storage location for the image file.
* `label` can be either a text-based label that will be integerized or integer

## Flexible Schema

TFRecorder's flexible schema system allows you to use any schema you want for your input data.

For example, the default image CSV schema input can be defined like this:
```python
import pandas as pd
import tfrecorder
from tfrecorder import input_schema
from tfrecorder import types

image_csv_schema = input_schema.Schema({
    'split': types.SplitKey,
    'image_uri': types.ImageUri,
    'label': types.StringLabel
})

# You can then pass the schema to `tfrecorder.create_tfrecords`.

df = pd.read_csv(...)
df.tensorflow.to_tfr(
    output_dir='gs://my/bucket',
    schema_map=image_csv_schema,
    runner='DataflowRunner',
    project='my-project',
    region='us-central1')
```

### Flexible Schema Example

Imagine that you have a dataset that you would like to convert to TFRecords that
looks like this:

| split | x     |   y  | label |
|-------|-------|------|-------|
| TRAIN | 0.32  | 42   |1      |

You can use TFRecorder as shown below:

```python
import pandas as pd
import tfrecorder
from tfrecorder import input_schema
from tfrecorder import types

# First create a schema map
schema = input_schema.Schema({
    'split': types.SplitKey,
    'x': types.FloatInput,
    'y': types.IntegerInput,
    'label': types.IntegerLabel,
})

# Now call TFRecorder with the specified schema_map

df = pd.read_csv(...)
df.tensorflow.to_tfr(
    output_dir='gs://my/bucket',
    schema=schema,
    runner='DataflowRunner',
    project='my-project',
    region='us-central1')
```
After calling TFRecorder's `to_tfr()` function, TFRecorder will create an Apache beam pipeline, either locally or in this case
using Google Cloud's Dataflow runner. This beam pipeline will use the schema map to identify the types you've associated with
each data column and process your data using [TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) and TFRecorder's image processing functions to convert the data into into TFRecords.

### Supported types

TFRecorder's schema system supports several types.
You can use these types by referencing them in the schema map.
Each type informs TFRecorder how to treat your DataFrame columns.

#### types.SplitKey

* A split key is required for TFRecorder at this time.
* Only one split key is allowed.
* Specifies a split key that TFRecorder will use to partition the
input dataset on.
* Allowed values are 'TRAIN', 'VALIDATION, and 'TEST'

Note: If you do not want your data to be partitioned, include a column with
`types.SplitKey` and set all the elements to `TRAIN`.

#### types.ImageUri

* Specifies the path to an image. When specified, TFRecorder
will load the specified image and store the image as a [base64 encoded](https://docs.python.org/3/library/base64.html)
 [tf.string](https://www.tensorflow.org/tutorials/load_data/unicode) in the key 'image'
along with the height, width, and image channels  as integers using the keys 'image_height', 'image_width', and 'image_channels'.
* A schema can contain only one imageUri column

#### types.IntegerInput

* Specifies an int input.
* Will be scaled to mean 0, variance 1.

#### types.FloatInput

* Specifies an float input.
* Will be scaled to mean 0, variance 1.

#### types.CategoricalInput

* Specifies a string input.
* Vocabulary computed and output integerized.

#### types.IntegerLabel

* Specifies an integer target.
* Not transformed.

#### types.StringLabel

* Specifies a string target.
* Vocabulary computed and *output integerized.*

## Contributing

Pull requests are welcome.
Please see our [code of conduct](docs/code-of-conduct.md) and [contributing guide](docs/contributing.md).

## Why TFRecorder?

Using the TFRecord storage format is important for optimal machine learning pipelines and getting the most from your hardware (in cloud or on prem).

TFRecords help when:
* Your model is input bound (reading data is impacting training time).
* Anytime you want to use tf.Dataset
* When your dataset can't fit into memory

Need help with using AI in the cloud?
Visit [Google Cloud AI Services](https://cloud.google.com/consulting).
