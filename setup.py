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

"""Package setup."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "apache-beam[gcp] >= 2.22.0",
    "avro >= 1.10.0",
    "coverage >= 5.1",
    "ipython >= 7.15.0",
    "fire >= 0.3.1",
    "frozendict >= 1.2",
    "nose >= 1.3.7",
    "numpy < 1.19.0",
    "pandas >= 1.0.4",
    "Pillow >= 7.1.2",
    "pyarrow >= 0.17, < 0.18.0",
    "pylint >= 2.5.3",
    "pytz >= 2020.1",
    "python-dateutil",
    "tensorflow == 2.3.0",
    "tensorflow_transform >= 0.22",
]


setup(
    name='tfrecorder',
    version='0.1.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='TFRecorder creates TensorFlow Records easily.',
    entry_points={
        'console_scripts': ['tfrecorder=tfrecorder.cli:main'],
    },
)
