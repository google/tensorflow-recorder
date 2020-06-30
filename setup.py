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

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "apache-beam[gcp] >= 2.22.0",
    "pandas >= 1.0.4",
    "tensorflow_transform >= 0.22",
    "Pillow >= 7.1.2",
    "coverage >= 5.1",
    "ipython >= 7.15.0",
    "nose >= 1.3.7",
    "pylint >= 2.5.3",
    "fire >= 0.3.1",
    "tensorflow >= 2.2.0",
    "gcsfs >= 0.6.2"
]


setup(
    name='tfrutil',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='TFRUtil creates TensorFlow Records easily.'
)
