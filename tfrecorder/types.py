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

"""Custom types."""

from typing import Tuple

from apache_beam.pvalue import PCollection
from tensorflow_transform import beam as tft_beam


BeamDatasetMetadata = tft_beam.tft_beam_io.beam_metadata_io.BeamDatasetMetadata
TransformedMetadata = BeamDatasetMetadata
TransformFn = Tuple[PCollection, TransformedMetadata]
