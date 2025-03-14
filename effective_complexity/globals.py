# Copyright 2023 Janek Bevendorff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from effective_complexity.models import list_models
from effective_complexity.datasets import list_datasets


_MODEL_FRIENDLY_NAME_MAP = dict(
    mlp='MLP',
    gpt='GPT',
    cnn='CNN',
    resnet='RESNET',
    lstm='LSTM',
)
MODELS_ALL = {k: _MODEL_FRIENDLY_NAME_MAP.get(k, k)
                for k in list_models()}

_DATASETS_FRIENDLY_NAME_MAP = dict(
    synthetic='SYNTHETIC',
    cifar10='CIFAR10',
    cifar100='CIFAR100',
)
DATASETS_ALL = {k: _DATASETS_FRIENDLY_NAME_MAP.get(k, k)
                for k in list_datasets()}