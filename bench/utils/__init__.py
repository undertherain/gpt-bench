#!/usr/bin/env python
# coding=utf-8

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from packaging import version

from .. import __version__

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
#TF2_WEIGHTS_NAME = "tf_model.h5"
#TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
#TF_WEIGHTS_NAME = "model.ckpt"
#FLAX_WEIGHTS_NAME = "flax_model.msgpack"
#FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = (
                "This example requires a source install from HuggingFace Transformers (see "
                "`https://huggingface.co/docs/transformers/installation#install-from-source`),"
            )
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(
            error_message
            + "Check out https://huggingface.co/transformers/examples.html for the examples corresponding to other "
            "versions of HuggingFace Transformers."
        )
