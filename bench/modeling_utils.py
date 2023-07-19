# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import collections
# import json
import logging
import os
import re
import warnings
from contextlib import contextmanager
# from dataclasses import dataclass
# from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity

from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .pytorch_utils import (Conv1D, apply_chunking_to_forward,  # noqa: F401
                            find_pruneable_heads_and_indices,
                            id_tensor_storage, prune_conv1d_layer, prune_layer,
                            prune_linear_layer)
from .utils import ModelOutput  # , replace_return_docstrings
from .utils.import_utils import (ENV_VARS_TRUE_VALUES, importlib_metadata,
                                 is_sagemaker_mp_enabled)
# from .utils.quantization_config import BitsAndBytesConfig
from .utils.versions import require_version_core

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()

# if is_accelerate_available():
#     from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
#     from accelerate.utils import (
#         check_tied_parameters_on_same_device,
#         find_tied_parameters,
#         get_balanced_memory,
#         load_offloaded_weights,
#         offload_weight,
#         save_offload_index,
#         set_module_tensor_to_device,
#     )

logger = logging.getLogger(__name__)


_init_weights = True


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


# try:
#     from torch.nn import Identity
# except ImportError:
#     # Older PyTorch compatibility
#     class Identity(nn.Module):
#         r"""A placeholder identity operator that is argument-insensitive."""

#         def __init__(self, *args, **kwargs):
#             super().__init__()

#         def forward(self, input):
#             return input


def get_parameter_device(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_first_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    """
    Returns the first parameter dtype (can be non-floating) or asserts if none were found.
    """
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch > 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def get_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # Adding fix for https://github.com/pytorch/xla/issues/4152
            # Fixes issue where the model code passes a value that is out of range for XLA_USE_BF16=1
            # and XLA_DOWNCAST_BF16=1 so the conversion would cast it to -inf
            # NOTE: `is_torch_tpu_available()` is checked last as it induces a graph break in torch dynamo
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype

    # fallback to buffer dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype


def get_state_dict_float_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` or asserts if none were found.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    raise ValueError("couldn't find any floating point dtypes in state_dict")


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    else:
        return next(state_dict.values()).dtype


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = "WEIGHTS_NAME"
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_sahrd_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}

    for key, weight in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(weight, str):
            continue
        else:
            storage_id = id_tensor_storage(weight)

        # If a `weight` shares the same underlying storage as another tensor, we put `weight` in the same `block`
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if last_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


# def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
#     """
#     This is the same as
#     [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
#     but for a sharded checkpoint.

#     This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
#     loaded in the model.

#     Args:
#         model (`torch.nn.Module`): The model in which to load the checkpoint.
#         folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
#         strict (`bool`, *optional`, defaults to `True`):
#             Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
#         prefer_safe (`bool`, *optional*, defaults to `False`)
#             If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
#             safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

#     Returns:
#         `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
#             - `missing_keys` is a list of str containing the missing keys
#             - `unexpected_keys` is a list of str containing the unexpected keys
#     """
#     # Load the index
#     index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
#     safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

#     index_present = os.path.isfile(index_file)
#     safe_index_present = os.path.isfile(safe_index_file)

#     if not index_present and not (safe_index_present and is_safetensors_available()):
#         filenames = (
#             (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
#         )
#         raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

#     load_safe = False
#     if safe_index_present:
#         if prefer_safe:
#             if is_safetensors_available():
#                 load_safe = True  # load safe due to preference
#             else:
#                 logger.warning(
#                     f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
#                 )
#         elif not index_present:
#             load_safe = True  # load safe since we have no other choice

#     load_index = safe_index_file if load_safe else index_file

#     with open(load_index, "r", encoding="utf-8") as f:
#         index = json.load(f)

#     shard_files = list(set(index["weight_map"].values()))

#     # If strict=True, error before loading any of the state dicts.
#     loaded_keys = index["weight_map"].keys()
#     model_keys = model.state_dict().keys()
#     missing_keys = [key for key in model_keys if key not in loaded_keys]
#     unexpected_keys = [key for key in loaded_keys if key not in model_keys]
#     if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
#         error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
#         if len(missing_keys) > 0:
#             str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
#             error_message += f"\nMissing key(s): {str_missing_keys}."
#         if len(unexpected_keys) > 0:
#             str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
#             error_message += f"\nMissing key(s): {str_unexpected_keys}."
#         raise RuntimeError(error_message)

#     loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu")

#     for shard_file in shard_files:
#         state_dict = loader(os.path.join(folder, shard_file))
#         model.load_state_dict(state_dict, strict=False)

#         # Make sure memory is freed before we load the next state dict.
#         del state_dict
#         gc.collect()

#     # Return the same thing as PyTorch load_state_dict function.
#     return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


# def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
#     """
#     Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
#     """
#     if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
#         # Check format of the archive
#         with safe_open(checkpoint_file, framework="pt") as f:
#             metadata = f.metadata()
#         if metadata.get("format") not in ["pt", "tf", "flax"]:
#             raise OSError(
#                 f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
#                 "you save your model with the `save_pretrained` method."
#             )
#         elif metadata["format"] != "pt":
#             raise NotImplementedError(
#                 f"Conversion from a {metadata['format']} safetensors archive to PyTorch is not implemented yet."
#             )
#         return safe_load_file(checkpoint_file)
#     try:
#         return torch.load(checkpoint_file, map_location="cpu")
#     except Exception as e:
#         try:
#             with open(checkpoint_file) as f:
#                 if f.read(7) == "version":
#                     raise OSError(
#                         "You seem to have cloned a repository without having git-lfs installed. Please install "
#                         "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
#                         "you cloned."
#                     )
#                 else:
#                     raise ValueError(
#                         f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
#                         "model. Make sure you have saved the model properly."
#                     ) from e
#         except (UnicodeDecodeError, ValueError):
#             raise OSError(
#                 f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
#                 f"at '{checkpoint_file}'. "
#                 "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
#             )


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    for module_name, module in model.named_modules():
        loaded_keys = [k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")]
        if len(set(module.state_dict().keys()) - set(loaded_keys)) == 0:
            module._is_hf_initialized = True


def find_submodule_and_param_name(model, long_key, start_prefix):
    """
    A helper util to find the last sub-module and the param/buffer name. If `start_prefix` is supplied it'll be removed
    from the start of the key
    """

    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        long_key = ".".join(long_key.split(".")[1:])

    split_key = long_key.split(".")
    submodule = model
    while len(split_key) > 1:
        if hasattr(submodule, split_key[0]):
            submodule = getattr(submodule, split_key[0])
            del split_key[0]
        else:
            submodule = None
            break
    if submodule == model:
        submodule = None
    return submodule, split_key[0]


# def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
#     """
#     Moves `loaded_state_dict_keys` in model to meta device which frees up the memory taken by those params.

#     `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
#     `bert.pooler.dense.weight`

#     """

#     # dematerialize param storage for keys that are going to be replaced by state_dict, by
#     # putting those on the meta device
#     for k in loaded_state_dict_keys:
#         submodule, param_name = find_submodule_and_param_name(model, k, start_prefix)
#         if submodule is not None:
#             # selectively switch to the meta device only those params/buffers that will
#             # be next replaced from state_dict. This a complex way to do p.to_("meta")
#             # since we have no in-place to_ for tensors.
#             new_val = getattr(submodule, param_name)
#             if isinstance(new_val, torch.nn.Parameter):
#                 # isinstance returns False for Params on meta device, so switch after the check
#                 new_val = torch.nn.Parameter(new_val.to("meta"))
#             else:
#                 new_val = new_val.to("meta")
#             setattr(submodule, param_name, new_val)


# def _load_state_dict_into_meta_model(
#     model,
#     state_dict,
#     loaded_state_dict_keys,  # left for now but could be removed, see below
#     start_prefix,
#     expected_keys,
#     device_map=None,
#     offload_folder=None,
#     offload_index=None,
#     state_dict_folder=None,
#     state_dict_index=None,
#     dtype=None,
#     is_quantized=False,
#     is_safetensors=False,
#     keep_in_fp32_modules=None,
# ):
#     """
#     This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
#     params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
#     params back to the normal device, but only for `loaded_state_dict_keys`.

#     `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
#     `bert.pooler.dense.weight`

#     """

#     # XXX: remaining features to implement to be fully compatible with _load_state_dict_into_model
#     # - deepspeed zero 3 support
#     # - need to copy metadata if any - see _load_state_dict_into_model
#     # - handling error_msgs - mimicking the error handling in module._load_from_state_dict()
#     # - Is there a situation where some keys aren't in `loaded_state_dict_keys` and in which case
#     #   they won't get loaded.

#     if is_quantized:
#         from .utils.bitsandbytes import set_module_quantized_tensor_to_device

#     error_msgs = []

#     old_keys = []
#     new_keys = []
#     for key in state_dict.keys():
#         new_key = None
#         if "gamma" in key:
#             new_key = key.replace("gamma", "weight")
#         if "beta" in key:
#             new_key = key.replace("beta", "bias")
#         if new_key:
#             old_keys.append(key)
#             new_keys.append(new_key)
#     for old_key, new_key in zip(old_keys, new_keys):
#         state_dict[new_key] = state_dict.pop(old_key)

#     for param_name, param in state_dict.items():
#         # First part of the test is always true as load_state_dict_keys always contains state_dict keys.
#         if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
#             continue

#         if param_name.startswith(start_prefix):
#             param_name = param_name[len(start_prefix) :]

#         module_name = param_name
#         set_module_kwargs = {}

#         # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
#         # in int/uint/bool and not cast them.
#         if dtype is not None and torch.is_floating_point(param):
#             if (
#                 keep_in_fp32_modules is not None
#                 and any(module_to_keep_in_fp32 in param_name for module_to_keep_in_fp32 in keep_in_fp32_modules)
#                 and dtype == torch.float16
#             ):
#                 param = param.to(torch.float32)

#                 # For backward compatibility with older versions of `accelerate`
#                 # TODO: @sgugger replace this check with version check at the next `accelerate` release
#                 if "dtype" in list(inspect.signature(set_module_tensor_to_device).parameters):
#                     set_module_kwargs["dtype"] = torch.float32
#             else:
#                 param = param.to(dtype)

#         # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
#         if dtype is None:
#             old_param = model
#             splits = param_name.split(".")
#             for split in splits:
#                 old_param = getattr(old_param, split)
#                 if old_param is None:
#                     break

#             if old_param is not None:
#                 param = param.to(old_param.dtype)

#         set_module_kwargs["value"] = param

#         if device_map is None:
#             param_device = "cpu"
#         else:
#             # find next higher level module that is defined in device_map:
#             # bert.lm_head.weight -> bert.lm_head -> bert -> ''
#             while len(module_name) > 0 and module_name not in device_map:
#                 module_name = ".".join(module_name.split(".")[:-1])
#             if module_name == "" and "" not in device_map:
#                 # TODO: group all errors and raise at the end.
#                 raise ValueError(f"{param_name} doesn't have any device set.")
#             param_device = device_map[module_name]

#         if param_device == "disk":
#             if not is_safetensors:
#                 offload_index = offload_weight(param, param_name, offload_folder, offload_index)
#         elif param_device == "cpu" and state_dict_index is not None:
#             state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
#         elif not is_quantized:
#             # For backward compatibility with older versions of `accelerate`
#             set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
#         else:
#             if param.dtype == torch.int8 and param_name.replace("weight", "SCB") in state_dict.keys():
#                 fp16_statistics = state_dict[param_name.replace("weight", "SCB")]
#             else:
#                 fp16_statistics = None

#             if "SCB" not in param_name:
#                 set_module_quantized_tensor_to_device(
#                     model, param_name, param_device, value=param, fp16_statistics=fp16_statistics
#                 )

#     return error_msgs, offload_index, state_dict_index


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class PreTrainedModel(nn.Module, ModuleUtilsMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a PyTorch model.
        """
        return "pt"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        # self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kwargs)
        else:
            model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    # @classmethod
    # def can_generate(cls) -> bool:
    #     """
    #     Returns whether this model can generate sequences with `.generate()`.

    #     Returns:
    #         `bool`: Whether this model can generate sequences with `.generate()`.
    #     """
    #     # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
    #     if "GenerationMixin" in str(cls.prepare_inputs_for_generation):
    #         return False
    #     return True

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        pass

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str):
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
                " weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_modules.keys()}
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                        ) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                            " a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """
        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # XXX: put the long block of code in a wrapper
        if is_deepspeed_zero3_enabled():
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    # Copy old lm head weights to new lm head
                    if not transposed:
                        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
                            :num_tokens_to_copy, :
                        ]
                    else:
                        new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
                            :, :num_tokens_to_copy
                        ]

                    # Copy bias weights to new lm head
                    if has_new_lm_head_bias:
                        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        else:
            # Copy old lm head weights to new lm head
            if not transposed:
                new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
            else:
                new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

            # Copy bias weights to new lm head
            if has_new_lm_head_bias:
                new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

        return new_lm_head

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    def to(self, *args, **kwargs):
        # Checks if the model has been loaded in 8-bit
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`."
            )
        else:
            return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model has been loaded in 8-bit
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model has been loaded in 8-bit
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        is_quantized=False,
        keep_in_fp32_modules=None,
    ):
        is_safetensors = False
        if is_quantized:
            from .utils.bitsandbytes import set_module_quantized_tensor_to_device

        if device_map is not None and "disk" in device_map.values():
            archive_file = (
                resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
            )
            is_safetensors = archive_file.endswith(".safetensors")
            if offload_folder is None and not is_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if offload_folder is not None:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True

        is_sharded_safetensors = is_safetensors and sharded_metadata is not None
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        def _fix_key(key):
            if "beta" in key:
                return key.replace("beta", "bias")
            if "gamma" in key:
                return key.replace("gamma", "weight")
            return key

        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        if is_accelerate_available():
            model.tie_weights()
            tied_params = find_tied_parameters(model)
        else:
            tied_params = []

        for group in tied_params:
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # retrieve weights on meta device and put them back on CPU.
        # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key in list(model_state_dict.keys()):
                    key = key
                elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                    key = f"{prefix}.{key}"
                elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                    key = ".".join(key.split(".")[1:])
                param = model_state_dict[key]

                # upcast in fp32 if any
                target_dtype = dtype
                if (
                    keep_in_fp32_modules is not None
                    and dtype == torch.float16
                    and any(module_to_keep_in_fp32 in key for module_to_keep_in_fp32 in keep_in_fp32_modules)
                ):
                    target_dtype = torch.float32

                if param.device == torch.device("meta"):
                    if not (is_quantized):
                        set_module_tensor_to_device(model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype))
                    else:
                        set_module_quantized_tensor_to_device(
                            model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype)
                        )

        # retrieve unintialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if remove_prefix_from_model:
                _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
            elif add_prefix_to_model:
                _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
            else:
                _loaded_keys = loaded_keys
            set_initialized_submodules(model, _loaded_keys)
            # This will only initialize submodules that are not marked as initialized by the line above.
            model.apply(model._initialize_weights)

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    param = param.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if resolved_archive_file is not None:
            folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        else:
            folder = None
        if device_map is not None and is_safetensors:
            param_device_map = expand_device_map(device_map, original_loaded_keys)

            str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
            if sharded_metadata is None:
                archive_file = (
                    resolved_archive_file[0]
                    if isinstance(resolved_archive_file, (list, tuple))
                    else resolved_archive_file
                )
                weight_map = {p: archive_file for p in original_loaded_keys}
            else:
                weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
            offload_index = {
                p: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
                for p, f in weight_map.items()
                if param_device_map[p] == "disk"
            }

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
            offload_index = None
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            if not is_safetensors:
                offload_index = {} if device_map is not None and "disk" in device_map.values() else None
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None

            if is_sharded_safetensors:
                disk_only_shard_files = get_disk_only_shard_files(device_map, sharded_metadata=sharded_metadata)
                disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
            else:
                disk_only_shard_files = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
            for shard_file in resolved_archive_file:
                # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
                if shard_file in disk_only_shard_files:
                    continue
                state_dict = load_state_dict(shard_file)

                # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if low_cpu_mem_usage:
                    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_keys,
                        start_prefix,
                        expected_keys,
                        device_map=device_map,
                        offload_folder=offload_folder,
                        offload_index=offload_index,
                        state_dict_folder=state_dict_folder,
                        state_dict_index=state_dict_index,
                        dtype=dtype,
                        is_quantized=is_quantized,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)

                # force memory release
                del state_dict
                gc.collect()

            if offload_index is not None and len(offload_index) > 0:
                if model != model_to_load:
                    # We need to add the prefix of the base model
                    prefix = cls.base_model_prefix
                    if not is_safetensors:
                        for weight_name in offload_index:
                            shutil.move(
                                os.path.join(offload_folder, f"{weight_name}.dat"),
                                os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                            )
                    offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
                if not is_safetensors:
                    save_offload_index(offload_index, offload_folder)
                    offload_index = None

            if offload_state_dict:
                # Load back temporarily offloaded state dict
                load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if is_quantized:
            unexpected_keys = [elem for elem in unexpected_keys if "SCB" not in elem]
            missing_keys = [elem for elem in missing_keys if "SCB" not in elem]

        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warn if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @staticmethod
    def _load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file, start_prefix=""):
        """
        This is an experimental function that loads the model using ~1.x model size CPU memory

        Before you call it do:

        1. save which state_dict keys are available
        2. drop state_dict before model is created, since the latter takes 1x model size memory

        Here then we continue:

        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed.
        """

        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)
        state_dict = load_state_dict(resolved_archive_file)
        error_msgs = _load_state_dict_into_meta_model(model, state_dict, loaded_state_dict_keys, start_prefix)
        return error_msgs

    # @classmethod
    # def register_for_auto_class(cls, auto_class="AutoModel"):
    #     """
    #     Register this class with a given auto class. This should only be used for custom models as the ones in the
    #     library are already mapped with an auto class.

    #     <Tip warning={true}>

    #     This API is experimental and may have some slight breaking changes in the next releases.

    #     </Tip>

    #     Args:
    #         auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
    #             The auto class to register this new model with.
    #     """
    #     if not isinstance(auto_class, str):
    #         auto_class = auto_class.__name__

    #     import transformers.models.auto as auto_module

    #     if not hasattr(auto_module, auto_class):
    #         raise ValueError(f"{auto_class} is not a valid auto class.")

    #     cls._auto_class = auto_class

    def to_bettertransformer(self) -> "PreTrainedModel":
        """
        Converts the model to use [PyTorch's native attention
        implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to
        Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a
        subset of all Transformers models are supported.

        PyTorch's attention fastpath allows to speed up inference through kernel fusions and the use of [nested
        tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog
        post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).

        Returns:
            [`PreTrainedModel`]: The model converted to BetterTransformer.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.transform(self)

    def reverse_bettertransformer(self):
        """
        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is
        used, for example in order to save the model.

        Returns:
            [`PreTrainedModel`]: The model converted back to the original modeling.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.reverse(self)


#PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
# if PreTrainedModel.push_to_hub.__doc__ is not None:
#     PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
#         object="model", object_class="AutoModel", object_files="model file"
#     )


class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


# @dataclass
# class SquadHeadOutput(ModelOutput):
#     """
#     Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
#             Classification loss as the sum of start token, end token (and is_impossible if provided) classification
#             losses.
#         start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
#             Log probabilities for the top config.start_n_top start token possibilities (beam-search).
#         start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
#             Indices for the top config.start_n_top start token possibilities (beam-search).
#         end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
#             Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
#             (beam-search).
#         end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
#             Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
#         cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
#             Log probabilities for the `is_impossible` label of the answers.

#     """

#     loss: Optional[torch.FloatTensor] = None
#     start_top_log_probs: Optional[torch.FloatTensor] = None
#     start_top_index: Optional[torch.LongTensor] = None
#     end_top_log_probs: Optional[torch.FloatTensor] = None
#     end_top_index: Optional[torch.LongTensor] = None
#     cls_logits: Optional[torch.FloatTensor] = None


# class SQuADHead(nn.Module):
#     r"""
#     A SQuAD head inspired by XLNet.

#     Args:
#         config ([`PretrainedConfig`]):
#             The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
#             to use.
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.start_n_top = config.start_n_top
#         self.end_n_top = config.end_n_top

#         self.start_logits = PoolerStartLogits(config)
#         self.end_logits = PoolerEndLogits(config)
#         self.answer_class = PoolerAnswerClass(config)

#     # @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
#     def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         cls_index: Optional[torch.LongTensor] = None,
#         is_impossible: Optional[torch.LongTensor] = None,
#         p_mask: Optional[torch.FloatTensor] = None,
#         return_dict: bool = False,
#     ) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
#                 Final hidden states of the model on the sequence tokens.
#             start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#                 Positions of the first token for the labeled span.
#             end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#                 Positions of the last token for the labeled span.
#             cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#                 Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
#             is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#                 Whether the question has a possible answer in the paragraph or not.
#             p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
#                 Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
#                 should be masked.
#             return_dict (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

#         Returns:
#         """
#         start_logits = self.start_logits(hidden_states, p_mask=p_mask)

#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, let's remove the dimension added by batch splitting
#             for x in (start_positions, end_positions, cls_index, is_impossible):
#                 if x is not None and x.dim() > 1:
#                     x.squeeze_(-1)

#             # during training, compute the end logits based on the ground truth of the start position
#             end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

#             loss_fct = CrossEntropyLoss()
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#             if cls_index is not None and is_impossible is not None:
#                 # Predict answerability from the representation of CLS and START
#                 cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
#                 loss_fct_cls = nn.BCEWithLogitsLoss()
#                 cls_loss = loss_fct_cls(cls_logits, is_impossible)

#                 # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
#                 total_loss += cls_loss * 0.5

#             return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

#         else:
#             # during inference, compute the end logits based on beam search
#             bsz, slen, hsz = hidden_states.size()
#             start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

#             start_top_log_probs, start_top_index = torch.topk(
#                 start_log_probs, self.start_n_top, dim=-1
#             )  # shape (bsz, start_n_top)
#             start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
#             start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
#             start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

#             hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
#                 start_states
#             )  # shape (bsz, slen, start_n_top, hsz)
#             p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
#             end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
#             end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

#             end_top_log_probs, end_top_index = torch.topk(
#                 end_log_probs, self.end_n_top, dim=1
#             )  # shape (bsz, end_n_top, start_n_top)
#             end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
#             end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

#             start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
#             cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

#             if not return_dict:
#                 return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
#             else:
#                 return SquadHeadOutput(
#                     start_top_log_probs=start_top_log_probs,
#                     start_top_index=start_top_index,
#                     end_top_log_probs=end_top_log_probs,
#                     end_top_index=end_top_index,
#                     cls_logits=cls_logits,
#                 )


# class SequenceSummary(nn.Module):
#     r"""
#     Compute a single vector summary of a sequence hidden states.

#     Args:
#         config ([`PretrainedConfig`]):
#             The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
#             config class of your model for the default values it uses):

#             - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

#                 - `"last"` -- Take the last token hidden state (like XLNet)
#                 - `"first"` -- Take the first token hidden state (like Bert)
#                 - `"mean"` -- Take the mean of all tokens hidden states
#                 - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
#                 - `"attn"` -- Not implemented now, use multi-head attention

#             - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
#             - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
#               (otherwise to `config.hidden_size`).
#             - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
#               another string or `None` will add no activation.
#             - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
#             - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__()

#         self.summary_type = getattr(config, "summary_type", "last")
#         if self.summary_type == "attn":
#             # We should use a standard multi-head attention module with absolute positional embedding for that.
#             # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
#             # We can probably just use the multi-head attention module of PyTorch >=1.1.0
#             raise NotImplementedError

#         self.summary = Identity()
#         if hasattr(config, "summary_use_proj") and config.summary_use_proj:
#             if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
#                 num_classes = config.num_labels
#             else:
#                 num_classes = config.hidden_size
#             self.summary = nn.Linear(config.hidden_size, num_classes)

#         activation_string = getattr(config, "summary_activation", None)
#         self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

#         self.first_dropout = Identity()
#         if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
#             self.first_dropout = nn.Dropout(config.summary_first_dropout)

#         self.last_dropout = Identity()
#         if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
#             self.last_dropout = nn.Dropout(config.summary_last_dropout)

#     def forward(
#         self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
#     ) -> torch.FloatTensor:
#         """
#         Compute a single vector summary of a sequence hidden states.

#         Args:
#             hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
#                 The hidden states of the last layer.
#             cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
#                 Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

#         Returns:
#             `torch.FloatTensor`: The summary of the sequence hidden states.
#         """
#         if self.summary_type == "last":
#             output = hidden_states[:, -1]
#         elif self.summary_type == "first":
#             output = hidden_states[:, 0]
#         elif self.summary_type == "mean":
#             output = hidden_states.mean(dim=1)
#         elif self.summary_type == "cls_index":
#             if cls_index is None:
#                 cls_index = torch.full_like(
#                     hidden_states[..., :1, :],
#                     hidden_states.shape[-2] - 1,
#                     dtype=torch.long,
#                 )
#             else:
#                 cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
#                 cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
#             # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
#             output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
#         elif self.summary_type == "attn":
#             raise NotImplementedError

#         output = self.first_dropout(output)
#         output = self.summary(output)
#         output = self.activation(output)
#         output = self.last_dropout(output)

#         return output


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def expand_device_map(device_map, param_names):
    """
    Expand a device map to return the correspondance parameter name to device.
    """
    new_device_map = {}
    for module, device in device_map.items():
        new_device_map.update({p: device for p in param_names if p == module or p.startswith(f"{module}.")})
    return new_device_map


def get_disk_only_shard_files(device_map, sharded_metadata):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """
    files_content = collections.defaultdict(list)
    for weight_name, filename in sharded_metadata["weight_map"].items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]
