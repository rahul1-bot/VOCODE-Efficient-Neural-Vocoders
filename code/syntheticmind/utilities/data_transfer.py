# This module:
# 1. Moves every tensor inside an arbitrarily nested batch structure (dicts, lists,
#    tuples, and namedtuples) onto a target device
# 2. Falls back to the DeviceTransferable protocol for tensor-like containers that
#    manage their own device placement
# 3. Passes scalars and unrecognized objects through unchanged
#
# Design decisions:
# - Tensor copies use non_blocking=True so host-to-device transfers from pinned
#   memory can overlap with computation instead of serializing on each copy
# - Namedtuples are detected through their _fields attribute and rebuilt with _make
#   so the concrete type and field names survive the move; plain tuples and lists
#   are rebuilt as plain containers
# - The DeviceTransferable branch runs last, after every concrete-container branch,
#   and retries without non_blocking for .to implementations that reject the keyword
# - Anything the traversal does not recognize is returned as-is, so string labels,
#   None markers, and numeric metadata survive transfer untouched
#
# Author: Rahul Sawhney

import torch

from syntheticmind.utilities.types import Batch, DeviceTransferable

__all__: list[str] = ["move_data_to_device"]


def move_data_to_device(data: Batch, device: torch.device) -> Batch:
    # Public entry point for batch device placement; delegates to the recursive
    # traversal so callers never deal with structure-specific handling.
    return _recursive_move(data, device)


def _recursive_move(data: Batch, device: torch.device) -> Batch:
    # Walks the nested structure one branch at a time: tensors move with
    # non_blocking=True, dicts recurse value-by-value, tuples recurse
    # element-by-element with namedtuple reconstruction via _make, lists recurse
    # into new lists, and DeviceTransferable objects move themselves with a
    # TypeError retry for .to signatures without the non_blocking keyword.
    # Everything else falls through unchanged.
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _recursive_move(v, device) for k, v in data.items()}
    if isinstance(data, tuple):
        items: tuple[Batch, ...] = tuple(_recursive_move(item, device) for item in data)
        if hasattr(data, "_fields"):
            return type(data)._make(items)
        return items
    if isinstance(data, list):
        items: list[Batch] = [_recursive_move(item, device) for item in data]
        return items
    if isinstance(data, DeviceTransferable):
        try:
            return data.to(device, non_blocking=True)
        except TypeError:
            return data.to(device)
    return data
