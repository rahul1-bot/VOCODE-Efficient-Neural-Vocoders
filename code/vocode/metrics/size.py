# This module:
# 1. Measures model storage at two definitions: the in-memory bytes of
#    registered parameters and buffers, and the exact serialized bytes of
#    the persisted state dictionary
#
# Design decisions:
# - Both definitions are reported because they answer different questions:
#   the in-memory value preserves continuity with historical rows, while
#   the serialized value is what a deployable artifact actually occupies
# - The serialized measurement is authoritative for packed quantized
#   weights, whose live objects hold dequantization scaffolding that never
#   ships
#
# Author: Rahul Sawhney

import io

import torch
from torch import nn

__all__: list[str] = ["ModelSize"]


class ModelSize:
    # Two-definition storage measurement over a torch.nn.Module network.
    #
    # The two definitions answer different questions and are both reported
    # rather than reconciled into one number. The in-memory value sums the
    # element bytes of everything the live object holds, which preserves
    # continuity with the historical rows of the study. The serialized
    # value is the exact byte length of the persisted state dictionary,
    # which is what a shipped artifact actually occupies.
    #
    # The two diverge wherever live state does not ship. A non-persistent
    # buffer occupies memory but never enters the state dictionary, and a
    # packed quantized weight carries dequantization scaffolding in the
    # live object that never reaches the artifact; the serialized value is
    # authoritative in both cases. Serialization also carries format
    # overhead of its own, so even a network holding no weights reports a
    # non-zero serialized size. Both measurements are pure reads that leave
    # the network untouched and repeat identically.
    def __call__(self, network: nn.Module) -> float:
        # Sums the element sizes of every registered parameter and buffer
        # and reports megabytes; this is the in-memory definition.
        #
        # Element size is read per tensor rather than assumed, so a
        # half-precision network reports half the size of the same geometry
        # in single precision. The reported unit is the binary megabyte of
        # 1024 squared bytes.
        parameter_bytes: int = sum(
            parameter.numel() * parameter.element_size() for parameter in network.parameters()
        )
        buffer_bytes: int = sum(
            buffer.numel() * buffer.element_size() for buffer in network.buffers()
        )
        total_bytes: int = parameter_bytes + buffer_bytes
        return total_bytes / (1024.0 * 1024.0)

    def serialized_megabytes(self, network: nn.Module) -> float:
        # Serializes the state dictionary into memory and reports its exact
        # byte size in megabytes; this is the deployable definition.
        #
        # Serialization targets an in-memory buffer rather than a file, so
        # measuring a network writes nothing to disk. The reported unit is
        # the binary megabyte of 1024 squared bytes, matching the
        # in-memory definition so the two are directly comparable.
        serialized_buffer: io.BytesIO = io.BytesIO()
        torch.save(network.state_dict(), serialized_buffer)
        return serialized_buffer.getbuffer().nbytes / (1024.0 * 1024.0)
