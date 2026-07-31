# This module:
# 1. Owns the single Modal application, volume, and image objects that
#    every launch file imports
#
# Design decisions:
# - Cloud infrastructure is defined exactly once; launch files import
#   these objects rather than constructing their own, so every job runs
#   on the identical application, storage, and image
#
# Author: Rahul Sawhney

from runtime import ModalVocodeImageBuilder

import modal

__all__: list[str] = ["app", "image", "volume"]

app: modal.App = modal.App("vocode-checkpoint01")

volume: modal.Volume = modal.Volume.from_name("vocode-data", create_if_missing=True)

image: modal.Image = ModalVocodeImageBuilder().build()
