#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""
from __future__ import annotations
from .utils import conematrix, matrix_size_from_condensed
from .totalpersistence import totalpersistence, kercoker_via_cone
__version__ = "0.0.2"

__all__ = ['conematrix', 'matrix_size_from_condensed']
