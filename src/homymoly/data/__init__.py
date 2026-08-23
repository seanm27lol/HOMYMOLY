"""Synthetic structured datasets and their typed batch containers."""

from .boundary import cycles_to_boundary_lists, triangles_to_boundary_lists
from .collate import (
    collate_mixed_structured,
    collate_structured,
    make_structured_collate,
)
from .confirmatory import ConfirmatoryConfig, ConfirmatoryStructuredSignal
from .conversion import ConversionConfig, ConversionDataset, ConversionSample
from .mixed_structured import MixedStructuredSignal
from .types import (
    SignalRegime,
    StructuredBatch,
    StructuredData,
    StructuredObservations,
    StructuredSample,
)

__all__ = [
    "ConfirmatoryConfig",
    "ConfirmatoryStructuredSignal",
    "ConversionConfig",
    "ConversionDataset",
    "ConversionSample",
    "MixedStructuredSignal",
    "SignalRegime",
    "StructuredBatch",
    "StructuredData",
    "StructuredObservations",
    "StructuredSample",
    "collate_mixed_structured",
    "collate_structured",
    "cycles_to_boundary_lists",
    "make_structured_collate",
    "triangles_to_boundary_lists",
]
