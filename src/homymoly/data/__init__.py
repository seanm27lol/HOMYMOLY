"""Synthetic structured datasets and their typed batch containers."""

from .collate import (
    collate_mixed_structured,
    collate_structured,
    make_structured_collate,
)
from .confirmatory import ConfirmatoryConfig, ConfirmatoryStructuredSignal
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
    "MixedStructuredSignal",
    "SignalRegime",
    "StructuredBatch",
    "StructuredData",
    "StructuredObservations",
    "StructuredSample",
    "collate_mixed_structured",
    "collate_structured",
    "make_structured_collate",
]
