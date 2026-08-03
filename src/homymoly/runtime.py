"""Apply the reproducibility and device policy declared by Stage-1 config."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
from torch import nn

from .config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Resolved runtime values for model and DataLoader construction."""

    device: torch.device
    neural_dtype: torch.dtype
    seed: int
    deterministic: bool
    allow_tf32: bool
    compile: bool
    num_workers: int


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def initialize_runtime(config: RuntimeConfig, *, seed: int) -> RuntimeState:
    """Seed libraries and apply deterministic, TF32, and device settings."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if config.device == "cuda"
        or (config.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.precision]

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not config.deterministic
        torch.backends.cudnn.deterministic = config.deterministic
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")

    return RuntimeState(
        device=device,
        neural_dtype=dtype,
        seed=seed,
        deterministic=config.deterministic,
        allow_tf32=config.allow_tf32,
        compile=config.compile,
        num_workers=config.num_workers,
    )


def maybe_compile(module: ModuleT, state: RuntimeState) -> ModuleT:
    """Compile a module only when the resolved runtime contract requests it."""

    if not state.compile:
        return module
    return torch.compile(module)  # type: ignore[return-value]


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


__all__ = ["RuntimeState", "initialize_runtime", "maybe_compile", "seed_worker"]
