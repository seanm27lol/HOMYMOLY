from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from homymoly.config import RuntimeConfig
from homymoly.runtime import initialize_runtime, maybe_compile


def test_runtime_initializer_applies_policy_and_reproducible_seeds() -> None:
    config = RuntimeConfig(
        device="cpu",
        precision="float32",
        deterministic=True,
        allow_tf32=False,
        compile=False,
        num_workers=2,
    )
    first = initialize_runtime(config, seed=91)
    values = (random.random(), float(np.random.rand()), float(torch.rand(())))
    second = initialize_runtime(config, seed=91)
    repeated = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert values == repeated
    assert first == second
    assert first.device == torch.device("cpu")
    assert first.neural_dtype == torch.float32
    assert first.num_workers == 2
    assert torch.are_deterministic_algorithms_enabled()
    module = nn.Linear(2, 2)
    assert maybe_compile(module, first) is module


def test_runtime_rejects_unavailable_explicit_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        initialize_runtime(RuntimeConfig(device="cuda"), seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_runtime_resolves_explicit_cuda_and_bfloat16() -> None:
    state = initialize_runtime(
        RuntimeConfig(device="cuda", precision="bfloat16"),
        seed=7,
    )

    assert state.device.type == "cuda"
    assert state.neural_dtype == torch.bfloat16
    assert torch.cuda.get_device_name(state.device) == "NVIDIA GB10"
