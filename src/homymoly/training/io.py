"""Atomic checkpoints and append-only metrics for resumable experiments."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported or malformed checkpoint: {path}")
    return payload


class MetricLogger(AbstractContextManager["MetricLogger"]):
    """Write durable JSONL metrics and TensorBoard scalars when available."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics" / "history.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_step = -1
        if self.metrics_path.is_file():
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    prior = json.loads(line)
                    self._last_step = max(self._last_step, int(prior["step"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        self._handle = self.metrics_path.open("a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self._writer = None
        else:
            self._writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def log(self, payload: Mapping[str, Any], *, step: int) -> None:
        if step < self._last_step:
            raise ValueError(
                f"metric step {step} precedes durable step {self._last_step}"
            )
        if step == self._last_step:
            return
        serializable: dict[str, Any] = {"step": int(step)}
        for key, value in payload.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError(f"metric tensor {key!r} must be scalar")
                value = value.detach().cpu().item()
            if isinstance(value, (int, float, str, bool)) or value is None:
                serializable[str(key)] = value
            else:
                raise TypeError(f"metric {key!r} is not JSON scalar compatible")
        self._handle.write(json.dumps(serializable, sort_keys=True) + "\n")
        self._handle.flush()
        self._last_step = step
        if self._writer is not None:
            for key, value in serializable.items():
                if key != "step" and isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if self._writer is not None:
            self._writer.close()

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["MetricLogger", "atomic_json", "atomic_torch_save", "load_checkpoint"]
