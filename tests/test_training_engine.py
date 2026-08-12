from __future__ import annotations

from pathlib import Path

import torch

from homymoly.config import ArtifactConfig, RuntimeConfig
from homymoly.data.confirmatory import ConfirmatoryStructuredSignal
from homymoly.runtime import initialize_runtime
from homymoly.training.config import (
    Gate2Config,
    Gate2DataConfig,
    Gate2ExperimentConfig,
    Gate2GateConfig,
    Gate2ModelConfig,
    Gate2TrainingConfig,
)
from homymoly.training.engine import _loader, run_training


def _tiny_config(tmp_path: Path) -> Gate2Config:
    return Gate2Config(
        experiment=Gate2ExperimentConfig(name="test", seed=17),
        data=Gate2DataConfig(
            num_samples=60,
            min_vertices=24,
            max_vertices=24,
            seed=19,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            precision="float32",
            deterministic=True,
            allow_tf32=False,
            compile=False,
            num_workers=0,
        ),
        model=Gate2ModelConfig(
            hidden_dim=16,
            embedding_dim=8,
            num_layers=1,
            dropout=0.0,
            router_hidden_dim=8,
        ),
        training=Gate2TrainingConfig(
            batch_size=6,
            fixed_expert_epochs=1,
            translator_epochs=1,
            router_warmup_epochs=1,
            joint_finetune_epochs=1,
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            label_smoothing=0.0,
            early_stopping_patience=2,
            checkpoint_every=1,
            max_steps_per_epoch=1,
            pin_memory=False,
        ),
        gates=Gate2GateConfig(enforce=False),
        artifacts=ArtifactConfig(root=str(tmp_path), run_name="gate2-test"),
        source_path=tmp_path / "config.yaml",
        project_root=Path(__file__).parents[1],
    )


def test_gate2_training_checkpoint_and_completed_resume(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    report = run_training(config)
    assert report["status"] == "completed"
    assert report["epochs_completed"] == 4
    assert report["steps_completed"] == 4
    assert 0.0 <= report["test"]["hard_accuracy"] <= 1.0
    run_dir = Path(report["run_dir"])
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "summary.json").is_file()

    resumed = run_training(config, resume=True)
    assert resumed["epochs_completed"] == 4
    assert resumed["validation"]

    def _without_wall_clock(metrics: dict) -> dict:
        return {
            key: value
            for key, value in metrics.items()
            if not key.endswith("_milliseconds_per_example")
        }

    assert _without_wall_clock(resumed["test"]) == _without_wall_clock(report["test"])


def test_gate2_dry_run_does_not_create_run_directory(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    report = run_training(config, dry_run=True)
    assert report["status"] == "dry-run-passed"
    assert report["expert_logits_shape"][1:] == [3, 2]
    assert not config.run_dir.exists()


def test_worker_shuffle_generator_resumes_exact_next_epoch() -> None:
    dataset = ConfirmatoryStructuredSignal(60, seed=29, num_vertices=24)
    indices = tuple(range(36))
    runtime = initialize_runtime(
        RuntimeConfig(device="cpu", precision="float32", num_workers=2), seed=31
    )
    continuous_generator = torch.Generator().manual_seed(37)
    continuous_loader = _loader(
        dataset,
        indices,
        batch_size=6,
        shuffle=True,
        runtime=runtime,
        pin_memory=False,
        generator=continuous_generator,
        smoke=False,
    )
    list(continuous_loader)
    saved_state = continuous_generator.get_state()
    expected = [sample_id for batch in continuous_loader for sample_id in batch.sample_ids]

    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved_state)
    resumed_loader = _loader(
        dataset,
        indices,
        batch_size=6,
        shuffle=True,
        runtime=runtime,
        pin_memory=False,
        generator=resumed_generator,
        smoke=False,
    )
    actual = [sample_id for batch in resumed_loader for sample_id in batch.sample_ids]
    assert actual == expected
