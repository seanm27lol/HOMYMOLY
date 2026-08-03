# Gate-2 training and automatic GB10 launch

Gate 2 turns the Stage-1 contracts into a trainable experiment. It is intentionally staged: fixed experts are established before translator objectives, and the router is trained only after both foundations have received direct supervision.

## What is implemented

The executable stack contains:

1. a confirmatory `ConfirmatoryStructuredSignal` dataset with group-disjoint splits and reduced scalar shortcuts;
2. fixed graph, active-cell, and rank-2 connection-sheaf experts with a shared embedding contract;
3. graph-to-cell and graph-to-sheaf neural translators with explicitly named reconstruction and consistency surrogates;
4. a diagnostic- and compute-aware three-route router supporting soft and hard decisions;
5. directional and symmetric differentiable **H0 RTD-style surrogates** plus exact small-sample H0 reference diagnostics;
6. phased optimization, evaluation, atomic checkpoints, deterministic resume, JSONL metrics, and TensorBoard output.

The H0 surrogate is not the complete published RTD/SRTD cross-barcode construction. Reproducing the reference implementation remains a separate comparison gate and must be reported separately.

## Configuration and smoke test

The canonical GB10 configuration is [`configs/gate2.yaml`](../configs/gate2.yaml). Validate it and run the bounded smoke path with:

```bash
homymoly check-gate2-config --config configs/gate2.yaml
homymoly train --config configs/gate2.yaml --smoke
```

The smoke path uses a small deterministic subset, one epoch per phase, and a bounded number of batches. It verifies integration; it is not an experimental result.

The full resumable run is:

```bash
homymoly train --config configs/gate2.yaml --resume
```

The pinned-container equivalent is:

```bash
docker compose --profile training run --rm trainer
```

## Training phases

| Phase | Direct objective | Main validation signal |
|---|---|---|
| Fixed experts | route-conditional classification | oracle-route accuracy |
| Translators | reconstruction, consistency, and qualified H0 topology surrogates | finite held-out structural loss |
| Router | task loss, route supervision, expected compute, entropy, and retained structural terms | hard-route task and route accuracy |

Checkpoints are written atomically after every configured interval. `last.pt` contains the next phase/epoch cursor, optimizer and scheduler state, model state, and random-generator state. A resume request rejects a checkpoint produced by a materially different configuration.

## Automatic idle-GPU launch

The managed user-cron entry can be installed idempotently with:

```bash
python scripts/install_training_cron.py --interval-minutes 5 --max-utilization 10
```

Every check takes three utilization samples and also inspects NVIDIA compute processes. Training starts only if all samples are at or below the threshold and no compute process is present. The launcher uses a filesystem lock, validates any prior PID, and honors a completion marker, so overlapping cron invocations cannot start duplicate runs.

Scheduler state is stored under `artifacts/gate2/scheduler/`:

- `events.jsonl` records idle/busy/launch transitions;
- `trainer.json` records the launched process;
- `training.log` captures the detached training output;
- `training.complete` prevents relaunch after successful completion.

Remove only the HOMYMOLY-managed cron block with:

```bash
python scripts/install_training_cron.py --remove
```

This does not alter unrelated crontab entries.

## Outputs and claim boundary

Metrics, checkpoints, TensorBoard events, configuration snapshots, and the final evaluation summary live under `artifacts/gate2/` and are intentionally ignored by Git. The repository records code and protocols; machine-specific checkpoints remain local unless explicitly promoted later.

Success on the confirmatory synthetic benchmark would support continuing the investigation. It would not by itself establish superiority on molecular data, validate Langlands analogies, or show that a direct mapping-cone objective improves learning.
