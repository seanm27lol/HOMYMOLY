# Stage-1 runtime build

This scaffold provides a reproducible HOMYMOLY configuration, an NVIDIA GB10 container path, artifact directories, a bounded hardware profile, and a local-only experiment dashboard. It intentionally does not define model or optimizer schemas.

## Runtime contract

The container is based on `nvcr.io/nvidia/pytorch:26.06-py3`. Runtime dependencies are bounded to compatible NumPy, NetworkX, PyYAML, and PyTorch releases; all have ARM64 support and none requires an x86-only graph-library wheel. PyTorch itself comes preinstalled in NVIDIA's image. The project requirement accepts that NGC build and must not be used to replace it with a PyPI CUDA stack. Development and dashboard packages are optional.

On GB10, use the NGC container rather than asking a fresh virtual environment to resolve a PyPI CUDA stack. The Docker build installs only the pure-Python runtime/dashboard dependencies first, verifies the image's preinstalled NumPy and PyTorch, and then installs HOMYMOLY with `--no-deps`. Freeze the final container digest and a complete environment lock before confirmatory experiments.

The active configuration defaults to `configs/stage1.yaml`. Validate it from the repository root:

```bash
python -m pip install -e '.[dev,dashboard]'
homymoly check-config --config configs/stage1.yaml
ruff check src tests scripts
python -m pytest
```

Training code must call `initialize_runtime(config.runtime, seed=config.experiment.seed)` before constructing models or loaders. It resolves the device and neural dtype, seeds Python/NumPy/PyTorch, applies deterministic-algorithm and TF32 policy, and supplies the configured worker count. `maybe_compile` applies compilation only after eager-mode correctness is established.

To create the declared directories:

```bash
homymoly check-config --config configs/stage1.yaml --create-artifacts
```

Run the mathematical/data integration gate:

```bash
homymoly validate-foundation \
  --config configs/stage1.yaml \
  --samples 6 \
  --vertices 24
```

The JSON report records balance counts, structural sizes, maximum boundary,
chain-map, cone-chain, and connection-operator residuals, plus fixed triangle
Betti sentinels. The sheaf convention maps each canonical edge tail into the
head frame and evaluates `x_head - T x_tail`; unit tests include pure-gauge and
nontrivial-holonomy sentinels. The gate uses float64 structural oracles
independently of the neural precision in the runtime configuration.

The synthetic generator is intentionally an easy bring-up tier. Measure and
record its cheap scalar shortcuts before using it:

```bash
python scripts/audit_shortcuts.py \
  --config configs/stage1.yaml \
  --samples 600 \
  --vertices 24
```

High amplitude-route or scalar-label accuracy is expected here and explicitly
disqualifies this tier from supporting routing-novelty claims. The command is a
required baseline and a design diagnostic for the confirmatory generator.

Relative artifact roots resolve from the repository containing `pyproject.toml`, not from the caller's current directory. The default layout is:

```text
artifacts/
└── stage1/
    ├── checkpoints/
    ├── metrics/
    ├── profiles/
    └── tensorboard/
```

Set a temporary root without editing YAML by passing `--artifact-root`, for example:

```bash
homymoly paths --config configs/stage1.yaml --artifact-root /tmp/homymoly-artifacts
```

## NVIDIA container

The host needs Docker, the NVIDIA Container Toolkit, and access to NGC. Build and validate the image with:

```bash
mkdir -p artifacts
docker compose build homymoly
docker compose run --rm homymoly
```

Source, configuration, and scripts are mounted read-only, while only `artifacts/` is writable. `PYTHONPATH` points to `/workspace/src`, so local Stage-1 code is used without rebuilding the image. Compose requests all visible NVIDIA GPUs and uses a private 16 GB shared-memory allocation. It defaults to UID/GID 1000; set `HOMYMOLY_UID` and `HOMYMOLY_GID` to the host user's numeric values when they differ so artifacts remain user-owned.

## GB10 profile

Run the bounded BF16 profile inside the NVIDIA container:

```bash
docker compose run --rm \
  --entrypoint python \
  homymoly \
  /workspace/scripts/profile_gb10.py \
  --config /workspace/configs/stage1.yaml \
  --require-gb10
```

The script records host architecture, Python and PyTorch versions, CUDA capability, device memory, BF16 support, peak allocation, and median matrix-multiplication throughput. It skips any configured matrix size whose estimated working set exceeds half the currently free CUDA memory. JSON is written atomically under `artifacts/stage1/profiles/`.

For a metadata-only check on a development machine:

```bash
python scripts/profile_gb10.py --config configs/stage1.yaml --skip-benchmark
```

Throughput is a smoke measurement, not a vendor benchmark. Record container digest, power mode, clocks, thermal state, and concurrent workload before using it in an experimental report.

## Local TensorBoard

Start TensorBoard directly:

```bash
./scripts/start_dashboard.sh
```

It binds to `127.0.0.1:6006` by default. The Compose dashboard also publishes only to host loopback:

```bash
docker compose --profile dashboard up tensorboard
```

Open `http://127.0.0.1:6006`. Override the port with `HOMYMOLY_TENSORBOARD_PORT`; do not change the host binding unless the surrounding network is trusted.

## Tailnet-only dashboard access

When Tailscale is already authenticated on the GB10 host, keep TensorBoard on loopback and proxy it with Serve:

```bash
tailscale serve status
tailscale serve --bg http://127.0.0.1:6006
tailscale serve status
```

Inspect the existing status before changing it, and use the HTTPS tailnet URL printed by `tailscale serve`. Tailnet ACLs still govern who can reach that URL. `tailscale serve reset` removes every Serve handler on the node, so use it only when HOMYMOLY owns the node's complete Serve configuration:

```bash
tailscale serve reset
```

Tailscale Serve limits access to the tailnet. Do not substitute Tailscale Funnel, which would make the dashboard publicly reachable. TensorBoard should contain metrics and profiling traces only; never log secrets or raw sensitive datasets.

Before relying on the dashboard, require a clean Tailscale health check and verify the printed HTTPS URL from the intended client device. Direct tailnet IP connectivity can work even when MagicDNS or certificate-related DNS is unhealthy.

## Acceptance checks

Stage-1 runtime scaffolding is accepted when:

- the YAML loads into dataclasses and rejects unknown or invalid fields;
- relative artifact paths resolve identically from different working directories;
- the configuration CLI can create and print the artifact tree;
- the foundation gate reports zero boundary, chain-map, and cone-chain residuals;
- the identity-cone and filled-triangle Betti sentinels match their exact values;
- the connection coboundary matches edge residuals and detects holonomy defects;
- the metadata-only profiler completes without CUDA;
- the container profiler identifies the GB10 and records BF16/CUDA metadata; and
- TensorBoard is reachable on loopback and, when requested, through Tailscale Serve only.
