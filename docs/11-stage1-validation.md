# Stage-1 validation record

This record captures the first execution of the HOMYMOLY foundation on the local GB10 host. It is a smoke and invariant result, not evidence for the proposed learning claims.

## Host profile

| Item | Observed value |
|---|---|
| Architecture | ARM64 (`aarch64`) |
| Accelerator | NVIDIA GB10 |
| Compute capability | 12.1 |
| Driver | 580.173.02 |
| PyTorch smoke environment | 2.13.0+cu130 |
| CUDA build | 13.0 |
| Container base | `nvcr.io/nvidia/pytorch:26.06-py3` |
| Container PyTorch | 2.13.0a0+8145d630e8.nv26.06 |
| Container CUDA build | 13.3 |
| BF16 | supported |
| PyTorch-reported unified memory | 130,662,936,576 bytes (121.69 GiB) |

The host-side bounded BF16 GEMM smoke measured 39.679 TFLOP/s at size 2048 and 57.673 TFLOP/s at size 4096. These numbers are synchronization-aware implementation checks, not vendor benchmarks; clocks, power, thermals, and concurrent workload were not controlled.

The rebuilt NGC container passed its packaged configuration and foundation gates on the GB10. Its bounded size-2048 BF16 GEMM smoke measured 92.731 TFLOP/s under the same caveats. This figure is an environment check, not a comparable product-performance result.

The raw local profile is stored under the ignored `artifacts/stage1/profiles/` directory so machine telemetry does not become a repository dependency.

## Mathematical and data gate

The six-sample exact-oracle run passed with:

- maximum `B1 @ B2` residual: `0.0`;
- maximum graph-to-cell chain-map residual: `0.0`;
- maximum mapping-cone chain residual: `0.0`;
- maximum connection-sheaf operator mismatch: `2.220446049250313e-16`;
- maximum transport orthogonality residual: `6.747796987838228e-08`;
- graph triangle Betti numbers: `[1, 1]`;
- filled triangle Betti numbers: `[1, 0, 0]`;
- identity-map cone Betti numbers: `[0, 0, 0]`;
- graph-to-filled-triangle inclusion cone Betti numbers: `[0, 0, 1]`.

The configured 6,144 samples resolve into one group-disjoint dataset with 4,302 training, 924 validation, and 918 test examples. All split sizes are multiples of the six-example counterfactual group.

## Shortcut audit

On 600 bring-up examples with 24 vertices:

- three amplitude summaries identified the latent regime with `1.00` test accuracy;
- the graph scalar label baseline achieved `1.00` conditional test accuracy;
- the sheaf local-residual baseline achieved `1.00` conditional test accuracy;
- after repairing the cell circulation leak, the cell global-edge-sum baseline achieved `0.40` conditional test accuracy;
- the routed cheap-scalar system achieved `0.80` test accuracy.

This is a successful falsification check: the current tier is useful for pipeline bring-up but is too easy to support a routing-novelty claim. Gate 2 requires a confirmatory generator with overlapping reliability distributions or structurally defined corruptions and the preregistered shortcut baselines in the experimental plan.

## Tailnet state

The GB10 and iPhone peers have direct tailnet connectivity. The TensorBoard container returned HTTP 200 through `127.0.0.1:6006` during its smoke test and was then stopped. Both Tailscale Serve and Funnel retain empty configurations, and Compose publishes TensorBoard only on host loopback. Tailscale currently reports unreachable configured DNS servers; that health issue must be resolved and the HTTPS tailnet URL verified from the phone before remote dashboard access is considered operational.

## Software quality gate

The Stage-1 tree passes Ruff, Python byte-compilation, shell syntax validation, Compose configuration resolution, and `git diff --check`. The local suite reports 48 passed tests and one environment-conditional skip.

## Claim boundary

These checks validate software contracts, numerical conventions, synthetic-data mechanics, and basic GB10 execution. They do not show that RTD or a cone-conditioned objective improves learning, that a router beats fixed experts, or that the method transfers to molecular data.
