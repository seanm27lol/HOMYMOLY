# Routing confirmatory v2 protocol (frozen 2026-08-13)

This protocol was written and the configurations below were frozen before any
result from seeds 20260906--20260910 was inspected. It repairs the adaptive
reuse of seeds 20260901--20260905 in the stabilization study.

## Claim under test

The narrow claim is that the stabilized, hard, one-expert-per-example router
improves test accuracy over the best single fixed expert on the confirmatory
synthetic benchmark. This is specifically the historical
**regime-distilled** router: validation regime labels estimate a
regime-by-expert utility table used as a training target. Regime is not an
inference input, but the supervision is privileged. The historical
target-view translators are retained so this campaign isolates routing; this
campaign does not test graph-only structure conversion or a cone loss.

## Frozen analysis

- Independent dataset/training seeds: 20260906, 20260907, 20260908,
  20260909, and 20260910.
- Primary per-seed endpoint: hard routed test accuracy minus the maximum of
  the three fixed-expert test accuracies.
- Primary aggregate: arithmetic mean and two-sided 95% Student-t confidence
  interval across the five per-seed margins.
- Decision rule: support the scoped claim only if the confidence interval's
  lower endpoint is greater than zero. Otherwise report it as inconclusive or
  unsupported, regardless of secondary endpoints.
- Secondary descriptive endpoints: routed-minus-dense accuracy, route
  accuracy, mutual information, utilization, measured latency, throughput,
  and peak allocated GPU memory. No multiplicity-adjusted confirmatory claim
  is attached to these endpoints.
- Every failed, gated, or collapsed run remains in the report. There is no
  seed replacement. Any post-result change starts a new exploratory study.

## Frozen configurations

| config | SHA-256 |
|---|---|
| `configs/routing-confirmatory-v2-s1.yaml` | `3295c825f19f28f5edc678d51be94cf7aa1eac5bf23b37b9994b4513e69f669e` |
| `configs/routing-confirmatory-v2-s2.yaml` | `f711646d1bf5c5c4ee9cb4fedf39f2b2edb5141444d3051042ecbacd4027fe13` |
| `configs/routing-confirmatory-v2-s3.yaml` | `072d66d569d168a1d6150403e0254ec9320dbc2c552b6945830fd915f858e75f` |
| `configs/routing-confirmatory-v2-s4.yaml` | `56b0059d4d41da4540ca186f7828cd87cfe7fc4a69f294dc21717a2a251c912b` |
| `configs/routing-confirmatory-v2-s5.yaml` | `b612ed011ef8e2ec139019a3713714d6e88896b3cf2ee9eefefab15a379ccaa1` |

The code revision is recorded in each run's environment metadata. The
configurations must not be edited after this freeze; fixes that can affect the
endpoint require new seeds and a new protocol version.

## Post-hoc provenance amendment

The following summarizer safeguards were documented while seed 5 was running,
after the valid campaign had begun. They are a post-hoc provenance amendment,
not part of the frozen statistical design. The primary endpoint, seed list,
decision rule, and five hashes above were already present in commit
`e69b07707950b6abe332366c51fe8c94254899f3`.

The campaign summarizer rejects any config-set/hash mismatch, any missing
final evaluation, and any disagreement in Git revision, executable-source
fingerprint, PyTorch/CUDA version, device, or device name. A gate-failed run
with a final test evaluation remains in the primary interval; it is never
replaced.

## Completed result

All five valid runs completed with no failed gate. The summarizer verified all
five config hashes and a shared Git revision, executable-source fingerprint,
and runtime environment:

- Git revision: `e69b07707950b6abe332366c51fe8c94254899f3`
- executable fingerprint:
  `473fb0f6714798274c38949107221df3bd941e89273a6eef76e54394d6c1f1d8`
- PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA GB10

Runs 2–5 recorded a dirty worktree as audit metadata, but all five valid runs
had the same executable fingerprint. The primary results in
`artifacts/routing-confirmatory-v2-summary.json` are:

| seed | hard accuracy | best fixed accuracy | primary margin | hard−dense |
|---|---:|---:|---:|---:|
| 20260906 | 0.802832 | 0.674292 | +0.128540 | +0.030501 |
| 20260907 | 0.774510 | 0.675381 | +0.099129 | +0.017429 |
| 20260908 | 0.767974 | 0.666667 | +0.101307 | +0.009804 |
| 20260909 | 0.778867 | 0.666667 | +0.112200 | +0.044662 |
| 20260910 | 0.774510 | 0.666667 | +0.107843 | +0.023965 |

The mean primary margin is **0.1098039216** (n=5; sample SD 0.0116918587;
two-sided Student-t 95% CI [0.0952865615, 0.1243212816]). The lower endpoint
is greater than zero, so the result **supports the scoped claim under the
frozen numerical decision rule**. All five seed margins are positive. As a
distribution-light sensitivity check, however, the exact two-sided sign-test
p-value is 0.0625. With n=5, distributional and independence assumptions are
not empirically checkable.

The test summaries contain `hard_milliseconds_per_example`; across seeds this
is 0.851879 ± 0.056617 ms/example (mean ± sample SD). It times one test-split
forward pass and excludes preprocessing and host transfer. There is no
separate profiler or peak-memory artifact, so this is not a complete latency,
throughput, or memory benchmark and no such secondary claim is made here.

## Protocol-integrity qualification

Before the valid seed-20260906 rerun, an aborted pre-commit attempt using a
different executable exposed validation metrics for that seed. The valid run
used the frozen config hash and shared committed executable fingerprint, and
the primary endpoint, decision rule, and config hashes were not changed.
Nevertheless, the prior observation is a procedural deviation: this campaign
must be described as **protocol-aligned with a disclosed deviation**, not as a
genuinely untouched campaign or pristine preregistration.

Finally, the result is limited to the historical regime-distilled router.
Training used privileged validation regime labels to form utility targets, and
inference used summaries of the available graph, active-face, and sheaf
transport observations. It is not evidence for graph-only or input-only
routing, graph-to-cell/sheaf conversion, a learned chain map, or a cone loss.
