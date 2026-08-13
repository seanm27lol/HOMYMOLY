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
