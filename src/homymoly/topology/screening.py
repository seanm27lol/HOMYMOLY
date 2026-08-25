"""Truth-aware heuristic screening for synthetic or oracle-known tasks.

The screen requires a known ground-truth candidate and a supplied sample of
reachable alternatives. It therefore applies directly to synthetic benchmarks
or oracle-known tasks, not ordinary unlabeled deployment data. The checks were
introduced during a retrospective audit; they are proposed for use before
freezing future campaigns, not claimed as prospective evidence for past ones.

Two inexpensive checks can expose an objective that is poorly matched to such a
task:

1. **The ground truth is not near a minimum of the term.** The objective then
   imposes bias away from the answer. This is a warning, not a proof of harm:
   biased regularisation can still improve finite-sample prediction.
2. **The term is constant over the supplied candidate class.** It then provides
   no differential signal for selecting among those candidates, whatever its
   weight.

A term passes this conservative screen only when the truth is near a minimum and
the term separates at least some supplied candidates. Passing is neither
necessary nor sufficient for better generalisation, and this screen does not
replace an experiment.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

Candidate = TypeVar("Candidate")

SATISFIED = "truth-no-higher-than-supplied-candidates-and-varies"
NOT_SATISFIED = "truth-higher-than-a-supplied-candidate"
CONSTANT = "constant-over-supplied-candidates"


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """Sample-relative heuristic verdict for one term and candidate set."""

    verdict: str
    truth_value: float
    truth_no_higher_than_supplied_candidates: bool
    candidate_minimum: float
    candidate_maximum: float
    relative_spread: float
    varies_over_supplied_candidates: bool

    @property
    def passes_heuristic(self) -> bool:
        """Whether this supplied sample passes both heuristic checks."""

        return self.verdict == SATISFIED

    def explain(self) -> str:
        if self.verdict == NOT_SATISFIED:
            return (
                "the truth scores higher than a supplied candidate "
                f"(truth {self.truth_value:.3e}, supplied-candidate minimum "
                f"{self.candidate_minimum:.3e}); the objective imposes bias, "
                "but this screen alone does not predict held-out harm"
            )
        if self.verdict == CONSTANT:
            return (
                "no variation was detected over the truth and supplied candidates "
                f"(relative spread {self.relative_spread:.3e}); it provides no "
                "differential selection signal there"
            )
        return (
            "the truth is no higher than any supplied candidate and the term "
            "separates supplied candidates "
            f"(relative spread {self.relative_spread:.3e}); this passes the "
            "heuristic screen but does not guarantee a training benefit"
        )


def screen_structural_term(
    term: Callable[[Candidate], Any],
    truth: Candidate,
    candidates: Sequence[Candidate],
    *,
    satisfied_atol: float = 1e-8,
    minimum_relative_spread: float = 1e-3,
) -> ScreeningResult:
    """Compare a truth score with, and test variation over, supplied candidates.

    ``term`` maps a candidate to a nonnegative scalar that a training objective
    would minimise. ``truth`` is the correct answer, and ``candidates`` are other
    members of the hypothesis class the model could reach.

    The truth passes the sample-relative ordering check when its value is no
    larger than the smallest supplied alternative up to ``satisfied_atol`` times
    the values' scale. This does not locate a class-wide or local optimum.
    Relative spread is measured over the truth *and* alternatives. Including the
    truth is essential: a term can separate a low-valued truth from supplied
    alternatives even when all those alternatives share one value.

    The result is a task-alignment diagnostic, not an estimator of generalisation
    benefit and not a substitute for a controlled experiment.
    """

    if not candidates:
        raise ValueError("screening needs at least one candidate to compare against")

    truth_value = float(term(truth))
    values = [float(term(candidate)) for candidate in candidates]
    all_values = [truth_value, *values]
    if not all(math.isfinite(value) for value in all_values):
        raise ValueError("screening term values must be finite")

    lowest, highest = min(values), max(values)
    overall_lowest, overall_highest = min(all_values), max(all_values)
    scale = max(abs(value) for value in all_values)
    spread = 0.0 if scale == 0.0 else (overall_highest - overall_lowest) / scale

    alignment_scale = max(1.0, abs(truth_value), abs(lowest))
    satisfied = truth_value <= lowest + satisfied_atol * alignment_scale
    varies = spread >= minimum_relative_spread
    if not satisfied:
        verdict = NOT_SATISFIED
    elif not varies:
        verdict = CONSTANT
    else:
        verdict = SATISFIED
    return ScreeningResult(
        verdict=verdict,
        truth_value=truth_value,
        truth_no_higher_than_supplied_candidates=satisfied,
        candidate_minimum=lowest,
        candidate_maximum=highest,
        relative_spread=spread,
        varies_over_supplied_candidates=varies,
    )


def boundary_compatibility_term(
    boundary: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``W -> ||boundary @ W^T||^2``, a compatibility diagnostic.

    The learned ``W`` implies a next boundary ``W^T``; this measures how far the
    implied complex is from satisfying ``d . d = 0``.  It does not measure
    exactness of a sequence: a zero candidate has zero defect regardless of
    whether its image equals the kernel of ``boundary``.
    """

    def term(candidate: torch.Tensor) -> torch.Tensor:
        return (boundary.to(candidate.dtype) @ candidate.mT).pow(2).sum()

    return term


def rtd_inspired_distance_term(
    source: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the campaign's normalized pairwise-distance surrogate.

    This helper intentionally does not call the quantity RTD: it contains no
    persistence or cross-barcode computation.  A candidate ``W`` maps the rows
    of ``source`` to ``source @ W.T``; the loss compares mean-normalized
    Euclidean distance matrices exactly as the historical campaign did.
    """

    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("source must be a rank-two floating tensor")
    source_distances = torch.cdist(source, source)
    normalized_source = source_distances / (source_distances.mean() + 1e-12)

    def term(candidate: torch.Tensor) -> torch.Tensor:
        if candidate.ndim != 2 or candidate.shape[1] != source.shape[1]:
            raise ValueError("candidate must have shape [output, source_features]")
        mapped_distances = torch.cdist(source @ candidate.mT, source @ candidate.mT)
        normalized_mapped = mapped_distances / (mapped_distances.mean() + 1e-12)
        return (normalized_mapped - normalized_source).pow(2).mean()

    return term


def exactness_term(boundary: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compatibility alias for the historical, mathematically imprecise name.

    New code should use :func:`boundary_compatibility_term`.  The alias remains
    available because early HOMYMOLY releases exported ``exactness_term``.
    """

    return boundary_compatibility_term(boundary)
