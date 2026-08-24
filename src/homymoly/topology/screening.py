"""Heuristically screen a structural term before freezing a training campaign.

Two inexpensive checks can expose an objective that is poorly matched to the
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

SATISFIED = "truth-near-minimum-and-varies"
NOT_SATISFIED = "truth-not-near-minimum"
CONSTANT = "constant-over-the-hypothesis-class"


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """Verdict on whether a structural term can carry information."""

    verdict: str
    truth_value: float
    satisfied_by_truth: bool
    candidate_minimum: float
    candidate_maximum: float
    relative_spread: float
    varies_over_class: bool

    @property
    def usable(self) -> bool:
        return self.verdict == SATISFIED

    def explain(self) -> str:
        if self.verdict == NOT_SATISFIED:
            return (
                "the ground truth is not near a minimum of this term "
                f"(truth {self.truth_value:.3e}, supplied-candidate minimum "
                f"{self.candidate_minimum:.3e}); the objective imposes bias, "
                "but this screen alone does not predict held-out harm"
            )
        if self.verdict == CONSTANT:
            return (
                "the term is effectively constant across the supplied candidates "
                f"(relative spread {self.relative_spread:.3e}); it provides no "
                "differential selection signal there"
            )
        return (
            "the truth is near a minimum and the term separates supplied candidates "
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
    """Check whether a loss is minimised near the truth and varies over candidates.

    ``term`` maps a candidate to a nonnegative scalar that a training objective
    would minimise. ``truth`` is the correct answer, and ``candidates`` are other
    members of the hypothesis class the model could reach.

    The truth is treated as near-minimal when its value is no larger than the
    smallest supplied alternative up to ``satisfied_atol`` times the values'
    scale. Relative spread is measured over the truth *and* alternatives. Including
    the truth is essential: a term can perfectly separate a low-valued truth from
    wrong alternatives even when all those alternatives share one value.

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
        satisfied_by_truth=satisfied,
        candidate_minimum=lowest,
        candidate_maximum=highest,
        relative_spread=spread,
        varies_over_class=varies,
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


def exactness_term(boundary: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compatibility alias for the historical, mathematically imprecise name.

    New code should use :func:`boundary_compatibility_term`.  The alias remains
    available because early HOMYMOLY releases exported ``exactness_term``.
    """

    return boundary_compatibility_term(boundary)
