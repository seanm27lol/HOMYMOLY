"""Screen a structural term before writing a frozen protocol around it.

Every structural term this project has tested failed for one of exactly two
reasons, and both are checkable analytically in seconds:

1. **The ground truth does not satisfy it.** Then the term pulls away from the
   answer and can only hurt. The mapping cone in the conversion campaign is this
   case: penalising near-collapse biases a learned map away from the true one.
2. **It is constant over the hypothesis class.** Then it carries no information,
   whatever its weight. Cone acyclicity on the identifiable annulus is this case
   -- all twelve candidates are invertible, so every one is acyclic.

A term is worth a campaign only when the truth satisfies it *and* it separates
good candidates from bad ones. Run :func:`screen_structural_term` before
committing a protocol; it would have predicted every result in
``docs/26`` and ``docs/28`` without running anything.

This is a screening gate, not a guarantee. Passing means the term is not
obviously useless, not that it will help.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

Candidate = TypeVar("Candidate")

SATISFIED = "satisfied-and-varies"
NOT_SATISFIED = "ground-truth-violates-the-term"
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
                "the ground truth does not satisfy this term "
                f"(value {self.truth_value:.3e}); it will pull away from the answer"
            )
        if self.verdict == CONSTANT:
            return (
                "the term is effectively constant across the hypothesis class "
                f"(relative spread {self.relative_spread:.3e}); it carries no "
                "information at any weight"
            )
        return (
            "the truth satisfies the term and it separates candidates "
            f"(relative spread {self.relative_spread:.3e})"
        )


def screen_structural_term(
    term: Callable[[Candidate], Any],
    truth: Candidate,
    candidates: Sequence[Candidate],
    *,
    satisfied_atol: float = 1e-8,
    minimum_relative_spread: float = 1e-3,
) -> ScreeningResult:
    """Check that a structural term is satisfied by the truth and varies.

    ``term`` maps a candidate to a nonnegative scalar that a training objective
    would minimise. ``truth`` is the correct answer, and ``candidates`` are other
    members of the hypothesis class the model could reach.

    Relative spread is measured against the largest candidate value, so it does
    not depend on the term's units.
    """

    if not candidates:
        raise ValueError("screening needs at least one candidate to compare against")

    truth_value = float(term(truth))
    values = [float(term(candidate)) for candidate in candidates]
    lowest, highest = min(values), max(values)
    scale = max(abs(highest), abs(truth_value))
    spread = 0.0 if scale == 0.0 else (highest - lowest) / scale

    satisfied = truth_value <= satisfied_atol
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


def exactness_term(boundary: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``W -> ||boundary @ W^T||^2``, the term confirmed in docs/28.

    The learned ``W`` implies a next boundary ``W^T``; this measures how far the
    implied complex is from satisfying ``d . d = 0``.
    """

    def term(candidate: torch.Tensor) -> torch.Tensor:
        return (boundary.to(candidate.dtype) @ candidate.mT).pow(2).sum()

    return term
