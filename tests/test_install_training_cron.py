from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "install_training_cron.py"
SPEC = importlib.util.spec_from_file_location("install_training_cron", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_managed_block_is_replaced_without_touching_other_jobs() -> None:
    document = (
        f"MAILTO=user@example.com\n{MODULE.BEGIN}\n*/5 * * * * old-command\n"
        f"{MODULE.END}\n0 2 * * * backup"
    )
    assert MODULE._without_managed_block(document) == [
        "MAILTO=user@example.com",
        "0 2 * * * backup",
    ]


@pytest.mark.parametrize(
    "document",
    (
        f"keep\n{MODULE.BEGIN}\nunsafe",
        f"keep\n{MODULE.END}\nunsafe",
        f"{MODULE.BEGIN}\n{MODULE.BEGIN}\n{MODULE.END}\n{MODULE.END}",
    ),
)
def test_malformed_managed_blocks_are_rejected(document: str) -> None:
    with pytest.raises(ValueError):
        MODULE._without_managed_block(document)
