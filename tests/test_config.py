from __future__ import annotations

from pathlib import Path

import pytest

from homymoly.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG = PROJECT_ROOT / "configs" / "stage1.yaml"


def test_stage1_config_loads_with_expected_data_envelope() -> None:
    config = load_config(STAGE1_CONFIG)

    assert config.schema_version == 1
    assert config.data.num_samples == 6144
    assert config.data.train_fraction == 0.7
    assert config.data.validation_fraction == 0.15
    assert config.data.min_vertices == 24
    assert config.data.max_vertices == 96
    assert config.data.node_feature_dim == 4
    assert config.data.edge_feature_dim == 2
    assert config.data.num_classes == 2
    assert config.runtime.precision == "bfloat16"
    assert config.profiling.matrix_sizes == (2048, 4096)


def test_artifact_override_resolves_and_creates_all_paths(tmp_path: Path) -> None:
    config = load_config(
        STAGE1_CONFIG,
        project_root=tmp_path,
        artifact_root="test-artifacts",
    )
    paths = config.artifact_paths().create()

    assert paths.root == (tmp_path / "test-artifacts").resolve()
    assert paths.tensorboard == paths.run / "tensorboard"
    assert all(Path(path).is_dir() for path in paths.as_dict().values())


@pytest.mark.parametrize(
    "document, message",
    [
        (
            "schema_version: 1\ndata:\n  min_vertices: 97\n  max_vertices: 96\n",
            "min_vertices",
        ),
        ("schema_version: 1\ndata:\n  num_classes: 3\n", "num_classes"),
        ("schema_version: 1\ndata:\n  num_samples: 17\n", "divisible by 6"),
        (
            "schema_version: 1\ndata:\n  train_fraction: 0.9\n  validation_fraction: 0.2\n",
            "must be less than 1",
        ),
        ("schema_version: 1\nunknown_section: true\n", "unknown top-level"),
        ("schema_version: 2\n", "unsupported schema_version"),
    ],
)
def test_invalid_config_is_rejected(
    tmp_path: Path, document: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path, project_root=tmp_path)
