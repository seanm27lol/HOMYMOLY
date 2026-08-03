"""OGBG-MOLHIV as structured samples with chemically valid ring 2-cells.

Per the plan's molecular gate: official OGB splits and evaluator are
preserved; rings enter only as oriented boundary-edge lists (never as
nonexistent triangles); the sheaf route is out of scope until a molecular
interpretation for its frames exists, so transports are identity.  Each
sample's ``regime`` is ``cell`` when the molecule contains at least one
ring and ``graph`` otherwise — the transfer analog of the synthetic
route-relevance design.  Samples are built once and cached to disk because
rdkit ring extraction over 41k molecules is not free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch.utils.data import Dataset

from .boundary import cycles_to_boundary_lists
from .types import SignalRegime, StructuredObservations, StructuredSample

_ATOM_DIMS = get_atom_feature_dims()
_BOND_DIMS = get_bond_feature_dims()
NODE_FEATURE_DIM = sum(_ATOM_DIMS)
EDGE_FEATURE_DIM = sum(_BOND_DIMS)


def _one_hot(features: np.ndarray, dims: list[int]) -> np.ndarray:
    encoded = np.zeros((features.shape[0], sum(dims)), dtype=np.float32)
    offset = 0
    for column, width in enumerate(dims):
        indices = features[:, column].clip(0, width - 1).astype(np.int64)
        encoded[np.arange(features.shape[0]), offset + indices] = 1.0
        offset += width
    return encoded


def _rings_from_smiles(smiles: str) -> list[tuple[int, ...]]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return []
    ring_info = molecule.GetRingInfo()
    return [tuple(int(atom) for atom in ring) for ring in ring_info.AtomRings()]


class MolecularHIVDataset(Dataset[StructuredSample]):
    """OGBG-MOLHIV with ring 2-cells in the boundary-list representation."""

    def __init__(self, root: str | Path = "artifacts/molecular") -> None:
        from ogb.graphproppred import GraphPropPredDataset

        self.root = Path(root)
        # OGB loads its own processed cache via a bare torch.load, which
        # fails under torch>=2.6's weights_only default; the cache is OGB's
        # own artifact, so a scoped weights_only=False shim is safe.
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(  # type: ignore[assignment]
            *args, **{**kwargs, "weights_only": False}
        )
        try:
            dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(self.root))
        finally:
            torch.load = original_load  # type: ignore[assignment]
        smiles = pd.read_csv(
            self.root / "ogbg_molhiv" / "mapping" / "mol.csv.gz"
        )["smiles"].tolist()
        cache = self.root / "ogbg_molhiv" / "processed" / "homymoly_samples.pt"
        if cache.exists():
            self._samples = torch.load(cache, weights_only=False)
        else:
            self._samples = [
                self._build_sample(index, dataset[index], smiles[index])
                for index in range(len(dataset))
            ]
            torch.save(self._samples, cache)
        self._split = dataset.get_idx_split()

    @property
    def splits(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(int(index) for index in indices)
            for name, indices in self._split.items()
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> StructuredSample:
        return self._samples[index]

    @staticmethod
    def _build_sample(index: int, graph_and_label, smiles: str) -> StructuredSample:
        graph, label = graph_and_label
        num_vertices = int(graph["node_feat"].shape[0])
        node_features = torch.as_tensor(
            _one_hot(graph["node_feat"], _ATOM_DIMS), dtype=torch.float32
        )

        # Canonical undirected edges: u < v, unique, lexicographically sorted.
        directed = graph["edge_index"].astype(np.int64)
        directed_feat = graph["edge_feat"]
        canonical: dict[tuple[int, int], np.ndarray] = {}
        for position in range(directed.shape[1]):
            u, v = (int(directed[0, position]), int(directed[1, position]))
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key not in canonical:
                canonical[key] = directed_feat[position]
        edges = sorted(canonical)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_features = torch.as_tensor(
            _one_hot(np.stack([canonical[edge] for edge in edges]), _BOND_DIMS),
            dtype=torch.float32,
        ) if edges else torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

        rings = [ring for ring in _rings_from_smiles(smiles) if len(ring) >= 3]
        if rings:
            face_boundary, face_vertices = cycles_to_boundary_lists(edge_index, rings)
        else:
            face_boundary = torch.zeros((0, 3, 2), dtype=torch.long)
            face_vertices = torch.full((0, 3), -1, dtype=torch.long)
        triangles = sorted(
            tuple(sorted(ring)) for ring in rings if len(ring) == 3
        )
        face_index = (
            torch.tensor(triangles, dtype=torch.long).t().contiguous()
            if triangles
            else torch.zeros((3, 0), dtype=torch.long)
        )
        face_active = torch.ones(len(rings), dtype=torch.bool)
        transport = torch.eye(2, dtype=torch.float32).repeat(edge_index.shape[1], 1, 1)

        return StructuredSample(
            observations=StructuredObservations(
                node_features=node_features,
                edge_features=edge_features,
            ),
            edge_index=edge_index,
            face_index=face_index,
            face_active=face_active,
            transport=transport,
            label=torch.tensor(int(label[0]), dtype=torch.long),
            regime=SignalRegime.CELL if rings else SignalRegime.GRAPH,
            sample_id=f"ogbg-molhiv-{index:06d}",
            metadata={
                "generator": "MolecularHIVDataset",
                "generator_version": 1,
                "num_vertices": num_vertices,
                "num_edges": int(edge_index.shape[1]),
                "num_faces": len(rings),
            },
            face_boundary=face_boundary,
            face_vertices=face_vertices,
        )


__all__ = ["EDGE_FEATURE_DIM", "NODE_FEATURE_DIM", "MolecularHIVDataset"]
