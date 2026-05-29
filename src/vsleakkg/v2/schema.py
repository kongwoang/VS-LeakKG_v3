"""v2 schema: node types, edge types, and default leakage weights.

The weights match proposal.tex Table 2 (the "Default edge types and leakage
weights" table). Edit DEFAULT_WEIGHTS to change a release-wide default and bump
the version string in __init__.py. For run-time overrides, pass a dict to
score_axis() / score_overall() in scoring.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    EXAMPLE = "Example"
    PROTEIN = "Protein"
    PROTEIN_CLUSTER = "ProteinCluster"  # per-resolution; props["resolution"] = "30" | "40" | "90"
    POCKET = "Pocket"
    POCKET_CLUSTER = "PocketCluster"
    LIGAND = "Ligand"
    SCAFFOLD = "Scaffold"
    ASSAY = "Assay"
    PUBLICATION = "Publication"
    DATASET_SOURCE = "DatasetSource"
    DECOY_PROTOCOL = "DecoyProtocol"
    TIMEBIN = "TimeBin"
    TRAINSET = "TrainSet"  # Mode B: model-specific; props["model"] = "<model_id>"


class EdgeType(str, Enum):
    # binding edges connecting an Example to its constituent entities
    EXAMPLE_HAS_LIGAND = "example_has_ligand"
    EXAMPLE_HAS_PROTEIN = "example_has_protein"
    EXAMPLE_HAS_POCKET = "example_has_pocket"
    EXAMPLE_FROM_ASSAY = "example_from_assay"
    EXAMPLE_FROM_PUBLICATION = "example_from_publication"
    EXAMPLE_FROM_SOURCE = "example_from_source"
    EXAMPLE_HAS_TIMEBIN = "example_has_timebin"
    EXAMPLE_IN_TRAINSET = "example_in_trainset"  # Mode B

    # identity / similarity edges between content nodes
    LIGAND_EXACT = "ligand_exact"
    LIGAND_SCAFFOLD = "ligand_scaffold"
    LIGAND_SIMILAR = "ligand_similar"  # Morgan Tanimoto >= 0.85
    PROTEIN_EXACT = "protein_exact"
    PROTEIN_IN_CLUSTER = "protein_in_cluster"  # resolution stored on the cluster node
    POCKET_IN_CLUSTER = "pocket_in_cluster"
    POCKET_SIMILAR = "pocket_similar"  # embedding cosine >= 0.80
    SOURCE_DECOY_PROTOCOL = "source_decoy_protocol"
    TIME_OVERLAP = "time_overlap"


# Default edge weights mirror proposal.tex Table 2.
# Weights are in (0, 1]; 1.0 means "exact identity / strongest possible leak."
DEFAULT_WEIGHTS: dict[str, float] = {
    EdgeType.EXAMPLE_HAS_LIGAND.value: 1.00,
    EdgeType.LIGAND_EXACT.value: 1.00,
    EdgeType.LIGAND_SCAFFOLD.value: 0.70,
    EdgeType.LIGAND_SIMILAR.value: 0.65,
    EdgeType.EXAMPLE_HAS_PROTEIN.value: 1.00,
    EdgeType.PROTEIN_EXACT.value: 1.00,
    "protein_cluster_90": 0.85,
    "protein_cluster_40": 0.60,
    "protein_cluster_30": 0.45,
    EdgeType.EXAMPLE_HAS_POCKET.value: 1.00,
    EdgeType.POCKET_IN_CLUSTER.value: 0.70,
    EdgeType.POCKET_SIMILAR.value: 0.60,
    EdgeType.EXAMPLE_FROM_ASSAY.value: 0.75,
    EdgeType.EXAMPLE_FROM_PUBLICATION.value: 0.55,
    EdgeType.EXAMPLE_FROM_SOURCE.value: 0.35,
    EdgeType.SOURCE_DECOY_PROTOCOL.value: 0.50,
    EdgeType.TIME_OVERLAP.value: 0.40,
    EdgeType.EXAMPLE_IN_TRAINSET.value: 1.00,
    EdgeType.EXAMPLE_HAS_TIMEBIN.value: 1.00,
}


# Axis subgraphs: each axis is computed on its own subgraph (proposal section 5.5).
# An axis-specific subgraph uses example_has_* binding edges PLUS the relational
# edges listed for that axis. This ensures per-axis decomposition is well-defined
# even when paths could mix multiple edge types.
AXIS_EDGE_TYPES: dict[str, list[str]] = {
    "ligand": [
        EdgeType.EXAMPLE_HAS_LIGAND.value,
        EdgeType.LIGAND_EXACT.value,
        EdgeType.LIGAND_SIMILAR.value,
    ],
    "scaffold": [
        EdgeType.EXAMPLE_HAS_LIGAND.value,
        EdgeType.LIGAND_SCAFFOLD.value,
    ],
    "protein": [
        EdgeType.EXAMPLE_HAS_PROTEIN.value,
        EdgeType.PROTEIN_EXACT.value,
        EdgeType.PROTEIN_IN_CLUSTER.value,
    ],
    "pocket": [
        EdgeType.EXAMPLE_HAS_POCKET.value,
        EdgeType.POCKET_IN_CLUSTER.value,
        EdgeType.POCKET_SIMILAR.value,
    ],
    "assay": [
        EdgeType.EXAMPLE_FROM_ASSAY.value,
        EdgeType.EXAMPLE_FROM_PUBLICATION.value,
    ],
    "source": [
        EdgeType.EXAMPLE_FROM_SOURCE.value,
        EdgeType.SOURCE_DECOY_PROTOCOL.value,
    ],
    "time": [
        EdgeType.EXAMPLE_HAS_TIMEBIN.value,
        EdgeType.TIME_OVERLAP.value,
    ],
}

AXES: tuple[str, ...] = tuple(AXIS_EDGE_TYPES.keys())


@dataclass(frozen=True)
class HubMitigationConfig:
    """Hub-pollution mitigation parameters (proposal section 5.3).

    - trivial_scaffold_max_atoms: scaffolds with <= this many heavy atoms (and
      no substituents) are dropped from the scaffold axis. Default 6 = single
      ring like benzene with no chains.
    - degree_cap: nodes with degree > cap are split into per-source shards.
    - idf_floor: minimum weight after IDF downweighting (relative to nominal).
    """
    trivial_scaffold_max_atoms: int = 6
    degree_cap: int = 1000
    idf_floor: float = 0.10


@dataclass(frozen=True)
class GiantComponentConfig:
    """Giant-component fallback thresholds (proposal section 5.9)."""
    rho_max_ok: float = 0.30
    rho_max_prune: float = 0.60
    # Above rho_max_prune we fall back to Louvain community detection.


@dataclass(frozen=True)
class SplitConstraints:
    """Default group-assignment constraints (proposal section 5.10)."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    label_balance_tol: float = 0.05  # |D_k+|/|D_k| - |D+|/|D|
    min_targets_per_partition: int = 5
    min_actives_per_partition: int = 20
    lambda_size: float = 1.0
    lambda_label: float = 1.0
    lambda_cover: float = 0.5
    lambda_resid: float = 1.0
