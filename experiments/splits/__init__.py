"""Split protocol factory + 10 protocol implementations.

Each protocol exposes the same interface:

    build_split(examples_df, kg_dir, *, seed, **params) -> pl.DataFrame
        Returns (node_id, fold, leak_mask) where:
          fold      ∈ {"train", "val", "test"}
          leak_mask : protocol-specific flag for residual leak (default False)

Saved artefacts:
    data/splits/<corpus>/<protocol>__<param_str>__seed<n>.parquet
    data/splits/audit_summary.csv

Protocols:
  Baseline (Nhóm 1):  random, scaffold, tanimoto_maxmin, protein_cluster
  Paper (Nhóm 2):     datasail, plinder_style, ave_wallach
  KG (Nhóm 3, ours):  kg_kdisjoint, kg_maxmin, kg_axis_budget
"""
from .base import SplitResult, register_protocol, PROTOCOLS
from .factory import build_split, list_protocols

__all__ = ["SplitResult", "build_split", "list_protocols", "PROTOCOLS"]
