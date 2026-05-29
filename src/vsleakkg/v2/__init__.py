"""VS-LeakKG v2: typed contamination graph with Mode A / Mode B separation.

v2 differs from v1 in the following ways (see proposal.tex for details):
- Mode A (clean-split construction) vs Mode B (model-specific leakage audit)
  are formally separated.
- Contamination score is multiplicative S(pi) = prod_e w_r(e), computed via
  -log Dijkstra on axis-specific subgraphs (v1 used per-axis hit-flag weights).
- Hub-pollution mitigation: trivial-scaffold filter, degree caps, IDF.
- Model-specific TrainSet_m nodes for Mode B audits.
- Leakage-hub diagnostic H(x_train).
- Validation-contamination matrices: C(train->test), C(train->val), C(val->test).
- Feature leakage (path-based) vs label leakage (exact-row) reported separately.
- Giant-component fallback to Louvain community detection.
- Group-atomic split assignment with explicit residual contamination reporting.
- Hydrate side-table: (example_id) -> rich-row, the bridge between v2 splits
  and model adapters (SPRINT, DrugCLIP, LigUnity). See vsleakkg.v2.hydrate.

v2 lives alongside v1 (it does not replace it). v1 reproducibility is unaffected.
"""
from __future__ import annotations

__version__ = "2.0.0-dev"
