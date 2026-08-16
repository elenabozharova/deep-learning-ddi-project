"""
GNN reproduction-gap diagnostics (Milestone 10, continued investigation).

Everything under this package is diagnostic-only: it reads the existing,
unmodified official artifacts (graph split, trained checkpoint) and/or runs
short instrumented training reusing the unchanged official recipe from
`polyllm.train_gnn`. Nothing here is imported by, or changes the behavior
of, the official pipeline (`train_gnn.py`, `evaluate_gnn.py`,
`models/gnn.py`, `graph/build_graph.py`). No diagnostic result is to be
treated as a new official result — see `notes/gnn_gap_diagnostics.md`.
"""
