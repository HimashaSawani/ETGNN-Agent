"""
Temporal Graph Topology Drift Detection Module.
Tracks shifts in in-degree, out-degree, neighborhood size, edge density,
zero-in-degree and zero-out-degree proportions, while preserving the Snapshot 43 scientific caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SnapshotGraphTopologySummary:
    snapshot_id: int
    num_nodes: int
    num_edges: int
    mean_in_degree: float
    mean_out_degree: float
    max_in_degree: int
    max_out_degree: int
    zero_in_degree_proportion: float
    zero_out_degree_proportion: float
    edge_density: float
    is_snapshot_43: bool
    scientific_caveat: Optional[str]


@dataclass
class GraphTopologyDriftReport:
    total_snapshots: int
    baseline_window: str
    target_window: str
    baseline_mean_in_degree: float
    target_mean_in_degree: float
    in_degree_shift_delta: float
    baseline_zero_in_proportion: float
    target_zero_in_proportion: float
    snapshot_43_in_degree: float
    snapshot_43_zero_in_proportion: float
    snapshot_43_caveat: str
    snapshot_topologies: List[SnapshotGraphTopologySummary]


class GraphTopologyDriftDetector:
    """Measures graph structural changes and edge connectivity across chronological snapshots."""

    SNAPSHOT_43_CAVEAT = (
        "Snapshot 43 documents severe structural network reorganization (in-degree drops, "
        "elevated zero-incoming-edge proportion). External Darknet market takedown hypotheses "
        "remain non-causal observational context, not verified causal drivers."
    )

    def analyze_graph_drift(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        baseline_snapshots: Tuple[int, int] = (1, 32),
        target_snapshots: Tuple[int, int] = (41, 49),
    ) -> GraphTopologyDriftReport:
        """
        Computes topological degrees and connectivity across all snapshots.
        """
        snapshots = sorted(nodes_df["time_step"].unique())
        topologies: List[SnapshotGraphTopologySummary] = []

        # Map nodes to degrees
        # edges_df has columns ['txId1', 'txId2'] where txId1 -> txId2
        # Therefore: txId1 is source (out-degree), txId2 is target (in-degree)
        out_counts = edges_df["txId1"].value_counts().to_dict()
        in_counts = edges_df["txId2"].value_counts().to_dict()

        nodes_df = nodes_df.copy()
        nodes_df["in_degree"] = nodes_df["txId"].map(in_counts).fillna(0).astype(int)
        nodes_df["out_degree"] = nodes_df["txId"].map(out_counts).fillna(0).astype(int)

        base_mask = (nodes_df["time_step"] >= baseline_snapshots[0]) & (nodes_df["time_step"] <= baseline_snapshots[1])
        tgt_mask = (nodes_df["time_step"] >= target_snapshots[0]) & (nodes_df["time_step"] <= target_snapshots[1])

        base_in_deg = float(nodes_df.loc[base_mask, "in_degree"].mean()) if base_mask.any() else 0.0
        tgt_in_deg = float(nodes_df.loc[tgt_mask, "in_degree"].mean()) if tgt_mask.any() else 0.0

        base_zero_in = float((nodes_df.loc[base_mask, "in_degree"] == 0).mean()) if base_mask.any() else 0.0
        tgt_zero_in = float((nodes_df.loc[tgt_mask, "in_degree"] == 0).mean()) if tgt_mask.any() else 0.0

        s43_in_deg = 0.0
        s43_zero_in = 0.0

        for s in snapshots:
            s_nodes = nodes_df[nodes_df["time_step"] == s]
            n_nodes = len(s_nodes)
            if n_nodes == 0:
                continue

            in_degs = s_nodes["in_degree"]
            out_degs = s_nodes["out_degree"]

            m_in = float(in_degs.mean())
            m_out = float(out_degs.mean())
            zero_in_pct = float((in_degs == 0).mean())
            zero_out_pct = float((out_degs == 0).mean())

            # Edge density: approximate E / (N * (N-1)) if N > 1
            approx_edges = int(out_degs.sum())
            density = approx_edges / max(1, n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

            is_43 = bool(s == 43)
            caveat = self.SNAPSHOT_43_CAVEAT if is_43 else None

            if is_43:
                s43_in_deg = m_in
                s43_zero_in = zero_in_pct

            summary = SnapshotGraphTopologySummary(
                snapshot_id=int(s),
                num_nodes=n_nodes,
                num_edges=approx_edges,
                mean_in_degree=round(m_in, 4),
                mean_out_degree=round(m_out, 4),
                max_in_degree=int(in_degs.max()) if len(in_degs) else 0,
                max_out_degree=int(out_degs.max()) if len(out_degs) else 0,
                zero_in_degree_proportion=round(zero_in_pct, 4),
                zero_out_degree_proportion=round(zero_out_pct, 4),
                edge_density=round(float(density), 8),
                is_snapshot_43=is_43,
                scientific_caveat=caveat,
            )
            topologies.append(summary)

        return GraphTopologyDriftReport(
            total_snapshots=len(snapshots),
            baseline_window=f"Snapshots {baseline_snapshots[0]}-{baseline_snapshots[1]}",
            target_window=f"Snapshots {target_snapshots[0]}-{target_snapshots[1]}",
            baseline_mean_in_degree=round(base_in_deg, 4),
            target_mean_in_degree=round(tgt_in_deg, 4),
            in_degree_shift_delta=round(tgt_in_deg - base_in_deg, 4),
            baseline_zero_in_proportion=round(base_zero_in, 4),
            target_zero_in_proportion=round(tgt_zero_in, 4),
            snapshot_43_in_degree=round(s43_in_deg, 4),
            snapshot_43_zero_in_proportion=round(s43_zero_in, 4),
            snapshot_43_caveat=self.SNAPSHOT_43_CAVEAT,
            snapshot_topologies=topologies,
        )
