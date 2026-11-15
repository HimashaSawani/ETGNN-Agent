import type React from 'react';
import { GitFork, ShieldAlert, ArrowDownLeft, ArrowUpRight, Network, ExternalLink } from 'lucide-react';
import type { GraphEvidenceSummary, GNNExplainerEvidenceSummary } from '../../types/agent';

interface Props {
  transactionId: number;
  snapshotId: number;
  graphEvidence?: GraphEvidenceSummary;
  gnnExplainerEvidence?: GNNExplainerEvidenceSummary | null;
  onNavigateToGraph?: () => void;
  onSelectNode?: (nodeId: number) => void;
}

export const InteractiveGraphEvidenceViewer: React.FC<Props> = ({
  transactionId,
  snapshotId,
  graphEvidence,
  gnnExplainerEvidence,
  onNavigateToGraph,
}) => {
  const inDegree = graphEvidence?.incoming_edges_count ?? 0;
  const outDegree = graphEvidence?.outgoing_edges_count ?? 0;
  const total2HopNodes = graphEvidence?.subgraph_nodes_2hop ?? 0;
  const illicitNeighbors = graphEvidence?.illicit_neighbors_count ?? 0;

  const isFullyIsolated = inDegree === 0 && outDegree === 0;
  const isIncomingIsolated = inDegree === 0 && outDegree > 0;
  const isOutgoingIsolated = inDegree > 0 && outDegree === 0;

  return (
    <div className="graph-evidence-viewer card" style={{ padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <GitFork size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Graph Topology & Neighborhood Evidence
          </h3>
        </div>

        {onNavigateToGraph && (
          <button
            className="btn btn-secondary"
            onClick={onNavigateToGraph}
            style={{ fontSize: 11, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 4 }}
          >
            <span>Open in Full Visualizer</span>
            <ExternalLink size={12} />
          </button>
        )}
      </div>

      {/* Snapshot 43 & Topology Qualification Badges */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
        <span
          className="badge"
          style={{ backgroundColor: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.3)' }}
        >
          Tx: #{transactionId}
        </span>
        <span
          className="badge"
          style={{
            fontSize: 11,
            backgroundColor: 'rgba(56, 189, 248, 0.12)',
            color: '#38bdf8',
            border: '1px solid rgba(56, 189, 248, 0.3)',
          }}
        >
          Snapshot ID: {snapshotId}
        </span>

        {isFullyIsolated ? (
          <span
            className="badge"
            title="Both in-degree and out-degree are zero in the examined graph snapshot."
            style={{
              fontSize: 11,
              backgroundColor: 'rgba(245, 158, 11, 0.15)',
              color: '#f59e0b',
              border: '1px solid rgba(245, 158, 11, 0.35)',
            }}
          >
            Fully Isolated Node (0 In, 0 Out)
          </span>
        ) : isIncomingIsolated ? (
          <span
            className="badge"
            title="Zero incoming observed edges in graph sample. Does not prove funding origination."
            style={{
              fontSize: 11,
              backgroundColor: 'rgba(56, 189, 248, 0.12)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
            }}
          >
            0 Observed In-Edges ({outDegree} Out)
          </span>
        ) : isOutgoingIsolated ? (
          <span
            className="badge"
            title="Zero outgoing observed edges in graph sample. Does not prove unspent funds or final accumulation."
            style={{
              fontSize: 11,
              backgroundColor: 'rgba(56, 189, 248, 0.12)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
            }}
          >
            {inDegree} In-Edges (0 Observed Out)
          </span>
        ) : (
          <span
            className="badge"
            style={{
              fontSize: 11,
              backgroundColor: 'rgba(16, 185, 129, 0.12)',
              color: '#10b981',
              border: '1px solid rgba(16, 185, 129, 0.3)',
            }}
          >
            Connected Ego-Net ({inDegree} In, {outDegree} Out)
          </span>
        )}

        {snapshotId === 43 && (
          <span
            className="badge"
            title="An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence."
            style={{
              fontSize: 11,
              backgroundColor: 'rgba(244, 63, 94, 0.15)',
              color: '#f43f5e',
              border: '1px solid rgba(244, 63, 94, 0.35)',
            }}
          >
            Snapshot 43: Observed Distribution Shift (Causality Not Established)
          </span>
        )}
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginBottom: 14 }}>
        <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-subtle, #182234)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted, #64748b)', fontSize: 11 }}>
            <ArrowDownLeft size={13} color="#38bdf8" /> In-Degree
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: '#38bdf8', marginTop: 2 }}>
            {inDegree}
          </div>
        </div>

        <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-subtle, #182234)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted, #64748b)', fontSize: 11 }}>
            <ArrowUpRight size={13} color="#a855f7" /> Out-Degree
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: '#a855f7', marginTop: 2 }}>
            {outDegree}
          </div>
        </div>

        <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-subtle, #182234)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted, #64748b)', fontSize: 11 }}>
            <Network size={13} color="#6366f1" /> 2-Hop Nodes
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: '#6366f1', marginTop: 2 }}>
            {total2HopNodes}
          </div>
        </div>

        <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-subtle, #182234)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted, #64748b)', fontSize: 11 }}>
            <ShieldAlert size={13} color={illicitNeighbors > 0 ? '#f43f5e' : '#10b981'} /> Illicit-Label Neighbors
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: illicitNeighbors > 0 ? '#f43f5e' : '#10b981', marginTop: 2 }}>
            {illicitNeighbors}
          </div>
        </div>
      </div>

      {/* Structural Description */}
      <div style={{ fontSize: 12, color: 'var(--text-secondary, #94a3b8)', lineHeight: 1.5 }}>
        {isFullyIsolated ? (
          <div>
            <strong>Topological Message Passing Status:</strong> Inactive. This transaction has zero observed directed edges in Snapshot {snapshotId} (both in-degree and out-degree are 0). ETGNN evaluates local tabular attributes without relational neighbor aggregation.
          </div>
        ) : isIncomingIsolated ? (
          <div>
            <strong>Topological Message Passing Status:</strong> Outgoing only ({outDegree} spending edge{outDegree > 1 ? 's' : ''}). Node has zero observed incoming edges in the dataset sample; incoming relational message passing does not contribute.
          </div>
        ) : isOutgoingIsolated ? (
          <div>
            <strong>Topological Message Passing Status:</strong> Incoming only ({inDegree} input edge{inDegree > 1 ? 's' : ''}). Node has zero observed outgoing edges in the dataset sample; downstream propagation does not contribute.
          </div>
        ) : (
          <div>
            <strong>Topological Message Passing Status:</strong> Active. ETGNN aggregated relational representations across {total2HopNodes} nodes in the 2-hop ego-network. Local graph context is integrated into the final layer representation.
          </div>
        )}
        <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', marginTop: 6, fontStyle: 'italic' }}>
          * Graph edges represent transaction payment linkages recorded in the Elliptic sample; they do not represent confirmed wallet identities, transaction amounts, or verified judicial findings.
        </div>
      </div>

      {/* GNNExplainer Attribution Summary */}
      {gnnExplainerEvidence && (
        <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border-subtle, #182234)', fontSize: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <strong style={{ color: 'var(--text-primary, #f8fafc)' }}>GNNExplainer Attribution:</strong>
            <span style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
              Top Feature Indices: [{gnnExplainerEvidence.top_feature_indices.join(', ')}] • Fidelity Drop: {gnnExplainerEvidence.fidelity_drop.toFixed(4)}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', fontStyle: 'italic', lineHeight: 1.4 }}>
            {gnnExplainerEvidence.interpretation_disclaimer ||
              'GNNExplainer identifies relative sensitivity in model representations, not deterministic legal proof of illicit activity.'}
          </div>
        </div>
      )}
    </div>
  );
};
