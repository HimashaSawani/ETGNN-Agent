import type React from 'react';
import { useState, useEffect } from 'react';
import {
  HelpCircle,
  AlertTriangle,
  ShieldCheck,
  Cpu,
  ArrowDownRight,
  AlertCircle,
  Scissors,
  CheckCircle2,
  RefreshCw,
  GitBranch,
} from 'lucide-react';
import { fetchExplanation, fetchIntervention } from '../api';
import type { ExplanationResponse, InterventionResponse } from '../api';

interface Props {
  transactionId: number;
  onSelectExemplar: (txId: number) => void;
}

export const ExplanationInspector: React.FC<Props> = ({ transactionId, onSelectExemplar }) => {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [intervention, setIntervention] = useState<InterventionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isIntervening, setIsIntervening] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const [expRes, intRes] = await Promise.all([
        fetchExplanation(id),
        fetchIntervention(id).catch(() => null),
      ]);
      setExplanation(expRes);
      setIntervention(intRes);
    } catch (err: any) {
      setError(err.message || `Failed to load explanation for ${id}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunIntervention = async () => {
    if (!transactionId) return;
    setIsIntervening(true);
    try {
      const res = await fetchIntervention(transactionId);
      setIntervention(res);
    } catch (err: any) {
      setError(err.message || 'Intervention recomputation failed');
    } finally {
      setIsIntervening(false);
    }
  };

  useEffect(() => {
    if (transactionId) {
      loadData(transactionId);
    }
  }, [transactionId]);

  return (
    <div className="explanation-inspector">
      {/* Top Banner Notice */}
      <div className="safety-banner danger" style={{ marginBottom: 20 }}>
        <AlertTriangle size={20} color="#f43f5e" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div className="safety-title" style={{ color: '#f43f5e' }}>
            CRITICAL SCIENTIFIC SAFETY & LEGAL BOUNDARY
          </div>
          <div className="safety-desc">
            Never interpret GNNExplainer edge importance weights as evidence of criminal conspiracy or empirical guilt.
            In graph neural networks, edge masks quantify mathematical model prediction sensitivity under intervention.
            They identify graph pathways the model leveraged for its classification, not ground-truth illicit transactions.
          </div>
        </div>
      </div>

      {/* Control Header with Exemplar Quick Buttons */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <HelpCircle size={22} color="#a855f7" />
            <div>
              <div style={{ fontSize: 17, fontWeight: 700, color: '#f8fafc' }}>
                Transaction Explainability & Edge Attribution
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Analyzing GNNExplainer edge attribution masks and post-intervention probability drops
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: '#64748b' }}>Exemplar Artifacts:</span>
            <button className="btn" style={{ padding: '5px 10px', fontSize: 11 }} onClick={() => onSelectExemplar(288717023)}>
              TP (288717023)
            </button>
            <button className="btn" style={{ padding: '5px 10px', fontSize: 11 }} onClick={() => onSelectExemplar(116898498)}>
              FP (116898498)
            </button>
            <button className="btn" style={{ padding: '5px 10px', fontSize: 11 }} onClick={() => onSelectExemplar(149691712)}>
              TN (149691712)
            </button>
            <button className="btn" style={{ padding: '5px 10px', fontSize: 11 }} onClick={() => onSelectExemplar(13735016)}>
              FN (13735016)
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="safety-banner danger">
          <AlertCircle size={20} color="#f43f5e" />
          <div className="safety-desc">{error}</div>
        </div>
      )}

      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 0', color: '#64748b' }}>
          Loading explanation and edge attribution records...
        </div>
      ) : explanation ? (
        <>
          {/* Provenance & Status Card */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {explanation.is_frozen_research_artifact ? (
                  <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
                    <ShieldCheck size={14} /> Frozen Research Artifact
                  </span>
                ) : (
                  <span className="badge" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.4)' }}>
                    <Cpu size={14} /> Live Demonstration Output
                  </span>
                )}
                {explanation.case && (
                  <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc' }}>
                    Case: {explanation.case}
                  </span>
                )}
              </div>

              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Provenance: <code style={{ color: '#38bdf8' }}>{explanation.provenance}</code>
              </div>
            </div>

            {explanation.disclaimer && (
              <div style={{ marginTop: 12, fontSize: 12, color: '#fbbf24', background: 'rgba(245, 158, 11, 0.08)', padding: 10, borderRadius: 6 }}>
                Notice: {explanation.disclaimer}
              </div>
            )}
          </div>

          {/* Metric Stats */}
          <div className="grid-3" style={{ marginBottom: 24 }}>
            <div className="stat-card">
              <div className="stat-label">Original Prediction Probability</div>
              <div className="stat-value" style={{ color: (explanation.original_predicted_class_probability || 0) >= 0.5 ? '#f43f5e' : '#10b981' }}>
                {explanation.original_predicted_class_probability !== undefined
                  ? `${(explanation.original_predicted_class_probability * 100).toFixed(1)}%`
                  : 'N/A'}
              </div>
              <div className="stat-subtext">Before edge deletion intervention</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Neighborhood Scope</div>
              <div className="stat-value">
                {explanation.subgraph_nodes || '—'}{' '}
                <span style={{ fontSize: 14, color: '#64748b' }}>nodes / {explanation.subgraph_edges || '—'} edges</span>
              </div>
              <div className="stat-subtext">2-hop receptive field of target transaction</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Intervention Deletion Drop</div>
              <div className="stat-value" style={{ color: '#38bdf8' }}>
                {explanation.signed_drop !== undefined
                  ? `${(explanation.signed_drop * 100).toFixed(2)}%`
                  : 'Computed'}
              </div>
              <div className="stat-subtext">Confidence reduction when top edges are pruned</div>
            </div>
          </div>

          {/* Top Edges Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <ArrowDownRight size={18} color="#38bdf8" />
                GNNExplainer Top Attributed Edges (Sorted by Mask Weight)
              </div>
            </div>

            {explanation.top_edges && explanation.top_edges.length > 0 ? (
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Source Node Index</th>
                      <th>Target Node Index</th>
                      <th>Direction</th>
                      <th>Edge Mask Weight</th>
                      <th>Visual Importance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explanation.top_edges.map((edge, index) => (
                      <tr key={index}>
                        <td style={{ fontWeight: 700, color: '#f8fafc' }}>#{index + 1}</td>
                        <td className="mono">{edge.source_node_index}</td>
                        <td className="mono">{edge.target_node_index}</td>
                        <td>
                          <span className="badge" style={{ background: '#1e293b', color: '#94a3b8' }}>
                            {edge.source_node_index === explanation.node_index
                              ? 'Outgoing'
                              : edge.target_node_index === explanation.node_index
                              ? 'Incoming to Target'
                              : 'Inter-Neighbor'}
                          </span>
                        </td>
                        <td className="mono" style={{ fontWeight: 700, color: '#38bdf8' }}>
                          {edge.mask_value.toFixed(4)}
                        </td>
                        <td style={{ width: '30%' }}>
                          <div className="gauge-bar-bg" style={{ marginTop: 0 }}>
                            <div
                              className="gauge-bar-fill"
                              style={{
                                width: `${Math.min(100, Math.max(10, edge.mask_value * 100))}%`,
                                background: 'linear-gradient(90deg, #38bdf8, #a855f7)',
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: '30px 0' }}>
                No edge masks available for this transaction (isolated or degree 0).
              </div>
            )}
          </div>

          {/* Controlled Edge Intervention Workflow Panel */}
          {intervention && (
            <div className="card" style={{ marginTop: 24, border: '1px solid #0284c7' }}>
              <div className="card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Scissors size={20} color="#38bdf8" />
                  <div>
                    <div className="card-title" style={{ fontSize: 16 }}>
                      Controlled Edge Intervention Workflow (Research Protocol)
                    </div>
                    <div style={{ fontSize: 12, color: '#94a3b8' }}>
                      Counterfactual edge deletion (Protocol: <code style={{ color: '#38bdf8' }}>k = ceil(0.25 * E_cand)</code>) on frozen trained ETGNN
                    </div>
                  </div>
                </div>

                <button
                  className="btn"
                  onClick={handleRunIntervention}
                  disabled={isIntervening}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}
                >
                  <RefreshCw size={14} className={isIntervening ? 'animate-spin' : ''} />
                  <span>{isIntervening ? 'Recomputing...' : 'Re-run Intervention'}</span>
                </button>
              </div>

              {/* 4-Stat Intervention Metric Row */}
              <div className="grid-4" style={{ margin: '18px 0' }}>
                <div className="stat-card">
                  <div className="stat-label">1. Baseline Prediction</div>
                  <div className="stat-value" style={{ color: intervention.original_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                    {(intervention.original_probability * 100).toFixed(1)}%
                  </div>
                  <div className="stat-subtext">
                    Class: <span style={{ fontWeight: 700 }}>{intervention.original_predicted_class === 1 ? 'Illicit' : 'Licit'}</span> (Threshold = 0.50)
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-label">2. Eligible Edges & Budget</div>
                  <div className="stat-value" style={{ color: '#38bdf8' }}>
                    k = {intervention.budget_k} <span style={{ fontSize: 14, color: '#64748b' }}>/ {intervention.candidate_edges_count} edges</span>
                  </div>
                  <div className="stat-subtext">
                    Budget: <span style={{ fontWeight: 700 }}>{intervention.discrete_deletion_pct.toFixed(1)}% deletion</span>
                    {intervention.discrete_pct_explanation && (
                      <div style={{ fontSize: 11, color: '#fbbf24', marginTop: 2 }}>
                        {intervention.discrete_pct_explanation}
                      </div>
                    )}
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-label">3. Post-Intervention Score</div>
                  <div className="stat-value" style={{ color: intervention.intervened_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                    {(intervention.intervened_probability * 100).toFixed(1)}%
                  </div>
                  <div className="stat-subtext">
                    New Class: <span style={{ fontWeight: 700 }}>{intervention.intervened_predicted_class === 1 ? 'Illicit' : 'Licit'}</span> (Trained model frozen)
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-label">4. Sensitivity & Class Shift</div>
                  <div className="stat-value" style={{ color: intervention.class_changed ? '#f43f5e' : '#38bdf8' }}>
                    {intervention.score_difference >= 0 ? '+' : ''}
                    {(intervention.score_difference * 100).toFixed(2)}%
                  </div>
                  <div className="stat-subtext">
                    {intervention.class_changed ? (
                      <span className="badge" style={{ background: 'rgba(244, 63, 94, 0.2)', color: '#f43f5e', border: '1px solid rgba(244, 63, 94, 0.4)', marginTop: 4 }}>
                        Counterfactual Flip!
                      </span>
                    ) : (
                      <span className="badge" style={{ background: '#1e293b', color: '#94a3b8', marginTop: 4 }}>
                        Class Unchanged
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Deleted Edges Detail List */}
              {intervention.deleted_edges && intervention.deleted_edges.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#f8fafc', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <GitBranch size={14} color="#38bdf8" /> Excised One-Hop Candidate Edges (Frozen Architecture Forward-Pass):
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {intervention.deleted_edges.map((e, idx) => (
                      <div
                        key={idx}
                        className="mono"
                        style={{
                          background: '#0b1120',
                          border: '1px solid #1e293b',
                          padding: '6px 12px',
                          borderRadius: 6,
                          fontSize: 12,
                          color: '#38bdf8',
                        }}
                      >
                        Pruned #{idx + 1}: Node {e.source_node_index} &rarr; Node {e.target_node_index}
                        {e.mask_value !== undefined && (
                          <span style={{ color: '#a855f7', marginLeft: 6 }}>
                            (mask={e.mask_value.toFixed(4)})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Epistemic Safety & Methodological Distinction Banner */}
              <div
                style={{
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: 8,
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 12,
                  marginTop: 14,
                }}
              >
                <CheckCircle2 size={18} color="#38bdf8" style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
                  <strong style={{ color: '#38bdf8' }}>Academic & Statistical Distinction:</strong>{' '}
                  {intervention.epistemic_distinction}
                </div>
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
};
