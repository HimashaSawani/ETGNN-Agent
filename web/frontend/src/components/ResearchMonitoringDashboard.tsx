import { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart2,
  FileCheck,
  GitBranch,
  Lock,
  RefreshCw,
  Scale,
  Shield,
  Sliders,
  TrendingDown,
} from 'lucide-react';
import {
  fetchMonitoringDrift,
  fetchMonitoringEvaluation,
  fetchMonitoringCandidates,
  fetchMonitoringFeedbackStats,
} from '../api';

export function ResearchMonitoringDashboard() {
  const [activeSection, setActiveSection] = useState<'drift' | 'agent' | 'continual' | 'feedback'>('drift');
  const [driftData, setDriftData] = useState<any>(null);
  const [evalData, setEvalData] = useState<any>(null);
  const [candidatesData, setCandidatesData] = useState<any>(null);
  const [feedbackData, setFeedbackData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [drift, evaluation, candidates, feedback] = await Promise.all([
        fetchMonitoringDrift(),
        fetchMonitoringEvaluation(),
        fetchMonitoringCandidates(),
        fetchMonitoringFeedbackStats(),
      ]);
      setDriftData(drift);
      setEvalData(evaluation);
      setCandidatesData(candidates);
      setFeedbackData(feedback);
    } catch (err: any) {
      setError(err.message || 'Failed to load monitoring data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>
        <RefreshCw className="spin" size={28} style={{ marginBottom: 12 }} />
        <div>Loading Phase 5 Research Monitoring Telemetry...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 24, margin: 20, backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', borderRadius: 8 }}>
        <div style={{ color: '#ef4444', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
          <AlertTriangle size={18} />
          <span>Error loading research monitoring metrics</span>
        </div>
        <p style={{ marginTop: 8, color: '#fca5a5', fontSize: 13 }}>{error}</p>
        <button className="btn btn-secondary" onClick={loadAllData} style={{ marginTop: 12 }}>
          Retry Load
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1400, margin: '0 auto' }}>
      {/* Platform Header Banner */}
      <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Activity size={24} color="#38bdf8" />
            <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#f8fafc' }}>
              Phase 5: Research Monitoring & Epistemic Governance
            </h2>
          </div>
          <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Continuous temporal drift monitoring, agent ablation benchmarking, isolated continual learning gates, and feedback quality governance.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={loadAllData}
          style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, padding: '6px 12px' }}
        >
          <RefreshCw size={13} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Sub-Navigation Tabs */}
      <div style={{ display: 'flex', gap: 8, borderBottom: '1px solid #334155', marginBottom: 20, paddingBottom: 8 }}>
        <button
          className={`tab-btn ${activeSection === 'drift' ? 'active' : ''}`}
          onClick={() => setActiveSection('drift')}
          style={{ padding: '8px 16px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}
        >
          <TrendingDown size={14} />
          <span>Temporal Drift Detection</span>
        </button>
        <button
          className={`tab-btn ${activeSection === 'agent' ? 'active' : ''}`}
          onClick={() => setActiveSection('agent')}
          style={{ padding: '8px 16px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}
        >
          <Scale size={14} />
          <span>Agent Ablation & Citations</span>
        </button>
        <button
          className={`tab-btn ${activeSection === 'continual' ? 'active' : ''}`}
          onClick={() => setActiveSection('continual')}
          style={{ padding: '8px 16px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}
        >
          <Lock size={14} />
          <span>Continual Learning & Gates</span>
        </button>
        <button
          className={`tab-btn ${activeSection === 'feedback' ? 'active' : ''}`}
          onClick={() => setActiveSection('feedback')}
          style={{ padding: '8px 16px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}
        >
          <FileCheck size={14} />
          <span>Analyst Feedback Governance</span>
        </button>
      </div>

      {/* SECTION 1: TEMPORAL CONCEPT DRIFT */}
      {activeSection === 'drift' && driftData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Key Metric Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Feature Drift (PSI)</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#38bdf8', marginTop: 4 }}>
                {driftData.feature_drift?.mean_psi ?? 'N/A'}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                {driftData.feature_drift?.high_drift_features_count} high-drift features (PSI ≥ 0.25)
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Probability Shift</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#f59e0b', marginTop: 4 }}>
                {driftData.prediction_drift?.probability_shift_delta > 0 ? '+' : ''}
                {driftData.prediction_drift?.probability_shift_delta ?? '0.000'}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                Mean test prob: {driftData.prediction_drift?.target_mean_prob ?? 'N/A'}
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Snapshot 43 In-Degree</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#ef4444', marginTop: 4 }}>
                {driftData.graph_drift?.snapshot_43_in_degree ?? 'N/A'}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                Zero-in proportion: {((driftData.graph_drift?.snapshot_43_zero_in_proportion || 0) * 100).toFixed(1)}%
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Unlabelled Unknown Nodes</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981', marginTop: 4 }}>
                {driftData.label_drift?.overall_unlabelled_count?.toLocaleString() ?? 'N/A'}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                {((driftData.label_drift?.overall_unlabelled_pct || 0) * 100).toFixed(1)}% kept strictly unknown
              </div>
            </div>
          </div>

          {/* Snapshot 43 Scientific Non-Causal Caveat Alert */}
          <div style={{ backgroundColor: 'rgba(234, 179, 8, 0.08)', border: '1px solid rgba(234, 179, 8, 0.3)', borderRadius: 8, padding: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#facc15', fontWeight: 600, fontSize: 13 }}>
              <AlertTriangle size={16} />
              <span>Mandatory Scientific Caveat — Snapshot 43 Network Reorganization</span>
            </div>
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>
              {driftData.graph_drift?.snapshot_43_caveat}
            </p>
          </div>

          {/* Top Drifting Features Table */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <BarChart2 size={16} color="#38bdf8" />
              <span>Top Drifting Features (Training vs Test Partition)</span>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                  <th style={{ padding: '8px 12px' }}>Feature</th>
                  <th style={{ padding: '8px 12px' }}>PSI</th>
                  <th style={{ padding: '8px 12px' }}>KS Statistic</th>
                  <th style={{ padding: '8px 12px' }}>Wasserstein Dist</th>
                  <th style={{ padding: '8px 12px' }}>JS Divergence</th>
                  <th style={{ padding: '8px 12px' }}>Drift Level</th>
                </tr>
              </thead>
              <tbody>
                {driftData.feature_drift?.top_drifting_features?.map((f: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #2d3748', color: '#e2e8f0' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{f.feature_name}</td>
                    <td style={{ padding: '8px 12px' }}>{f.psi}</td>
                    <td style={{ padding: '8px 12px' }}>{f.ks_statistic} (p={f.ks_pvalue.toExponential(2)})</td>
                    <td style={{ padding: '8px 12px' }}>{f.wasserstein_distance}</td>
                    <td style={{ padding: '8px 12px' }}>{f.js_divergence}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 10,
                          fontWeight: 700,
                          backgroundColor:
                            f.drift_level === 'HIGH'
                              ? 'rgba(239, 68, 68, 0.2)'
                              : f.drift_level === 'MODERATE'
                              ? 'rgba(245, 158, 11, 0.2)'
                              : 'rgba(16, 185, 129, 0.2)',
                          color:
                            f.drift_level === 'HIGH'
                              ? '#ef4444'
                              : f.drift_level === 'MODERATE'
                              ? '#f59e0b'
                              : '#10b981',
                        }}
                      >
                        {f.drift_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SECTION 2: AGENT ABLATION & CITATIONS */}
      {activeSection === 'agent' && evalData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Live Provider Status Pill */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#1e293b', padding: 12, borderRadius: 8, border: '1px solid #334155' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#e2e8f0' }}>
              <Shield size={16} color="#38bdf8" />
              <span>Provider Execution Mode:</span>
              <span style={{ fontWeight: 700, color: '#a5b4fc' }}>
                {evalData.ablation_study_metadata?.live_provider_status?.status_text ?? 'Sandboxed Mock Mode'}
              </span>
            </div>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>
              Per Phase 5 rules, unconfigured live providers are marked NOT_RUN with zero fabricated results.
            </span>
          </div>

          {/* 6-System Ablation Table */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Scale size={16} color="#38bdf8" />
              <span>Phase 5 Multi-System Ablation Benchmark</span>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                  <th style={{ padding: '8px 12px' }}>System Configuration</th>
                  <th style={{ padding: '8px 12px' }}>Status</th>
                  <th style={{ padding: '8px 12px' }}>Factual Correctness</th>
                  <th style={{ padding: '8px 12px' }}>Citation Precision</th>
                  <th style={{ padding: '8px 12px' }}>Coverage</th>
                  <th style={{ padding: '8px 12px' }}>Unsupported Rate</th>
                  <th style={{ padding: '8px 12px' }}>Mean Latency</th>
                </tr>
              </thead>
              <tbody>
                {evalData.systems?.map((sys: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #2d3748', color: '#e2e8f0' }}>
                    <td style={{ padding: '8px 12px' }}>
                      <div style={{ fontWeight: 600 }}>{sys.system_name}</div>
                      <div style={{ fontSize: 10, color: '#64748b' }}>{sys.provider_type}</div>
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 10,
                          fontWeight: 700,
                          backgroundColor:
                            sys.execution_status === 'EXECUTED'
                              ? 'rgba(16, 185, 129, 0.2)'
                              : 'rgba(148, 163, 184, 0.2)',
                          color: sys.execution_status === 'EXECUTED' ? '#10b981' : '#94a3b8',
                        }}
                      >
                        {sys.execution_status}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{(sys.factual_correctness * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 12px' }}>{(sys.citation_precision * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 12px' }}>{(sys.citation_coverage * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 12px', color: sys.unsupported_claim_rate > 0.1 ? '#ef4444' : '#10b981' }}>
                      {(sys.unsupported_claim_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '8px 12px' }}>{sys.mean_latency_ms.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Semantic Citation Faithfulness Benchmark */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
                <FileCheck size={16} color="#10b981" />
                <span>Semantic Citation Faithfulness Benchmark</span>
              </div>
              <div style={{ fontSize: 12, color: '#a5b4fc', fontWeight: 600 }}>
                Accuracy vs Ground Truth: {((evalData.semantic_citation_benchmark?.accuracy_against_ground_truth || 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
              {Object.entries(evalData.semantic_citation_benchmark?.predicted_distribution || {}).map(([clsName, count]: any) => (
                <div key={clsName} style={{ backgroundColor: '#0f172a', padding: 10, borderRadius: 6, border: '1px solid #334155' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>{clsName}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc', marginTop: 2 }}>{count} items</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              Inter-rater agreement between deterministic rules and fallible judge:{' '}
              <strong style={{ color: '#38bdf8' }}>
                {((evalData.semantic_citation_benchmark?.inter_rater_agreement || 0) * 100).toFixed(1)}%
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 3: CONTINUAL LEARNING & GATES */}
      {activeSection === 'continual' && candidatesData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Checkpoints Immutable Status Card */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Lock size={20} color="#10b981" />
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc' }}>
                  Model Checkpoints Protection: IMMUTABLE READ-ONLY
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                  All 11 baseline checkpoints in results/checkpoints/ are locked and verified by SHA-256 manifests.
                </div>
              </div>
            </div>
          </div>

          {/* Promotion Gate Policies */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Sliders size={16} color="#38bdf8" />
              <span>Deterministic Model Promotion Gate Requirements</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <div style={{ backgroundColor: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Primary Metric</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#38bdf8', marginTop: 4 }}>AUPRC</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Min improvement: +1.5%</div>
              </div>
              <div style={{ backgroundColor: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>FPR Tolerance</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#f59e0b', marginTop: 4 }}>Δ ≤ +2.0%</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Zero regression allowed</div>
              </div>
              <div style={{ backgroundColor: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Governance Rule</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#ef4444', marginTop: 4 }}>Human Approval</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Automatic promotion prohibited</div>
              </div>
            </div>
          </div>

          {/* Candidate Models List */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: 8, border: '1px solid #334155', padding: 16 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
              <GitBranch size={16} color="#a5b4fc" />
              <span>Isolated Continual Learning Candidates (artifacts/phase5_candidates/)</span>
            </div>
            {candidatesData.candidates?.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12, padding: 12, textAlign: 'center' }}>
                No candidate retraining runs initiated. Original checkpoints remain untouched.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {candidatesData.candidates?.map((cand: any, idx: number) => (
                  <div key={idx} style={{ backgroundColor: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #334155' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ fontWeight: 600, color: '#f8fafc', fontSize: 13 }}>{cand.candidate_id}</div>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 10,
                          fontWeight: 700,
                          backgroundColor:
                            cand.status === 'BLOCKED' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(56, 189, 248, 0.2)',
                          color: cand.status === 'BLOCKED' ? '#ef4444' : '#38bdf8',
                        }}
                      >
                        {cand.status}
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                      Base Model: {cand.base_model_name} | Target: {cand.target_architecture}
                    </div>
                    {cand.blocking_reasons?.length > 0 && (
                      <div style={{ marginTop: 6, fontSize: 11, color: '#fca5a5' }}>
                        {cand.blocking_reasons.map((r: string, rIdx: number) => (
                          <div key={rIdx}>• {r}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* SECTION 4: ANALYST FEEDBACK GOVERNANCE */}
      {activeSection === 'feedback' && feedbackData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Safeguard Notice */}
          <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: 8, padding: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontWeight: 600, fontSize: 13 }}>
              <Shield size={16} />
              <span>Epistemic Governance Rule — Provenance Classification</span>
            </div>
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>
              {feedbackData.scientific_safeguard}
            </p>
          </div>

          {/* Feedback Statistics Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Total Records</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#38bdf8', marginTop: 4 }}>
                {feedbackData.total_feedback_records}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                {feedbackData.unique_transactions} unique transactions
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Mean Rating</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981', marginTop: 4 }}>
                {feedbackData.mean_rating?.toFixed(1)} / 5.0
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                {feedbackData.unique_reviewers} distinct reviewers
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Review Disagreement</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#a5b4fc', marginTop: 4 }}>
                {feedbackData.disagreement_incidents}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                Transactions with conflicting ratings
              </div>
            </div>

            <div style={{ backgroundColor: '#1e293b', padding: 16, borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>Citation Issue Rate</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#f59e0b', marginTop: 4 }}>
                {((feedbackData.citation_issue_frequency || 0) * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                Missing evidence: {((feedbackData.missing_evidence_frequency || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
