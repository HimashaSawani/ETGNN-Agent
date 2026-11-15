import type React from 'react';
import { useState, useEffect } from 'react';
import {
  GitCompare,
  TrendingDown,
  Layers,
  AlertTriangle,
  Info,
  CheckCircle2,
  Clock,
  Filter,
} from 'lucide-react';
import {
  fetchResearchTemporalStability,
  fetchResearchErrorMitigation,
  type TemporalStabilityResponse,
  type ErrorMitigationResponse,
} from '../api';

export const TemporalRobustness: React.FC = () => {
  const [stabData, setStabData] = useState<TemporalStabilityResponse | null>(null);
  const [mitData, setMitData] = useState<ErrorMitigationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [activeSubTab, setActiveSubTab] = useState<'census' | 'mitigation'>('census');
  const [familyFilter, setFamilyFilter] = useState<string>('ALL');
  const [selectedPeriod, setSelectedPeriod] = useState<string>('Full_Test_41_49 (N=9,973)');

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchResearchTemporalStability(), fetchResearchErrorMitigation()])
      .then(([sRes, mRes]) => {
        setStabData(sRes);
        setMitData(mRes);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load temporal robustness data');
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
        <Clock size={28} style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 14px' }} />
        <div>Loading Phase 12 Temporal Stability & Error Mitigation Evidence...</div>
      </div>
    );
  }

  if (error || !stabData || !mitData) {
    return (
      <div className="safety-banner danger" style={{ margin: '20px 0' }}>
        <AlertTriangle size={20} color="#f43f5e" />
        <div className="safety-desc">{error || 'Data could not be retrieved'}</div>
      </div>
    );
  }

  const census = stabData.census_summary;
  const filteredStrategies = mitData.mitigation_results.filter((strat) => {
    if (strat.Period !== selectedPeriod) return false;
    if (familyFilter === 'ALL') return true;
    return strat.Family.toUpperCase().includes(familyFilter);
  });

  return (
    <div className="temporal-robustness-view">
      {/* Header Banner */}
      <div className="card" style={{ marginBottom: 20, borderLeft: '4px solid #38bdf8' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <Clock size={22} color="#38bdf8" />
              <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc', margin: 0 }}>
                Temporal Analysis: Network Disruption & Mitigation Auditing
              </h2>
            </div>
            <p style={{ fontSize: 13, color: '#94a3b8', margin: 0, maxWidth: 840, lineHeight: 1.5 }}>
              Evaluating model resilience across the Step 43 darknet marketplace takedown, candidate-edge censorship census,
              and prospective 21-strategy error mitigation auditing on out-of-time test steps 41..49 (N=9,973 nodes).
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              className={`pill-btn ${activeSubTab === 'census' ? 'active' : ''}`}
              onClick={() => setActiveSubTab('census')}
              style={{ padding: '8px 16px', fontSize: 13 }}
            >
              <GitCompare size={14} style={{ marginRight: 6 }} />
              Exp A: Edge Shift Census & Stability
            </button>
            <button
              className={`pill-btn ${activeSubTab === 'mitigation' ? 'active' : ''}`}
              onClick={() => setActiveSubTab('mitigation')}
              style={{ padding: '8px 16px', fontSize: 13 }}
            >
              <Layers size={14} style={{ marginRight: 6 }} />
              Exp B: 21 Mitigation Strategies
            </button>
          </div>
        </div>
      </div>

      {activeSubTab === 'census' && (
        <>
          {/* Key Metric Highlights */}
          <div className="grid-4" style={{ marginBottom: 24 }}>
            <div className="stat-card" style={{ borderColor: 'rgba(244, 63, 94, 0.4)' }}>
              <div className="stat-label">Illicit Isolated Nodes (Pre-Shift)</div>
              <div className="stat-value" style={{ color: '#38bdf8' }}>
                {census.preshift_illicit_zero_edge_pct.toFixed(2)}%
              </div>
              <div className="stat-subtext">Steps 41..42: 48 of 355 illicit nodes zero-edge</div>
            </div>

            <div className="stat-card" style={{ borderColor: 'rgba(244, 63, 94, 0.7)' }}>
              <div className="stat-label">Illicit Isolated Nodes (Post-Shift)</div>
              <div className="stat-value" style={{ color: '#f43f5e' }}>
                {census.postshift_illicit_zero_edge_pct.toFixed(2)}%
              </div>
              <div className="stat-subtext">Steps 43..49: 130 of 169 illicit nodes zero-edge (+63.4% jump)</div>
            </div>

            <div className="stat-card" style={{ borderColor: 'rgba(168, 85, 247, 0.4)' }}>
              <div className="stat-label">Stratum 2 Matched Cohort</div>
              <div className="stat-value" style={{ color: '#c084fc' }}>
                N = {stabData.stability_summary.stratum_2_matched_pairs}
              </div>
              <div className="stat-subtext">Connected illicit pairs controlling for degree & time</div>
            </div>

            <div className="stat-card" style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}>
              <div className="stat-label">TOST Equivalence Verdict</div>
              <div className="stat-value" style={{ color: '#fbbf24', fontSize: 18, marginTop: 4 }}>
                Non-Equivalent
              </div>
              <div className="stat-subtext">Bounds ±0.05 and ±0.10 failed rejection (p ≥ 0.0821)</div>
            </div>
          </div>

          {/* Scientific Notice Banner */}
          <div className="safety-banner warning" style={{ marginBottom: 24 }}>
            <Info size={20} color="#fbbf24" style={{ flexShrink: 0 }} />
            <div className="safety-desc">
              <strong>Epistemic Warning & Scientific Disclosure:</strong>{' '}
              {stabData.scientific_takeaway}
            </div>
          </div>

          {/* Full Census Table */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">
                <GitCompare size={18} color="#38bdf8" />
                Table 5.8: Candidate-Edge Censorship Census Across Step 43 Structural Disruption
              </div>
              <span className="badge badge-dataset">Held-Out Test Nodes: 9,973</span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Evaluation Cohort</th>
                    <th>N Nodes</th>
                    <th>Illicit N (%)</th>
                    <th>E_cand = 0 (Isolated)</th>
                    <th>E_cand = 1</th>
                    <th>E_cand ≥ 2</th>
                    <th>Mean E_cand</th>
                    <th>ETGNN Empty Shift |Δp|</th>
                  </tr>
                </thead>
                <tbody>
                  {census.census_records.map((row, idx) => (
                    <tr key={idx} style={{ background: row.Cohort.includes('Post') ? 'rgba(244, 63, 94, 0.04)' : undefined }}>
                      <td style={{ fontWeight: 600, color: '#f8fafc' }}>{row.Cohort}</td>
                      <td>{row.N_Nodes.toLocaleString()}</td>
                      <td>
                        {row.Illicit_Count} ({row.Illicit_Pct.toFixed(1)}%)
                      </td>
                      <td style={{ color: row.E_cand_0_Pct > 50 ? '#f43f5e' : '#f8fafc', fontWeight: 600 }}>
                        {row.E_cand_0_Count} ({row.E_cand_0_Pct.toFixed(1)}%)
                      </td>
                      <td>
                        {row.E_cand_1_Count} ({row.E_cand_1_Pct.toFixed(1)}%)
                      </td>
                      <td>
                        {row.E_cand_ge2_Count} ({row.E_cand_ge2_Pct.toFixed(1)}%)
                      </td>
                      <td>{row.Mean_E_cand.toFixed(2)}</td>
                      <td>{row.Mean_Abs_Shift_Empty_ET165 ? row.Mean_Abs_Shift_Empty_ET165.toFixed(4) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Stratum 2 Multi-Seed Stability Audit */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <TrendingDown size={18} color="#c084fc" />
                Table 5.9: Stratum 2 Multi-Seed Paired Stability Audit (N=59 Matched Pairs)
              </div>
              <span className="badge" style={{ background: 'rgba(192, 132, 252, 0.15)', color: '#c084fc' }}>
                Seeds 42..46 Factorial
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Stratum</th>
                    <th>Metric</th>
                    <th>Pre-Shift Mean [95% CI]</th>
                    <th>Post-Shift Mean [95% CI]</th>
                    <th>Diff (Post - Pre) [95% CI]</th>
                    <th>Mann-Whitney p</th>
                    <th>Permutation p</th>
                  </tr>
                </thead>
                <tbody>
                  {stabData.stability_summary.strata_records.map((row, idx) => (
                    <tr key={idx}>
                      <td>
                        <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                          {row.Stratum}
                        </span>
                      </td>
                      <td style={{ fontWeight: 600, color: '#f8fafc' }}>{row.Metric_Description}</td>
                      <td>
                        {row.Pre_Mean.toFixed(4)} [{row.Pre_CI95_Low.toFixed(3)}, {row.Pre_CI95_High.toFixed(3)}]
                      </td>
                      <td>
                        {row.Post_Mean.toFixed(4)} [{row.Post_CI95_Low.toFixed(3)}, {row.Post_CI95_High.toFixed(3)}]
                      </td>
                      <td style={{ color: row.Diff_Post_minus_Pre < 0 ? '#f43f5e' : '#34d399' }}>
                        {row.Diff_Post_minus_Pre > 0 ? '+' : ''}
                        {row.Diff_Post_minus_Pre.toFixed(4)} [{row.Diff_CI95_Low.toFixed(3)}, {row.Diff_CI95_High.toFixed(3)}]
                      </td>
                      <td>{row.MannWhitney_P.toFixed(4)}</td>
                      <td>{row.Permutation_P.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {activeSubTab === 'mitigation' && (
        <>
          {/* Mitigation Overview Cards */}
          <div className="grid-3" style={{ marginBottom: 24 }}>
            <div className="stat-card" style={{ borderColor: 'rgba(56, 189, 248, 0.4)' }}>
              <div className="stat-label">Explanation-Guided Strategies (M1a..M3b)</div>
              <div className="stat-value" style={{ color: '#f43f5e', fontSize: 20 }}>
                Failed vs Controls
              </div>
              <div className="stat-subtext">M1a (0.2721) &lt; C1a (0.2915), p_Holm = 0.0140 (underperformed random)</div>
            </div>

            <div className="stat-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
              <div className="stat-label">Top Architectural Control (13_C2b)</div>
              <div className="stat-value" style={{ color: '#34d399', fontSize: 20 }}>
                AP: 0.4808
              </div>
              <div className="stat-subtext">Predicted-risk cascade; FPR 0.0144; Post-shift TP: 3 vs 5 (ΔTP = -2 vs baseline)</div>
            </div>

            <div className="stat-card" style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}>
              <div className="stat-label">Local Veto Control (19_C3c)</div>
              <div className="stat-value" style={{ color: '#fbbf24', fontSize: 20 }}>
                AP: 0.4442
              </div>
              <div className="stat-subtext">Local veto control; FPR 0.0177; Post-shift TP: 2 vs 5 (ΔTP = -3 vs base, -9 vs standalone 11 TP)</div>
            </div>
          </div>

          {/* Scientific Notice Banner */}
          <div className="safety-banner danger" style={{ marginBottom: 20 }}>
            <AlertTriangle size={20} color="#f43f5e" style={{ flexShrink: 0 }} />
            <div className="safety-desc">
              <strong>Critical Thesis Conclusion (Experiment B):</strong>{' '}
              {mitData.scientific_takeaway}
            </div>
          </div>

          {/* Filter Bar */}
          <div className="card" style={{ marginBottom: 20, padding: '12px 18px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 13, color: '#94a3b8', fontWeight: 600 }}>Evaluation Window:</span>
                <select
                  value={selectedPeriod}
                  onChange={(e) => setSelectedPeriod(e.target.value)}
                  className="search-input"
                  style={{ width: 230, fontSize: 12, padding: '4px 8px' }}
                >
                  <option value="Full_Test_41_49 (N=9,973)">Full Test Steps 41..49 (N=9,973)</option>
                  <option value="Pre_Shift_41_42 (N=3,286)">Pre-Shift Steps 41..42 (N=3,286)</option>
                  <option value="Post_Shift_43_49 (N=6,687)">Post-Shift Steps 43..49 (N=6,687)</option>
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <Filter size={15} color="#38bdf8" />
                <span style={{ fontSize: 12, color: '#cbd5e1' }}>Family:</span>
                {['ALL', 'EXPLANATION', 'CASCADE', 'ROUTING', 'VETO'].map((fam) => (
                  <button
                    key={fam}
                    className={`pill-btn ${familyFilter === fam ? 'active' : ''}`}
                    onClick={() => setFamilyFilter(fam)}
                    style={{ fontSize: 11, padding: '3px 10px' }}
                  >
                    {fam}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* 21 Strategies Full Table */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">
                <Layers size={18} color="#38bdf8" />
                Table 5.10: Prospective 21-Strategy Error Mitigation Census (Steps 41..49)
              </div>
              <span className="badge badge-dataset">Showing {filteredStrategies.length} of 21 Strategies</span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Strategy ID</th>
                    <th>Description</th>
                    <th>Family</th>
                    <th>Primary AP</th>
                    <th>ROC AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>FPR</th>
                    <th>ΔTP vs Base</th>
                    <th>ΔFP vs Base</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStrategies.map((row, idx) => {
                    const isTop = row.Strategy_ID.startsWith('13_C2b') || row.Strategy_ID.startsWith('19_C3c');
                    const isExpl = row.Family.toLowerCase().includes('explanation');
                    return (
                      <tr
                        key={idx}
                        style={{
                          background: isTop ? 'rgba(16, 185, 129, 0.07)' : isExpl ? 'rgba(244, 63, 94, 0.04)' : undefined,
                        }}
                      >
                        <td style={{ fontWeight: 700, color: isTop ? '#34d399' : '#f8fafc' }}>{row.Strategy_ID}</td>
                        <td style={{ fontSize: 12, maxWidth: 260 }}>{row.Strategy_Description}</td>
                        <td>
                          <span
                            className="badge"
                            style={{
                              background: isExpl ? 'rgba(244, 63, 94, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                              color: isExpl ? '#f43f5e' : '#38bdf8',
                            }}
                          >
                            {row.Family}
                          </span>
                        </td>
                        <td style={{ fontWeight: 700, color: row.AP_Primary >= 0.4 ? '#34d399' : '#f8fafc' }}>
                          {row.AP_Primary.toFixed(4)}
                        </td>
                        <td>{row.ROC_AUC.toFixed(4)}</td>
                        <td>{row.Precision.toFixed(4)}</td>
                        <td>{row.Recall.toFixed(4)}</td>
                        <td>{row.F1.toFixed(4)}</td>
                        <td>{row.FPR.toFixed(4)}</td>
                        <td style={{ color: row.Delta_TP_vs_Base < 0 ? '#f43f5e' : '#34d399' }}>
                          {row.Delta_TP_vs_Base > 0 ? '+' : ''}
                          {row.Delta_TP_vs_Base}
                        </td>
                        <td style={{ color: row.Delta_FP_vs_Base > 0 ? '#f43f5e' : '#34d399' }}>
                          {row.Delta_FP_vs_Base > 0 ? '+' : ''}
                          {row.Delta_FP_vs_Base}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Paired Bootstrap Contrasts */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <CheckCircle2 size={18} color="#34d399" />
                Table 5.11: Pre-Registered Paired Bootstrap Contrasts (10,000 Resamples, Holm FWER)
              </div>
              <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                Statistical Significance Audit
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Contrast</th>
                    <th>ΔAP [95% Bootstrap CI]</th>
                    <th>AP Raw p</th>
                    <th>AP Holm-Adj p</th>
                    <th>Holm Significant (α=0.05)</th>
                    <th>ΔF1 [95% CI]</th>
                    <th>ΔFPR [95% CI]</th>
                  </tr>
                </thead>
                <tbody>
                  {mitData.paired_bootstrap_contrasts.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ fontWeight: 600, color: '#f8fafc', fontSize: 13 }}>{row.Contrast}</td>
                      <td style={{ color: row.Delta_AP > 0 ? '#34d399' : '#f43f5e', fontWeight: 600 }}>
                        {row.Delta_AP > 0 ? '+' : ''}
                        {row.Delta_AP.toFixed(4)} [{row.AP_CI95_Low.toFixed(4)}, {row.AP_CI95_High.toFixed(4)}]
                      </td>
                      <td>{row.AP_Exact_P.toFixed(4)}</td>
                      <td style={{ fontWeight: 600, color: row['AP_Holm_Significant_0.05'] ? '#34d399' : '#94a3b8' }}>
                        {row.AP_Holm_Adjusted_P.toFixed(4)}
                      </td>
                      <td>
                        {row['AP_Holm_Significant_0.05'] ? (
                          <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>
                            Significant
                          </span>
                        ) : (
                          <span className="badge" style={{ background: 'rgba(148, 163, 184, 0.15)', color: '#94a3b8' }}>
                            Not Significant
                          </span>
                        )}
                      </td>
                      <td>
                        {row.Delta_F1 > 0 ? '+' : ''}
                        {row.Delta_F1.toFixed(4)} [{row.F1_CI95_Low.toFixed(3)}, {row.F1_CI95_High.toFixed(3)}]
                      </td>
                      <td>
                        {row.Delta_FPR > 0 ? '+' : ''}
                        {row.Delta_FPR.toFixed(4)} [{row.FPR_CI95_Low.toFixed(3)}, {row.FPR_CI95_High.toFixed(3)}]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
