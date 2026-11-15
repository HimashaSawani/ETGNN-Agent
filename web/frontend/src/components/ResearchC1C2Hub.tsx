import type React from 'react';
import { useState, useEffect } from 'react';
import { GitCompare, TrendingDown, AlertTriangle } from 'lucide-react';
import { fetchResearchC1, fetchResearchC2 } from '../api';
import type { C1Response, C2Response } from '../api';

export const ResearchC1C2Hub: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'c1' | 'c2'>('c2');
  const [c1Data, setC1Data] = useState<C1Response | null>(null);
  const [c2Data, setC2Data] = useState<C2Response | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([fetchResearchC1(), fetchResearchC2()])
      .then(([c1, c2]) => {
        setC1Data(c1);
        setC2Data(c2);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="c1-c2-hub">
      {/* Sub-tab navigation */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <button
          className={`tab-btn ${activeSubTab === 'c2' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('c2')}
          style={{ fontSize: 13, padding: '8px 16px' }}
        >
          Hypothesis C2: Matched Error Attribution (8 Registers)
        </button>
        <button
          className={`tab-btn ${activeSubTab === 'c1' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('c1')}
          style={{ fontSize: 13, padding: '8px 16px' }}
        >
          Hypothesis C1: Temporal Explanation Sensitivity
        </button>
      </div>

      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 0', color: '#64748b' }}>
          Loading frozen research artifacts...
        </div>
      ) : activeSubTab === 'c2' && c2Data ? (
        <div>
          {/* C2 Explanatory Banner */}
          <div className="safety-banner danger" style={{ marginBottom: 20 }}>
            <AlertTriangle size={20} color="#f43f5e" style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div className="safety-title" style={{ color: '#f43f5e' }}>
                Epistemic Meaning of C2: Prediction Sensitivity vs Causal Correctness
              </div>
              <div className="safety-desc">
                Experiment C2 evaluates whether explanation-guided edge deletions produce larger probability drops
                in correct predictions than in incorrect predictions under matched confidence and graph degree.
                Crucially, C2 measures model prediction sensitivity under intervention, NOT causal ground-truth correctness.
              </div>
            </div>
          </div>

          {/* Interactive Forest Plot */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">
                <GitCompare size={18} color="#38bdf8" />
                C2 Forest Plot: Paired Differences with 95% Bootstrap Confidence Intervals
              </div>
              <span className="badge badge-prototype">N=420 Matched Pairs Across 8 Registers</span>
            </div>

            <div style={{ padding: '10px 0' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 140px', gap: 14, paddingBottom: 10, borderBottom: '1px solid #1e293b', fontSize: 11, fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
                <div>Register Name</div>
                <div style={{ textAlign: 'center' }}>95% Bootstrap CI [Low, High] (Zero Line = Vertical Marker)</div>
                <div style={{ textAlign: 'right' }}>Mean Diff (SE)</div>
              </div>

              {c2Data.registers.map((reg) => {
                // Scale range from -0.15 to +0.25
                const minRange = -0.12;
                const maxRange = 0.22;
                const totalRange = maxRange - minRange;

                const leftPct = ((reg.ci_95_low - minRange) / totalRange) * 100;
                const rightPct = ((reg.ci_95_high - minRange) / totalRange) * 100;
                const meanPct = ((reg.mean_diff - minRange) / totalRange) * 100;
                const zeroPct = ((0 - minRange) / totalRange) * 100;

                const crossesZero = reg.ci_crosses_zero;
                const lineColor = crossesZero ? '#94a3b8' : '#38bdf8';

                return (
                  <div
                    key={reg.name}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '220px 1fr 140px',
                      gap: 14,
                      alignItems: 'center',
                      padding: '14px 0',
                      borderBottom: '1px solid rgba(30, 41, 59, 0.6)',
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: '#f8fafc' }}>{reg.name}</div>
                      <div style={{ fontSize: 11, color: crossesZero ? '#64748b' : '#38bdf8' }}>
                        {crossesZero ? 'Spans Zero (Inconclusive)' : 'CI Excludes Zero'}
                      </div>
                    </div>

                    {/* Forest Plot Whisker */}
                    <div style={{ position: 'relative', height: 26, background: '#0b101d', borderRadius: 4, overflow: 'hidden' }}>
                      {/* Zero line */}
                      <div
                        style={{
                          position: 'absolute',
                          left: `${zeroPct}%`,
                          top: 0,
                          bottom: 0,
                          width: 2,
                          background: 'rgba(244, 63, 94, 0.5)',
                          zIndex: 2,
                        }}
                      />

                      {/* CI Whisker bar */}
                      <div
                        style={{
                          position: 'absolute',
                          left: `${Math.max(0, leftPct)}%`,
                          width: `${Math.max(2, rightPct - leftPct)}%`,
                          top: 11,
                          height: 4,
                          background: lineColor,
                          borderRadius: 2,
                          zIndex: 3,
                        }}
                      />

                      {/* Mean dot */}
                      <div
                        style={{
                          position: 'absolute',
                          left: `calc(${meanPct}% - 5px)`,
                          top: 8,
                          width: 10,
                          height: 10,
                          borderRadius: '50%',
                          background: crossesZero ? '#fbbf24' : '#38bdf8',
                          boxShadow: '0 0 6px rgba(56, 189, 248, 0.6)',
                          zIndex: 4,
                        }}
                      />
                    </div>

                    {/* Numerical Column */}
                    <div className="mono" style={{ textAlign: 'right', fontSize: 12 }}>
                      <div style={{ fontWeight: 700, color: '#f8fafc' }}>
                        {reg.mean_diff > 0 ? `+${reg.mean_diff.toFixed(4)}` : reg.mean_diff.toFixed(4)}
                      </div>
                      <div style={{ fontSize: 11, color: '#64748b' }}>
                        ±{reg.bootstrap_se.toFixed(4)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop: 14, display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b' }}>
              <span>Legend: Red vertical line indicates zero difference.</span>
              <span>Cyan: CI excludes zero | Amber: CI crosses zero</span>
            </div>
          </div>

          {/* Detailed Registers Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Official C2 Register Interpretations</div>
            </div>

            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Register</th>
                    <th>Pairs</th>
                    <th>Mean Diff</th>
                    <th>95% CI</th>
                    <th>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {c2Data.registers.map((r) => (
                    <tr key={r.name}>
                      <td style={{ fontWeight: 600, color: '#f8fafc' }}>{r.name}</td>
                      <td className="mono">420</td>
                      <td className="mono" style={{ fontWeight: 700, color: r.mean_diff > 0 ? '#38bdf8' : '#94a3b8' }}>
                        {r.mean_diff > 0 ? `+${r.mean_diff.toFixed(4)}` : r.mean_diff.toFixed(4)}
                      </td>
                      <td className="mono">[{r.ci_95_low.toFixed(4)}, {r.ci_95_high.toFixed(4)}]</td>
                      <td style={{ fontSize: 12, color: '#cbd5e1' }}>{r.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : activeSubTab === 'c1' && c1Data ? (
        <div>
          {/* C1 Inconclusive Banner */}
          <div className="safety-banner info" style={{ marginBottom: 20 }}>
            <div>
              <div className="safety-title">Hypothesis C1 Verdict: INCONCLUSIVE</div>
              <div className="safety-desc">
                C1 evaluated whether ETGNN shows a larger prediction drop than Matched GAT when explanatory edges are deleted.
                Because the 95% confidence intervals cross zero across all cohorts, the experimental evidence is inconclusive.
                Sinusoidal time encodings do not confer statistically detectable explanation sensitivity.
              </div>
            </div>
          </div>

          {/* C1 Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <TrendingDown size={18} color="#38bdf8" />
                C1 Cohort Response Summary (results/tables/c1_response_summary.csv)
              </div>
            </div>

            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Cohort</th>
                    <th>N</th>
                    <th>Delta Adj Mean</th>
                    <th>Delta Adj SD</th>
                    <th>Delta Adj Median</th>
                    <th>Adv ETGNN Mean</th>
                    <th>Adv GAT Mean</th>
                  </tr>
                </thead>
                <tbody>
                  {c1Data.data.map((row) => (
                    <tr key={row.Cohort}>
                      <td style={{ fontWeight: 600, color: '#f8fafc' }}>{row.Cohort}</td>
                      <td className="mono">{row.N}</td>
                      <td className="mono">{row.delta_adj_mean.toFixed(4)}</td>
                      <td className="mono">±{row.delta_adj_sample_sd.toFixed(4)}</td>
                      <td className="mono">{row.delta_adj_median.toFixed(4)}</td>
                      <td className="mono">{row.adv_etgnn_mean.toFixed(4)}</td>
                      <td className="mono">{row.adv_gat_common_mean.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: 16, fontSize: 12, color: '#64748b' }}>
              Note: Sample SD reflects transaction variance across cohorts. Standard errors of the cohort means are smaller by 1/√N.
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
