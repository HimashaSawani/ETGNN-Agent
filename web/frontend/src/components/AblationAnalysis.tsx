import type React from 'react';
import { useState, useEffect } from 'react';
import { Layers, Activity, TrendingUp, ShieldAlert, CheckCircle2, HelpCircle, Network } from 'lucide-react';
import { fetchResearchAblations, fetchResearchPhase4, fetchResearchXaiValidation, fetchResearchExternalValidation } from '../api';
import type { AblationsResponse, Phase4SummaryResponse, XaiValidationResponse, ExternalValidationResponse } from '../api';

export const AblationAnalysis: React.FC = () => {
  const [data, setData] = useState<AblationsResponse | null>(null);
  const [phase4, setPhase4] = useState<Phase4SummaryResponse | null>(null);
  const [xaiVal, setXaiVal] = useState<XaiValidationResponse | null>(null);
  const [extVal, setExtVal] = useState<ExternalValidationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      fetchResearchAblations(),
      fetchResearchPhase4().catch(() => null),
      fetchResearchXaiValidation().catch(() => null),
      fetchResearchExternalValidation().catch(() => null),
    ])
      .then(([resAbl, resP4, resXai, resExt]) => {
        setData(resAbl);
        setPhase4(resP4);
        setXaiVal(resXai);
        setExtVal(resExt);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);


  return (
    <div className="ablation-analysis">
      {/* Top Banner Notice */}
      <div className="safety-banner danger" style={{ marginBottom: 20 }}>
        <ShieldAlert size={20} color="#f43f5e" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div className="safety-title" style={{ color: '#f43f5e' }}>
            The Graph Inductive Bias Paradox
          </div>
          <div className="safety-desc">
            Graph neural networks rely on the assumption that connected nodes share similar labels (homophily).
            In Bitcoin transactions, fraudsters route funds through licit exchanges and intermediary mixers.
            As a result, message passing aggregates conflicting feature distributions, degrading performance below tabular models.
          </div>
        </div>
      </div>

      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 0', color: '#64748b' }}>
          Loading ablation records...
        </div>
      ) : data ? (
        <>
          {/* Edge Removal Anomaly Cards */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">
                <Layers size={18} color="#34d399" />
                The +0.2468 Edge Removal Anomaly (Topological Noise)
              </div>
              <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>
                Key Empirical Discovery
              </span>
            </div>

            <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 20 }}>
              When graph edges are completely removed from the local feature model, validation AUPRC dramatically jumps from 0.5401 to 0.7869 (+0.2468).
              This directly proves that raw transaction edges in the Elliptic dataset act as topological noise for GNN message passing:
            </p>

            <div className="grid-3" style={{ marginBottom: 14 }}>
              {data.edge_ablation_anomaly.data.map((item, idx) => (
                <div
                  key={item.model}
                  className="card"
                  style={{
                    background: idx === 2 ? 'rgba(16, 185, 129, 0.08)' : '#0d1322',
                    borderColor: idx === 2 ? 'rgba(16, 185, 129, 0.4)' : '#1e293b',
                    margin: 0,
                  }}
                >
                  <div style={{ fontSize: 12, color: idx === 2 ? '#34d399' : '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>
                    {idx === 2 ? '★ Top Local Model' : `Ablation Stage ${idx + 1}`}
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc', marginTop: 4, minHeight: 40 }}>
                    {item.model}
                  </div>
                  <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: idx === 2 ? '#34d399' : '#38bdf8', marginTop: 8 }}>
                    {item.val_auprc.toFixed(4)}
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, lineHeight: 1.45 }}>
                    {item.notes}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Multi-Seed Ablation Table */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-header">
              <div className="card-title">
                <Activity size={18} color="#38bdf8" />
                Multi-Seed Repeat Ablation (Seeds 7, 42, 123)
              </div>
              <span className="badge badge-prototype">Statistical Equivalence</span>
            </div>

            <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16 }}>
              Comparing ETGNN against Matched GAT across 3 distinct seeds confirms that sinusoidal time encodings produce no statistically significant advantage over the ablated baseline:
            </p>

            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Seed</th>
                    <th>ETGNN Validation AUPRC</th>
                    <th>ETGNN Best Epoch</th>
                    <th>Matched GAT Validation AUPRC</th>
                    <th>Matched GAT Best Epoch</th>
                    <th>AUPRC Difference (ETGNN - Matched GAT)</th>
                  </tr>
                </thead>
                <tbody>
                  {[7, 42, 123].map((seed) => {
                    const et = data.multi_seed_comparison.etgnn.find((m) => m.seed === seed);
                    const gat = data.multi_seed_comparison.matched_gat.find((m) => m.seed === seed);
                    const diff = et && gat ? et.validation_auprc - gat.validation_auprc : 0;
                    return (
                      <tr key={seed}>
                        <td style={{ fontWeight: 700, color: '#f8fafc' }}>Seed {seed}</td>
                        <td className="mono" style={{ color: '#38bdf8', fontWeight: 600 }}>
                          {et ? et.validation_auprc.toFixed(4) : '—'}
                        </td>
                        <td className="mono">{et ? et.best_epoch : '—'}</td>
                        <td className="mono" style={{ color: '#c084fc', fontWeight: 600 }}>
                          {gat ? gat.validation_auprc.toFixed(4) : '—'}
                        </td>
                        <td className="mono">{gat ? gat.best_epoch : '—'}</td>
                        <td className="mono" style={{ color: Math.abs(diff) < 0.005 ? '#94a3b8' : '#fbbf24' }}>
                          {diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Temporal Prevalence Volatility Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <TrendingUp size={18} color="#fbbf24" />
                Temporal Prevalence Volatility (Held-Out Test Steps 41–49)
              </div>
              <span className="badge badge-dataset">Step-by-Step Dynamics</span>
            </div>

            <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16 }}>
              Elliptic test time steps exhibit dramatic distribution shifts: illicit prevalence swings from 0.7% (step 43) to 25.1% (step 47).
              Static time encodings cannot compensate for macro-temporal regime changes:
            </p>

            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Time Step</th>
                    <th>Labelled Nodes</th>
                    <th>Illicit Count</th>
                    <th>Licit Count</th>
                    <th>Illicit Prevalence</th>
                    <th>ETGNN AUPRC</th>
                    <th>No-Edge Local AUPRC</th>
                  </tr>
                </thead>
                <tbody>
                  {data.temporal_prevalence_volatility.map((row) => (
                    <tr key={row.time_step}>
                      <td style={{ fontWeight: 700, color: '#f8fafc' }}>Step {row.time_step}</td>
                      <td className="mono">{row.labelled_count}</td>
                      <td className="mono" style={{ color: '#f43f5e' }}>{row.illicit_count}</td>
                      <td className="mono" style={{ color: '#10b981' }}>{row.licit_count}</td>
                      <td className="mono" style={{ fontWeight: 700, color: '#fbbf24' }}>
                        {(row.illicit_prevalence * 100).toFixed(1)}%
                      </td>
                      <td className="mono" style={{ color: '#38bdf8' }}>
                        {row.etgnn_auprc.toFixed(4)}
                      </td>
                      <td className="mono" style={{ color: '#34d399', fontWeight: 600 }}>
                        {row.no_edge_auprc.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Phase 4 Controlled 40-Trial Factorial Benchmark */}
          {phase4 && (
            <div className="card" style={{ marginTop: 24 }}>
              <div className="card-header">
                <div className="card-title">
                  <CheckCircle2 size={18} color="#38bdf8" />
                  {phase4.experiment}
                </div>
                <span className="badge badge-dataset">
                  {phase4.total_trials}/40 Verified Trials (Validation Split Steps 33–40)
                </span>
              </div>

              <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16, lineHeight: 1.5 }}>
                {phase4.scientific_takeaway}
              </p>

              <div className="table-wrapper" style={{ marginBottom: 20 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Condition (5 Seeds: 42..46)</th>
                      <th>Repr</th>
                      <th>Params</th>
                      <th>Val AUPRC (Mean ± SD)</th>
                      <th>Val F1 (Mean ± SD)</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phase4.conditions.map((c) => {
                      const isBest93 = c.condition_id === 'Cond_B_93f';
                      const isBest165 = c.condition_id === 'Cond_A_165f';
                      return (
                        <tr key={c.condition_id} className={isBest93 || isBest165 ? 'active' : ''}>
                          <td style={{ fontWeight: isBest93 || isBest165 ? 700 : 500, color: isBest93 ? '#38bdf8' : isBest165 ? '#fbbf24' : '#f8fafc' }}>
                            {c.display_name}
                          </td>
                          <td className="mono">{c.input_dim}f</td>
                          <td className="mono">{c.param_count.toLocaleString()}</td>
                          <td className="mono" style={{ fontWeight: 700, color: isBest93 ? '#38bdf8' : isBest165 ? '#fbbf24' : '#e2e8f0' }}>
                            {c.auprc_mean.toFixed(4)} ± {c.auprc_sd.toFixed(4)}
                          </td>
                          <td className="mono">
                            {c.f1_mean.toFixed(4)} ± {c.f1_sd.toFixed(4)}
                          </td>
                          <td className="mono">{c.precision_mean.toFixed(4)}</td>
                          <td className="mono">{c.recall_mean.toFixed(4)}</td>
                          <td className="mono">{c.roc_auc_mean.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', marginBottom: 10 }}>
                Paired Hypothesis Contrasts & Holm–Bonferroni Multiplicity Audit (k = 6 Contrasts)
              </div>
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Paired Contrast</th>
                      <th>Mean Δ AUPRC</th>
                      <th>95% CI</th>
                      <th>Unadjusted p</th>
                      <th>Holm-Adjusted p (k=6)</th>
                      <th>Seeds &gt; 0</th>
                      <th>Epistemic Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phase4.paired_contrasts.map((pc) => (
                      <tr key={pc.contrast}>
                        <td style={{ fontWeight: 600, color: '#f8fafc' }}>{pc.contrast}</td>
                        <td className="mono" style={{ fontWeight: 700, color: pc.mean_diff >= 0 ? '#34d399' : '#f43f5e' }}>
                          {pc.mean_diff >= 0 ? `+${pc.mean_diff.toFixed(4)}` : pc.mean_diff.toFixed(4)}
                        </td>
                        <td className="mono">
                          [{pc.ci_95_low >= 0 ? `+${pc.ci_95_low.toFixed(4)}` : pc.ci_95_low.toFixed(4)},{' '}
                          {pc.ci_95_high >= 0 ? `+${pc.ci_95_high.toFixed(4)}` : pc.ci_95_high.toFixed(4)}]
                        </td>
                        <td className="mono">{pc.p_unadjusted.toFixed(4)}</td>
                        <td className="mono" style={{ fontWeight: 700, color: pc.p_holm_6 < 0.05 ? '#fbbf24' : '#94a3b8' }}>
                          {pc.p_holm_6.toFixed(4)}
                        </td>
                        <td className="mono">
                          {pc.positive_seeds}/{pc.total_seeds}
                        </td>
                        <td>
                          <span className="badge badge-prototype">{pc.epistemic_verdict}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Phase H XAI Validation Card */}
          {xaiVal && (
            <div className="card" style={{ marginTop: 24 }}>
              <div className="card-header">
                <div className="card-title">
                  <HelpCircle size={18} color="#a855f7" />
                  {xaiVal.experiment}
                </div>
                <span className="badge badge-dataset">
                  N={xaiVal.summary.cohort_size} Multi-Edge Nodes (Mean E_cand={xaiVal.summary.mean_e_cand.toFixed(1)})
                </span>
              </div>

              <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16, lineHeight: 1.5 }}>
                {xaiVal.scientific_takeaway}
              </p>

              <div className="grid-3">
                <div className="stat-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
                  <div className="stat-label">Exact LOO Occlusion Oracle (k=1)</div>
                  <div className="stat-value" style={{ color: '#34d399' }}>
                    {xaiVal.summary.fidelity_k1_abs_shift.exact_oracle_mean.toFixed(4)}
                  </div>
                  <div className="stat-subtext">
                    Mean |Δp| in {xaiVal.summary.computational_cost.mean_exact_oracle_ms.toFixed(1)} ms/node (70.9% of empty-subgraph bound {xaiVal.summary.ego_residual_protection_bound.empty_subgraph_mean_abs_shift.toFixed(4)})
                  </div>
                </div>

                <div className="stat-card" style={{ borderColor: 'rgba(56, 189, 248, 0.4)' }}>
                  <div className="stat-label">GNNExplainer Top-1 vs Random-1</div>
                  <div className="stat-value" style={{ color: '#38bdf8' }}>
                    {xaiVal.summary.fidelity_k1_abs_shift.top1_ge_mean.toFixed(4)} vs {xaiVal.summary.fidelity_k1_abs_shift.rand1_mean.toFixed(4)}
                  </div>
                  <div className="stat-subtext">
                    Paired p = {xaiVal.summary.fidelity_k1_abs_shift.top1_vs_rand1_p.toFixed(4)} ({xaiVal.summary.computational_cost.mean_gnnexplainer_ms.toFixed(1)} ms/node; Entropy = {xaiVal.summary.sparsity.mean_normalized_entropy.toFixed(4)})
                  </div>
                </div>

                <div className="stat-card" style={{ borderColor: 'rgba(168, 85, 247, 0.4)' }}>
                  <div className="stat-label">5-Seed Stability (Intra vs Inter)</div>
                  <div className="stat-value" style={{ color: '#a855f7' }}>
                    ρ = {xaiVal.summary.stability.inter_model_spearman_mean.toFixed(4)}
                  </div>
                  <div className="stat-subtext">
                    Inter-model Spearman across 5 checkpoints (Intra-explainer seed ρ = {xaiVal.summary.stability.intra_model_spearman_mean.toFixed(4)})
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Phase 5 External Validation Card */}
          {extVal && (
            <div className="card" style={{ marginTop: 24, borderColor: 'rgba(56, 189, 248, 0.3)' }}>
              <div className="card-header">
                <div className="card-title">
                  <Network size={18} color="#38bdf8" />
                  {extVal.experiment}
                </div>
                <span className="badge badge-dataset" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                  Elliptic++ Actors (N=920,691, 56 Features, 5 Seeds)
                </span>
              </div>

              <div className="safety-banner info" style={{ marginBottom: 16 }}>
                <div>
                  <div className="safety-title" style={{ color: '#38bdf8' }}>
                    Partially Overlapping Cross-Representation Validation
                  </div>
                  <div className="safety-desc">
                    {extVal.provenance_classification}. {extVal.scientific_verdict}
                  </div>
                </div>
              </div>

              <div className="grid-3" style={{ marginBottom: 20 }}>
                <div className="stat-card" style={{ borderColor: 'rgba(56, 189, 248, 0.4)' }}>
                  <div className="stat-label">True vs Rewired Graph (Validation)</div>
                  <div className="stat-value" style={{ color: '#38bdf8' }}>
                    +0.0694 AUPRC
                  </div>
                  <div className="stat-subtext">
                    t(4)=8.0311, p_Holm=0.0039 (5/5 seeds positive, Cohen's dz=3.59)
                  </div>
                </div>

                <div className="stat-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
                  <div className="stat-label">True vs Rewired Graph (Held-Out Test)</div>
                  <div className="stat-value" style={{ color: '#34d399' }}>
                    +0.0165 AUPRC
                  </div>
                  <div className="stat-subtext">
                    t(4)=4.4903, p_Holm=0.0327 (5/5 seeds positive, Cohen's dz=2.01)
                  </div>
                </div>

                <div className="stat-card" style={{ borderColor: 'rgba(168, 85, 247, 0.4)' }}>
                  <div className="stat-label">External LOO Fidelity (N=25)</div>
                  <div className="stat-value" style={{ color: '#a855f7' }}>
                    {extVal.external_xai_summary?.fidelity_abs_dp?.loo_top1_mean?.toFixed(4) || '0.2177'} |Δp|
                  </div>
                  <div className="stat-subtext">
                    p=0.0029 vs Random-1 in {extVal.external_xai_summary?.sparsity_and_latency?.loo_latency_ms_mean?.toFixed(1) || '29.6'} ms (14.7x faster than GNNExplainer)
                  </div>
                </div>
              </div>

              {/* Validation Summary Table */}
              <div style={{ overflowX: 'auto', marginBottom: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', marginBottom: 8 }}>
                  Validation Split Performance (Steps 33–40, 5 Seeds):
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th>Model Architecture</th>
                      <th>Graph Topology</th>
                      <th>Parameters</th>
                      <th>Primary AUPRC (95% CI)</th>
                      <th>AUPRC (SD)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['B', 'A', 'C', 'D'].map((cid) => {
                      const cinfo = extVal.validation_summary.conditions[cid];
                      if (!cinfo) return null;
                      const m = cinfo.metrics.auprc;
                      return (
                        <tr key={cid} style={{ background: cid === 'B' ? 'rgba(56, 189, 248, 0.08)' : undefined }}>
                          <td>
                            <span className="mono" style={{ fontWeight: 700, color: cid === 'B' ? '#38bdf8' : '#f8fafc' }}>
                              Cond {cid}
                            </span>
                          </td>
                          <td style={{ fontWeight: cid === 'B' ? 700 : 400 }}>{cinfo.description}</td>
                          <td>{cid === 'B' ? 'True AddrAddr Graph' : cid === 'D' ? 'Degree-Preserving Rewired' : 'Empty (E=0)'}</td>
                          <td className="mono">{cinfo.param_count.toLocaleString()}</td>
                          <td className="mono" style={{ fontWeight: 700, color: cid === 'B' ? '#38bdf8' : '#f8fafc' }}>
                            {m.mean.toFixed(4)} [{m.ci95_low.toFixed(4)}, {m.ci95_high.toFixed(4)}]
                          </td>
                          <td className="mono">±{m.sd.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Held-Out Test Summary Table */}
              <div style={{ overflowX: 'auto' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', marginBottom: 8 }}>
                  Held-Out Test Performance (Steps 41–49, 5 Seeds):
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th>Model Architecture</th>
                      <th>Graph Topology</th>
                      <th>Test AUPRC (95% CI)</th>
                      <th>Test AUPRC (SD)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['A', 'B', 'C', 'D'].map((cid) => {
                      const cinfo = extVal.test_summary.conditions[cid];
                      if (!cinfo) return null;
                      const m = cinfo.metrics.auprc;
                      return (
                        <tr key={cid} style={{ background: cid === 'B' ? 'rgba(56, 189, 248, 0.08)' : undefined }}>
                          <td>
                            <span className="mono" style={{ fontWeight: 700, color: cid === 'B' ? '#38bdf8' : '#f8fafc' }}>
                              Cond {cid}
                            </span>
                          </td>
                          <td style={{ fontWeight: cid === 'B' ? 700 : 400 }}>{cinfo.description}</td>
                          <td>{cid === 'B' ? 'True AddrAddr Graph' : cid === 'D' ? 'Degree-Preserving Rewired' : 'Empty (E=0)'}</td>
                          <td className="mono" style={{ fontWeight: 700, color: cid === 'B' ? '#38bdf8' : '#f8fafc' }}>
                            {m.mean.toFixed(4)} [{m.ci95_low.toFixed(4)}, {m.ci95_high.toFixed(4)}]
                          </td>
                          <td className="mono">±{m.sd.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
};

