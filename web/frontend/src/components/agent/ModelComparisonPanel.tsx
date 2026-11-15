import type React from 'react';
import { GitCompare, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import type { ModelPredictionsSummary } from '../../types/agent';

interface Props {
  predictions?: ModelPredictionsSummary;
  overallProb?: number;
  overallDisagreement?: boolean;
  overallSpread?: number;
}

export const ModelComparisonPanel: React.FC<Props> = ({
  predictions,
  overallProb = 0.0,
  overallDisagreement = false,
  overallSpread = 0.0,
}) => {
  const etgnnProb = predictions?.canonical_etgnn_prob ?? overallProb;
  const gatProb = predictions?.matched_gat_prob ?? null;
  const rfProb = predictions?.random_forest_prob ?? null;
  const xgbProb = predictions?.xgboost_prob ?? null;

  const validProbs = [etgnnProb, gatProb, rfProb, xgbProb].filter((p): p is number => p !== null);
  const spread = validProbs.length >= 2 ? Math.max(...validProbs) - Math.min(...validProbs) : overallSpread;
  const isDivergent = spread >= 0.30 || overallDisagreement;

  const models = [
    {
      id: 'etgnn',
      name: 'ETGNN (Canonical)',
      architecture: 'Temporal Graph Neural Network (2-hop message passing)',
      prob: etgnnProb,
      threshold: 0.50,
      evidenceId: 'EV-PRED-ETGNN',
      notes: 'Integrates local features and temporal graph neighborhood structure.',
    },
    {
      id: 'matched_gat',
      name: 'Matched GAT',
      architecture: 'Static Graph Attention Network (2 heads, 2 layers)',
      prob: gatProb,
      threshold: 0.50,
      evidenceId: 'EV-PRED-GAT',
      notes: 'Evaluates 2-hop graph attention without temporal positional encoding.',
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      architecture: 'Ensemble Decision Trees (100 estimators)',
      prob: rfProb,
      threshold: 0.50,
      evidenceId: 'EV-PRED-RF',
      notes: 'Evaluates 165 local and aggregate features independently of edge topology.',
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      architecture: 'Gradient Boosted Decision Trees',
      prob: xgbProb,
      threshold: 0.50,
      evidenceId: 'EV-PRED-XGB',
      notes: 'Evaluates tabular gradient boosting without structural graph propagation.',
    },
  ];

  return (
    <div className="model-comparison-panel card" style={{ padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <GitCompare size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Multi-Model Consensus & Probability Spread
          </h3>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            borderRadius: 'var(--radius-sm, 6px)',
            fontSize: 12,
            fontWeight: 600,
            backgroundColor: isDivergent ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)',
            border: `1px solid ${isDivergent ? 'rgba(245, 158, 11, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
            color: isDivergent ? '#f59e0b' : '#10b981',
          }}
        >
          {isDivergent ? <AlertTriangle size={13} /> : <CheckCircle size={13} />}
          <span>Spread: {spread.toFixed(4)} ({isDivergent ? 'Divergent' : 'Aligned'})</span>
        </div>
      </div>

      {isDivergent && (
        <div
          style={{
            padding: '10px 14px',
            borderRadius: 'var(--radius-sm, 6px)',
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            marginBottom: 14,
            fontSize: 12,
            lineHeight: 1.5,
            color: 'var(--text-secondary, #94a3b8)',
          }}
        >
          <strong style={{ color: '#f59e0b' }}>Model Divergence Detected (Spread &ge; 0.30):</strong>{' '}
          ETGNN incorporates temporal message passing across the transaction's graph neighborhood, dampening local tabular anomalies that cause tree models to classify with differing risk scores.
        </div>
      )}

      {/* Model Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 12 }}>
        {models.map((m) => {
          const isElevated = m.prob !== null && m.prob >= m.threshold;
          return (
            <div
              key={m.id}
              style={{
                padding: '12px 14px',
                borderRadius: 'var(--radius-md, 10px)',
                backgroundColor: 'rgba(15, 23, 42, 0.35)',
                border: '1px solid var(--border-subtle, #182234)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary, #f8fafc)' }}>
                    {m.name}
                  </div>
                  {m.prob !== null ? (
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        padding: '2px 6px',
                        borderRadius: 4,
                        backgroundColor: isElevated ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                        color: isElevated ? '#f43f5e' : '#10b981',
                        border: `1px solid ${isElevated ? 'rgba(244, 63, 94, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                      }}
                    >
                      {isElevated ? 'Elevated Risk' : 'Licit Risk'}
                    </span>
                  ) : (
                    <span style={{ fontSize: 10, color: 'var(--text-muted, #64748b)' }}>Unavailable</span>
                  )}
                </div>

                <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', marginTop: 2, marginBottom: 10 }}>
                  {m.architecture}
                </div>

                <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 6 }}>
                  <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--font-mono, monospace)', color: isElevated ? '#f43f5e' : '#38bdf8' }}>
                    {m.prob !== null ? m.prob.toFixed(4) : 'N/A'}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
                    / threshold {m.threshold.toFixed(2)}
                  </div>
                </div>

                {/* Probability Bar */}
                <div style={{ height: 6, borderRadius: 3, backgroundColor: 'rgba(255, 255, 255, 0.1)', overflow: 'hidden', marginBottom: 8 }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${m.prob !== null ? Math.min(Math.max(m.prob * 100, 0), 100) : 0}%`,
                      backgroundColor: isElevated ? '#f43f5e' : '#38bdf8',
                    }}
                  />
                </div>
              </div>

              <div style={{ fontSize: 10.5, color: 'var(--text-secondary, #94a3b8)', lineHeight: 1.4, borderTop: '1px solid var(--border-subtle, #182234)', paddingTop: 6, marginTop: 4 }}>
                {m.notes}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
        <Info size={13} />
        <span>Epistemic Disclosure: Output probabilities reflect raw uncalibrated classifier outputs, not Bayesian posterior confidence scores.</span>
      </div>
    </div>
  );
};

