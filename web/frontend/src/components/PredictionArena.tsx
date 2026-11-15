import type React from 'react';
import { useState, useEffect } from 'react';
import { Cpu, RefreshCw, Sliders, AlertCircle, Sparkles } from 'lucide-react';
import { fetchTransaction, predictTransaction } from '../api';
import type { TransactionDetail } from '../api';

interface Props {
  transactionId: number;
  onChangeTransactionId: (txId: number) => void;
  onOpenCopilot?: (txId: number) => void;
}

export const PredictionArena: React.FC<Props> = ({ transactionId, onChangeTransactionId, onOpenCopilot }) => {
  const [detail, setDetail] = useState<TransactionDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Live overrides state
  const [feat0Override, setFeat0Override] = useState<number | null>(null);
  const [feat1Override, setFeat1Override] = useState<number | null>(null);
  const [livePredictions, setLivePredictions] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);

  // Load transaction base details
  const loadDetail = async (id: number) => {
    setLoading(true);
    setError(null);
    setLivePredictions(null);
    setFeat0Override(null);
    setFeat1Override(null);
    try {
      const res = await fetchTransaction(id);
      setDetail(res);
      setLivePredictions(res.predictions);
    } catch (err: any) {
      setError(err.message || `Failed to load transaction ${id}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (transactionId) {
      loadDetail(transactionId);
    }
  }, [transactionId]);

  // Execute forward pass with feature overrides
  const handleRunOverride = async () => {
    if (!detail) return;
    setIsPredicting(true);
    try {
      const overrides: Record<string, number> = {};
      if (feat0Override !== null) overrides['feat_0'] = feat0Override;
      if (feat1Override !== null) overrides['feat_1'] = feat1Override;

      const res = await predictTransaction(detail.transaction_id, overrides);
      setLivePredictions(res.predictions);
    } catch (err: any) {
      setError(err.message || 'Live forward pass failed');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleResetOverrides = () => {
    if (!detail) return;
    setFeat0Override(null);
    setFeat1Override(null);
    setLivePredictions(detail.predictions);
  };

  return (
    <div className="prediction-arena">
      {/* Transaction Control Header */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Cpu size={22} color="#38bdf8" />
            <div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>
                Model Predictions
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Live forward-pass inference across ETGNN, Residual ETGNN, Matched GAT, Random Forest, and XGBoost
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 13, color: '#94a3b8' }}>Inspect Tx ID:</span>
            <input
              type="number"
              value={transactionId || ''}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) onChangeTransactionId(val);
              }}
              className="search-input mono"
              style={{ width: 150 }}
            />
            <button className="btn" onClick={() => loadDetail(transactionId)}>
              <RefreshCw size={14} /> Refresh
            </button>
            {onOpenCopilot && (
              <button
                className="btn btn-primary"
                onClick={() => onOpenCopilot(transactionId)}
                style={{ display: 'flex', alignItems: 'center', gap: 6, backgroundColor: '#2563eb', color: '#fff' }}
                title="Launch AI Investigation Copilot"
              >
                <Sparkles size={14} /> Investigate
              </button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="safety-banner danger">
          <AlertCircle size={20} color="#f43f5e" />
          <div className="safety-desc">{error}</div>
        </div>
      )}

      {loading && !detail ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 0', color: '#64748b' }}>
          Loading transaction and executing forward pass...
        </div>
      ) : detail ? (
        <>
          {/* Metadata Summary Banner */}
          <div className="grid-4" style={{ marginBottom: 24 }}>
            <div className="stat-card">
              <div className="stat-label">Ground Truth Label</div>
              <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {detail.actual_label === 1 ? (
                  <span style={{ color: '#f43f5e' }}>Illicit (1)</span>
                ) : detail.actual_label === 0 ? (
                  <span style={{ color: '#10b981' }}>Licit (0)</span>
                ) : (
                  <span style={{ color: '#94a3b8' }}>Unknown (-1)</span>
                )}
              </div>
              <div className="stat-subtext">Elliptic historical ground truth label</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Time Step & Split</div>
              <div className="stat-value">t = {detail.time_step}</div>
              <div className="stat-subtext">Split: <span style={{ textTransform: 'capitalize', color: '#38bdf8' }}>{detail.split}</span></div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Local Graph Topology</div>
              <div className="stat-value">{detail.graph_degree.total_degree} <span style={{ fontSize: 14, color: '#64748b' }}>edges</span></div>
              <div className="stat-subtext">In-deg: {detail.graph_degree.in_degree} | Out-deg: {detail.graph_degree.out_degree}</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Features Vector</div>
              <div className="stat-value">165 <span style={{ fontSize: 14, color: '#64748b' }}>dims</span></div>
              <div className="stat-subtext">93 local + 72 1-hop aggregate features</div>
            </div>
          </div>

          {/* Model Prediction Cards Grid */}
          <div className="grid-5" style={{ marginBottom: 24 }}>
            {/* ETGNN */}
            {livePredictions && livePredictions.etgnn && (
              <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <div>
                    <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', marginBottom: 6 }}>
                      Research GNN
                    </span>
                    <h3 style={{ fontSize: 15, color: '#f8fafc' }}>ETGNN</h3>
                  </div>
                  {livePredictions.etgnn.predicted_class === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                      Pred: Illicit
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                      Pred: Licit
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: livePredictions.etgnn.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                  {(livePredictions.etgnn.illicit_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Illicit Probability (Threshold = 0.5)</div>

                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${livePredictions.etgnn.illicit_probability * 100}%`,
                      background: livePredictions.etgnn.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981',
                    }}
                  />
                </div>

                <div style={{ marginTop: 14, fontSize: 11, color: '#64748b', lineHeight: 1.45 }}>
                  Static 2-layer GAT with sinusoidal time encoding. Test AUPRC: 0.2250.
                </div>
              </div>
            )}

            {/* Matched GAT */}
            {livePredictions && livePredictions.matched_gat && (
              <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <div>
                    <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc', marginBottom: 6 }}>
                      Ablated GNN
                    </span>
                    <h3 style={{ fontSize: 15, color: '#f8fafc' }}>Matched GAT</h3>
                  </div>
                  {livePredictions.matched_gat.predicted_class === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                      Pred: Illicit
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                      Pred: Licit
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: livePredictions.matched_gat.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                  {(livePredictions.matched_gat.illicit_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Illicit Probability (Threshold = 0.5)</div>

                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${livePredictions.matched_gat.illicit_probability * 100}%`,
                      background: livePredictions.matched_gat.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981',
                    }}
                  />
                </div>

                <div style={{ marginTop: 14, fontSize: 11, color: '#64748b', lineHeight: 1.45 }}>
                  Identical GAT architecture with zeroed time vectors. Test AUPRC: 0.2099.
                </div>
              </div>
            )}

            {/* Residual ETGNN (93f Local) */}
            {livePredictions && livePredictions.residual_etgnn && (
              <div className="card" style={{ background: '#0f172a', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <div>
                    <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', marginBottom: 6 }}>
                      Ego-Residual GNN
                    </span>
                    <h3 style={{ fontSize: 15, color: '#f8fafc' }}>Residual ETGNN</h3>
                  </div>
                  {livePredictions.residual_etgnn.predicted_class === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                      Pred: Illicit
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                      Pred: Licit
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: livePredictions.residual_etgnn.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                  {(livePredictions.residual_etgnn.illicit_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Illicit Probability (Threshold = 0.5)</div>

                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${livePredictions.residual_etgnn.illicit_probability * 100}%`,
                      background: livePredictions.residual_etgnn.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981',
                    }}
                  />
                </div>

                <div style={{ marginTop: 14, fontSize: 11, color: '#34d399', lineHeight: 1.45 }}>
                  Ego-skip projections preserve local features. Held-out Test AP: 0.4223 (+0.1973 vs ETGNN).
                </div>
              </div>
            )}

            {/* Random Forest */}
            {livePredictions && livePredictions.random_forest && (
              <div className="card" style={{ background: '#0f172a', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <div>
                    <span className="badge" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24', marginBottom: 6 }}>
                      Top Baseline
                    </span>
                    <h3 style={{ fontSize: 15, color: '#f8fafc' }}>Random Forest</h3>
                  </div>
                  {livePredictions.random_forest.predicted_class === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                      Pred: Illicit
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                      Pred: Licit
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: livePredictions.random_forest.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                  {(livePredictions.random_forest.illicit_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Illicit Probability (Threshold = 0.5)</div>

                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${livePredictions.random_forest.illicit_probability * 100}%`,
                      background: livePredictions.random_forest.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981',
                    }}
                  />
                </div>

                <div style={{ marginTop: 14, fontSize: 11, color: '#fbbf24', lineHeight: 1.45 }}>
                  Highest performing model on test data. Test AUPRC: 0.6323 (+0.4073 vs ETGNN).
                </div>
              </div>
            )}

            {/* XGBoost */}
            {livePredictions && livePredictions.xgboost && (
              <div className="card" style={{ background: '#0f172a', border: '1px solid #1e293b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <div>
                    <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', marginBottom: 6 }}>
                      Tabular Baseline
                    </span>
                    <h3 style={{ fontSize: 15, color: '#f8fafc' }}>XGBoost</h3>
                  </div>
                  {livePredictions.xgboost.predicted_class === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                      Pred: Illicit
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                      Pred: Licit
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 32, fontWeight: 800, fontFamily: 'var(--font-heading)', color: livePredictions.xgboost.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                  {(livePredictions.xgboost.illicit_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Illicit Probability (Threshold = 0.5)</div>

                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${livePredictions.xgboost.illicit_probability * 100}%`,
                      background: livePredictions.xgboost.illicit_probability >= 0.5 ? '#f43f5e' : '#10b981',
                    }}
                  />
                </div>

                <div style={{ marginTop: 14, fontSize: 11, color: '#64748b', lineHeight: 1.45 }}>
                  Gradient boosted trees on 165 features. Test AUPRC: 0.5472.
                </div>
              </div>
            )}
          </div>

          {/* Interactive Feature Perturbation Panel */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Sliders size={18} color="#38bdf8" />
                Live Feature Sensitivity Simulation
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                {(feat0Override !== null || feat1Override !== null) && (
                  <button className="btn" onClick={handleResetOverrides}>
                    Reset to Actual Values
                  </button>
                )}
                <button
                  className="btn btn-primary"
                  onClick={handleRunOverride}
                  disabled={isPredicting}
                >
                  {isPredicting ? 'Running GNN Forward Pass...' : 'Run Forward Pass'}
                </button>
              </div>
            </div>

            <p style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16 }}>
              Perturb standardized feature values for this node to test how the tabular models and the 2-hop GAT
              respond in real time. (Features are standard-scaled in the Elliptic pipeline).
            </p>

            <div className="grid-2">
              <div style={{ background: '#0d1322', padding: 16, borderRadius: 8, border: '1px solid #1e293b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 13, color: '#f8fafc' }}>
                    Feature 0 (Local Transaction Feature):
                  </span>
                  <span className="mono" style={{ color: '#38bdf8' }}>
                    {feat0Override !== null
                      ? feat0Override.toFixed(3)
                      : (detail.features_summary.sample_features['feat_0'] || 0).toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  min="-4"
                  max="4"
                  step="0.1"
                  value={
                    feat0Override !== null
                      ? feat0Override
                      : detail.features_summary.sample_features['feat_0'] || 0
                  }
                  onChange={(e) => setFeat0Override(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: '#38bdf8' }}
                />
              </div>

              <div style={{ background: '#0d1322', padding: 16, borderRadius: 8, border: '1px solid #1e293b' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 13, color: '#f8fafc' }}>
                    Feature 1 (Local Transaction Feature):
                  </span>
                  <span className="mono" style={{ color: '#38bdf8' }}>
                    {feat1Override !== null
                      ? feat1Override.toFixed(3)
                      : (detail.features_summary.sample_features['feat_1'] || 0).toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  min="-4"
                  max="4"
                  step="0.1"
                  value={
                    feat1Override !== null
                      ? feat1Override
                      : detail.features_summary.sample_features['feat_1'] || 0
                  }
                  onChange={(e) => setFeat1Override(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: '#38bdf8' }}
                />
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
