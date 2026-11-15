import type React from 'react';
import { useState, useEffect } from 'react';
import { BarChart3, Award, AlertCircle } from 'lucide-react';
import { fetchMetrics } from '../api';
import type { MetricsResponse, MetricItem } from '../api';

export const BenchmarkComparison: React.FC = () => {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof MetricItem>('auprc');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  useEffect(() => {
    fetchMetrics()
      .then((res) => setData(res))
      .catch((err) => setError(err.message || 'Failed to load metrics'))
      .finally(() => setLoading(false));
  }, []);

  const handleSort = (key: keyof MetricItem) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sortedMetrics = data
    ? [...data.metrics].sort((a, b) => {
        const valA = a[sortKey];
        const valB = b[sortKey];
        if (typeof valA === 'number' && typeof valB === 'number') {
          return sortAsc ? valA - valB : valB - valA;
        }
        return 0;
      })
    : [];

  const maxAuprc = 0.70;

  return (
    <div className="benchmark-comparison">
      {/* Overview Stat Cards */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <div className="stat-card" style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}>
          <div className="stat-label">Top Performer (Held-Out Test)</div>
          <div className="stat-value" style={{ color: '#fbbf24' }}>Random Forest</div>
          <div className="stat-subtext">Average Precision: 0.6323 | Precision: 0.6966 | F1: 0.6319</div>
        </div>

        <div className="stat-card" style={{ borderColor: 'rgba(56, 189, 248, 0.4)' }}>
          <div className="stat-label">ETGNN (Static Time-GAT)</div>
          <div className="stat-value" style={{ color: '#38bdf8' }}>0.2250 AP</div>
          <div className="stat-subtext">Held-out test steps 41–49 (-0.4073 vs Random Forest)</div>
        </div>

        <div className="stat-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
          <div className="stat-label">Edge Removal Gain</div>
          <div className="stat-value" style={{ color: '#34d399' }}>0.4812 AP</div>
          <div className="stat-subtext">ETGNN Local No Edges (+0.2468 AP over full graph)</div>
        </div>
      </div>

      {/* Visual Bar Chart */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title">
            <BarChart3 size={18} color="#38bdf8" />
            Held-Out Test Performance: Average Precision (AP)
          </div>
          <span className="badge badge-dataset">Chronological Steps 41–49 (N=9,973)</span>
        </div>

        {error && (
          <div className="safety-banner danger">
            <AlertCircle size={18} color="#f43f5e" />
            <div className="safety-desc">{error}</div>
          </div>
        )}

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: '#64748b' }}>
            Loading benchmark metrics...
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {data?.metrics.map((item) => {
              const isRF = item.model.includes('Random Forest');
              const isETGNN = item.model === 'ETGNN';
              const isNoEdges = item.model.includes('no edges');
              const barColor = isRF
                ? '#fbbf24'
                : isETGNN
                ? '#38bdf8'
                : isNoEdges
                ? '#34d399'
                : '#64748b';

              return (
                <div key={item.model} style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  <div style={{ width: 180, fontSize: 13, fontWeight: isRF || isETGNN ? 700 : 500, color: isRF || isETGNN ? '#f8fafc' : '#94a3b8' }}>
                    {item.model}
                  </div>
                  <div style={{ flex: 1, background: '#1e293b', height: 18, borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${(item.auprc / maxAuprc) * 100}%`,
                        height: '100%',
                        background: barColor,
                        borderRadius: 4,
                        transition: 'width 0.4s ease',
                      }}
                    />
                  </div>
                  <div className="mono" style={{ width: 70, textAlign: 'right', fontSize: 13, fontWeight: 700, color: barColor }}>
                    {item.auprc.toFixed(4)}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Complete Metrics Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Award size={18} color="#a855f7" />
            Official Evaluated Models Metrics Table
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>
            Click column headers to sort by metric
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('model')} style={{ cursor: 'pointer' }}>
                  Model Name {sortKey === 'model' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('auprc')} style={{ cursor: 'pointer', color: '#38bdf8' }}>
                  AUPRC {sortKey === 'auprc' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('roc_auc')} style={{ cursor: 'pointer' }}>
                  ROC-AUC {sortKey === 'roc_auc' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('f1')} style={{ cursor: 'pointer' }}>
                  F1 Score {sortKey === 'f1' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('precision')} style={{ cursor: 'pointer' }}>
                  Precision {sortKey === 'precision' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('recall')} style={{ cursor: 'pointer' }}>
                  Recall {sortKey === 'recall' && (sortAsc ? '▲' : '▼')}
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedMetrics.map((row) => {
                const isWinner = row.model === 'Random Forest';
                const isETGNN = row.model === 'ETGNN';
                return (
                  <tr key={row.model} className={isWinner ? 'active' : ''}>
                    <td style={{ fontWeight: isWinner || isETGNN ? 700 : 500, color: isWinner ? '#fbbf24' : isETGNN ? '#38bdf8' : '#f8fafc' }}>
                      {row.model} {isWinner && '★ (Top)'}
                    </td>
                    <td className="mono" style={{ fontWeight: 700, color: '#38bdf8' }}>
                      {row.auprc.toFixed(4)}
                    </td>
                    <td className="mono">{row.roc_auc.toFixed(4)}</td>
                    <td className="mono">{row.f1.toFixed(4)}</td>
                    <td className="mono">{row.precision.toFixed(4)}</td>
                    <td className="mono">{row.recall.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 16, fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
          <div>
            <strong>Evaluation Split Distinction:</strong> Artifact source <code>results/tables/final_test_results.csv</code> reports{' '}
            <strong>Historical Chronological Test Split (Steps 41–49, N=9,973 labelled nodes: 524 illicit [5.25%], 9,449 licit)</strong>{' '}
            trained on Steps 1–32 and selected on Validation Steps 33–40 (validation illicit prevalence = 10.47%).
          </div>
          <div style={{ marginTop: 4 }}>
            <strong>Phase 4 &amp; Phase 5 Status:</strong> Controlled 40-trial Residual ETGNN vs. Deep MLP experiments (see <em>Ablation Analysis</em> tab)
            are reported strictly on the Chronological Validation Split (Steps 33–40). External cross-representation evaluation on{' '}
            <code>Elliptic++ Actors</code> (56 wallet features) has passed Stage 1/2 synthetic &amp; preflight verification and is gated on explicit external dataset acquisition approval.
          </div>
        </div>
      </div>
    </div>
  );
};
