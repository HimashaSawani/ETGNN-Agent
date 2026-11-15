import type React from 'react';
import { useState, useEffect } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, Eye, Shield, AlertCircle, Sparkles } from 'lucide-react';
import { fetchTransactions } from '../api';
import type { TransactionsPageResponse, TransactionSummary } from '../api';

interface Props {
  selectedTxId: number | null;
  onSelectTransaction: (txId: number, targetTab?: string) => void;
  onOpenCopilot?: (txId: number) => void;
}

export const TransactionExplorer: React.FC<Props> = ({ selectedTxId, onSelectTransaction, onOpenCopilot }) => {
  const [data, setData] = useState<TransactionsPageResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [limit] = useState<number>(20);
  const [offset, setOffset] = useState<number>(0);
  const [split, setSplit] = useState<string>('');
  const [label, setLabel] = useState<string>('');
  const [timeStep, setTimeStep] = useState<string>('');
  const [search, setSearch] = useState<string>('');

  const loadPage = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchTransactions({
        limit,
        offset,
        split: split || undefined,
        label: label !== '' ? parseInt(label) : undefined,
        time_step: timeStep !== '' ? parseInt(timeStep) : undefined,
        search: search.trim() || undefined,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load transactions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPage();
  }, [offset, split, label, timeStep]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setOffset(0);
    loadPage();
  };

  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = data ? Math.ceil(data.total / limit) : 1;

  return (
    <div className="explorer-container">
      {/* Filter and Search Bar */}
      <div className="card" style={{ marginBottom: 20 }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 260 }}>
            <Search size={16} color="#64748b" />
            <input
              type="text"
              placeholder="Search by Bitcoin Transaction ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="search-input"
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <Filter size={16} color="#64748b" />
            <select
              value={split}
              onChange={(e) => { setSplit(e.target.value); setOffset(0); }}
              className="select-input"
            >
              <option value="">All Chronological Splits</option>
              <option value="train">Train (Steps 1–32)</option>
              <option value="validation">Validation (Steps 33–40)</option>
              <option value="test">Test (Steps 41–49)</option>
              <option value="unlabelled">Unlabelled</option>
            </select>

            <select
              value={label}
              onChange={(e) => { setLabel(e.target.value); setOffset(0); }}
              className="select-input"
            >
              <option value="">All Ground Truth Labels</option>
              <option value="1">Illicit (Class 1)</option>
              <option value="0">Licit (Class 0)</option>
              <option value="-1">Unknown (-1)</option>
            </select>

            <input
              type="number"
              placeholder="Step (1–49)"
              min={1}
              max={49}
              value={timeStep}
              onChange={(e) => { setTimeStep(e.target.value); setOffset(0); }}
              className="search-input"
              style={{ width: 110 }}
            />

            <button type="submit" className="btn btn-primary">
              Filter
            </button>
            {(search || split || label || timeStep) && (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setSearch('');
                  setSplit('');
                  setLabel('');
                  setTimeStep('');
                  setOffset(0);
                }}
              >
                Reset
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Main Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Shield size={18} color="#38bdf8" />
            Elliptic Transaction Register ({data ? data.total.toLocaleString() : '...'} total)
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8' }}>
            Page {currentPage} of {totalPages}
          </div>
        </div>

        {error && (
          <div className="safety-banner danger" style={{ margin: '0 0 16px 0' }}>
            <AlertCircle size={18} color="#f43f5e" />
            <div className="safety-desc">{error}</div>
          </div>
        )}

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Tx ID</th>
                <th>Time Step</th>
                <th>Split</th>
                <th>Actual Label</th>
                <th>ETGNN Prob</th>
                <th>Matched GAT Prob</th>
                <th>Artifacts</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', padding: '32px 0', color: '#64748b' }}>
                    Loading transactions...
                  </td>
                </tr>
              ) : data && data.items.length > 0 ? (
                data.items.map((tx: TransactionSummary) => {
                  const isSelected = selectedTxId === tx.transaction_id;
                  return (
                    <tr key={tx.transaction_id} className={isSelected ? 'active' : ''}>
                      <td className="mono" style={{ fontWeight: 600, color: '#f8fafc' }}>
                        {tx.transaction_id}
                      </td>
                      <td>
                        <span className="badge" style={{ background: '#1e293b', color: '#94a3b8' }}>
                          t = {tx.time_step}
                        </span>
                      </td>
                      <td>
                        <span
                          className="badge"
                          style={{
                            background:
                              tx.split === 'test'
                                ? 'rgba(56, 189, 248, 0.15)'
                                : tx.split === 'validation'
                                ? 'rgba(168, 85, 247, 0.15)'
                                : '#1e293b',
                            color:
                              tx.split === 'test'
                                ? '#38bdf8'
                                : tx.split === 'validation'
                                ? '#c084fc'
                                : '#94a3b8',
                          }}
                        >
                          {tx.split}
                        </span>
                      </td>
                      <td>
                        {tx.actual_label === 1 ? (
                          <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)', border: '1px solid var(--illicit-border)' }}>
                            Illicit (1)
                          </span>
                        ) : tx.actual_label === 0 ? (
                          <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)', border: '1px solid var(--licit-border)' }}>
                            Licit (0)
                          </span>
                        ) : (
                          <span className="badge" style={{ background: 'var(--unknown-bg)', color: 'var(--unknown-gray)' }}>
                            Unlabelled
                          </span>
                        )}
                      </td>
                      <td className="mono">
                        <span style={{ color: tx.etgnn_prob >= 0.5 ? '#f43f5e' : '#10b981', fontWeight: 600 }}>
                          {(tx.etgnn_prob * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="mono">
                        <span style={{ color: tx.matched_gat_prob >= 0.5 ? '#f43f5e' : '#10b981', fontWeight: 600 }}>
                          {(tx.matched_gat_prob * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td>
                        {tx.has_frozen_explanation ? (
                          <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc' }}>
                            Exemplar
                          </span>
                        ) : tx.has_intervention_record ? (
                          <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                            C1/C2 Row
                          </span>
                        ) : (
                          <span style={{ color: '#475569', fontSize: 12 }}>—</span>
                        )}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', gap: 6 }}>
                          <button
                            className="btn"
                            style={{ padding: '4px 10px', fontSize: 11 }}
                            onClick={() => onSelectTransaction(tx.transaction_id, 'prediction')}
                            title="Inspect multi-model predictions"
                          >
                            <Eye size={12} /> Predict
                          </button>
                          <button
                            className="btn"
                            style={{ padding: '4px 10px', fontSize: 11 }}
                            onClick={() => onSelectTransaction(tx.transaction_id, 'graph')}
                            title="View neighborhood graph"
                          >
                            Graph
                          </button>
                          <button
                            className="btn"
                            style={{ padding: '4px 10px', fontSize: 11 }}
                            onClick={() => onSelectTransaction(tx.transaction_id, 'explanation')}
                            title="Inspect GNNExplainer analysis"
                          >
                            Explain
                          </button>
                          {onOpenCopilot && (
                            <button
                              className="btn btn-primary"
                              style={{ padding: '4px 10px', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 4, backgroundColor: '#2563eb', color: '#fff' }}
                              onClick={() => onOpenCopilot(tx.transaction_id)}
                              title="Investigate with AI Copilot"
                            >
                              <Sparkles size={11} /> AI
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', padding: '32px 0', color: '#64748b' }}>
                    No transactions match current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 18 }}>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Showing {data ? offset + 1 : 0}–{data ? Math.min(offset + limit, data.total) : 0} of {data ? data.total.toLocaleString() : 0}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn"
              disabled={offset === 0 || loading}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              <ChevronLeft size={14} /> Previous
            </button>
            <button
              className="btn"
              disabled={!data || offset + limit >= data.total || loading}
              onClick={() => setOffset(offset + limit)}
            >
              Next <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
