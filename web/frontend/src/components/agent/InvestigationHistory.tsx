import type React from 'react';
import { useState, useEffect } from 'react';
import { History, RefreshCw, ChevronRight } from 'lucide-react';
import type { InvestigationHistoryItem } from '../../types/agent';
import { fetchInvestigationList } from '../../services/agentApi';

interface Props {
  onSelectInvestigation: (investigationId: string) => void;
  activeInvestigationId?: string;
}

export const InvestigationHistory: React.FC<Props> = ({
  onSelectInvestigation,
  activeInvestigationId,
}) => {
  const [items, setItems] = useState<InvestigationHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const list = await fetchInvestigationList();
      setItems(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  return (
    <div className="investigation-history card" style={{ padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <History size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Session Investigation History ({items.length})
          </h3>
        </div>

        <button
          className="btn btn-secondary"
          onClick={loadHistory}
          disabled={loading}
          style={{ fontSize: 11, padding: '4px 8px', display: 'flex', alignItems: 'center', gap: 4 }}
        >
          <RefreshCw size={12} className={loading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', marginBottom: 10 }}>
        Session-limited in-memory cache (maximum 100 entries, thread-safe with LRU eviction).
      </div>

      {items.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-muted, #64748b)', fontStyle: 'italic', padding: '12px 0' }}>
          No investigations recorded in this session yet. Launch an investigation to see history.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
          {items.map((item) => {
            const isActive = activeInvestigationId === item.investigation_id;
            return (
              <div
                key={item.investigation_id}
                onClick={() => onSelectInvestigation(item.investigation_id)}
                style={{
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-sm, 6px)',
                  backgroundColor: isActive ? 'rgba(56, 189, 248, 0.12)' : 'rgba(15, 23, 42, 0.3)',
                  border: `1px solid ${isActive ? 'rgba(56, 189, 248, 0.4)' : 'var(--border-subtle, #182234)'}`,
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'all 0.15s ease',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 12.5, fontWeight: 700, color: '#f8fafc', fontFamily: 'var(--font-mono, monospace)' }}>
                      Tx {item.transaction_id}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        padding: '1px 6px',
                        borderRadius: 4,
                        backgroundColor: item.status === 'COMPLETED' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                        color: item.status === 'COMPLETED' ? '#10b981' : '#f59e0b',
                      }}
                    >
                      {item.status}
                    </span>
                    {item.model_disagreement && (
                      <span style={{ fontSize: 10, color: '#f59e0b', fontWeight: 600 }}>
                        (Spread: {item.probability_spread?.toFixed(2)})
                      </span>
                    )}
                  </div>

                  <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)', marginTop: 3 }}>
                    ID: {item.investigation_id} • {item.provider_type} ({item.generation_mode})
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#38bdf8', fontSize: 12 }}>
                  <span>Open</span>
                  <ChevronRight size={14} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
