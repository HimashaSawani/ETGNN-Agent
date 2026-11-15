import type React from 'react';
import { useState, useEffect } from 'react';
import {
  X,
  Play,
  FileText,
  Download,
  Printer,
  Sparkles,
  Database,
  MessageSquare,
  History,
  Shield,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import type {
  ForensicInvestigationReport,
  AgentInvestigationEvent,
} from '../../types/agent';
import {
  runAgentInvestigation,
  fetchInvestigation,
  fetchInvestigationEvents,
  getInvestigationPdfUrl,
} from '../../services/agentApi';
import { InvestigationWorkflowStepper } from './InvestigationWorkflowStepper';
import { ModelComparisonPanel } from './ModelComparisonPanel';
import { InteractiveGraphEvidenceViewer } from './InteractiveGraphEvidenceViewer';
import { EvidenceExplorer } from './EvidenceExplorer';
import { InvestigationReportViewer } from './InvestigationReportViewer';
import { InvestigationChat } from './InvestigationChat';
import { AnalystFeedbackForm } from './AnalystFeedbackForm';
import { InvestigationHistory } from './InvestigationHistory';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  selectedTxId: number;
  onNavigateToGraph?: () => void;
}

export const InvestigationCopilotDrawer: React.FC<Props> = ({
  isOpen,
  onClose,
  selectedTxId,
  onNavigateToGraph,
}) => {
  const [activeTab, setActiveTab] = useState<'report' | 'evidence' | 'chat' | 'feedback' | 'history'>('report');
  const [txIdInput, setTxIdInput] = useState<number>(selectedTxId);
  const [evidenceMode, setEvidenceMode] = useState<'RETROSPECTIVE_INVESTIGATION' | 'HISTORICAL_EVALUATION'>('RETROSPECTIVE_INVESTIGATION');
  const [providerMode, setProviderMode] = useState<'mock' | 'evidence_only' | 'gemini' | 'anthropic' | 'ollama'>('mock');
  const [customQuestion, setCustomQuestion] = useState<string>('Forensic fraud risk and counterparty investigation');
  const [allowFallback, setAllowFallback] = useState<boolean>(true);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [investigationId, setInvestigationId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('NOT_STARTED');
  const [report, setReport] = useState<ForensicInvestigationReport | null>(null);
  const [events, setEvents] = useState<AgentInvestigationEvent[]>([]);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);

  // Sync with prop when drawer opens
  useEffect(() => {
    if (isOpen && selectedTxId) {
      setTxIdInput(selectedTxId);
    }
  }, [isOpen, selectedTxId]);

  // Handle keyboard escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleLaunchInvestigation = async () => {
    setLoading(true);
    setError(null);
    setStatus('RUNNING');
    try {
      const res = await runAgentInvestigation({
        transaction_id: txIdInput,
        question: customQuestion.trim() || undefined,
        evidence_mode: evidenceMode,
        llm_provider: providerMode,
        allow_evidence_only_fallback: allowFallback,
      });

      setInvestigationId(res.investigation_id);
      setStatus(res.status);
      setReport(res.report || null);

      // Fetch observable telemetry events
      if (res.investigation_id) {
        const evs = await fetchInvestigationEvents(res.investigation_id);
        setEvents(evs);
      }
    } catch (err: any) {
      setError(err.message || 'Investigation execution failed');
      setStatus('FAILED');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadInvestigationFromHistory = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchInvestigation(id);
      setInvestigationId(res.investigation_id);
      setStatus(res.status);
      setReport(res.report || null);
      if (res.report?.target_transaction_id) {
        setTxIdInput(res.report.target_transaction_id);
      }
      const evs = await fetchInvestigationEvents(id);
      setEvents(evs);
      setActiveTab('report');
    } catch (err: any) {
      setError(err.message || 'Failed to load investigation');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ETGNN_Investigation_${report.investigation_id}_Tx${report.target_transaction_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSelectCitation = (cid: string) => {
    setSelectedCitationId(cid);
    setActiveTab('evidence');
  };

  if (!isOpen) return null;

  return (
    <div
      className="investigation-drawer-backdrop"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(3px)',
        zIndex: 9999,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="investigation-drawer"
        style={{
          width: '100%',
          maxWidth: 820,
          height: '100vh',
          backgroundColor: 'var(--bg-main, #090d16)',
          borderLeft: '1px solid var(--border-color, #1e293b)',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--shadow-lg, 0 10px 25px -3px rgba(0,0,0,0.6))',
          animation: 'slideInRight 0.25s ease-out',
        }}
      >
        {/* Drawer Header */}
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid var(--border-color, #1e293b)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: 'var(--bg-header, rgba(15, 23, 42, 0.94))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 'var(--radius-sm, 6px)',
                backgroundColor: 'rgba(56, 189, 248, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Sparkles size={18} color="var(--accent-cyan, #38bdf8)" />
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary, #f8fafc)' }}>
                ETGNN AI Investigation Copilot
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>
                Official LangGraph • Evidence-Grounded Graph-RAG • Human Review Loop
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="btn btn-secondary"
            style={{ padding: 6, borderRadius: '50%' }}
            title="Close Drawer (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        {/* Configuration Bar */}
        <div
          style={{
            padding: '12px 24px',
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            borderBottom: '1px solid var(--border-subtle, #182234)',
            display: 'flex',
            flexWrap: 'wrap',
            gap: 12,
            alignItems: 'center',
          }}
        >
          {/* Tx ID Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #64748b)' }}>Tx ID:</span>
            <input
              type="number"
              value={txIdInput}
              onChange={(e) => setTxIdInput(parseInt(e.target.value) || 0)}
              style={{
                width: 120,
                padding: '4px 8px',
                borderRadius: 4,
                backgroundColor: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 12,
                fontFamily: 'var(--font-mono, monospace)',
              }}
            />
          </div>

          {/* Mode Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #64748b)' }}>Mode:</span>
            <select
              value={evidenceMode}
              onChange={(e) => setEvidenceMode(e.target.value as any)}
              style={{
                padding: '4px 8px',
                borderRadius: 4,
                backgroundColor: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 11,
              }}
            >
              <option value="RETROSPECTIVE_INVESTIGATION">Retrospective (All)</option>
              <option value="HISTORICAL_EVALUATION">Point-in-Time (Strict)</option>
            </select>
          </div>

          {/* Provider Mode Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #64748b)' }}>Provider:</span>
            <select
              value={providerMode}
              onChange={(e) => setProviderMode(e.target.value as any)}
              style={{
                padding: '4px 8px',
                borderRadius: 4,
                backgroundColor: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 11,
              }}
            >
              <option value="mock">MOCK (Deterministic Offline)</option>
              <option value="evidence_only">EVIDENCE_ONLY (Factual Non-Generative)</option>
              <option value="gemini">Google Gemini (Live API)</option>
              <option value="anthropic">Anthropic Claude (Live API)</option>
              <option value="ollama">Local Ollama (Local Daemon)</option>
            </select>
          </div>

          {/* Custom Question / Focus Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: '1 1 200px' }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted, #64748b)' }}>Focus:</span>
            <input
              type="text"
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              placeholder="e.g. Risk score and counterparty patterns"
              style={{
                width: '100%',
                padding: '4px 8px',
                borderRadius: 4,
                backgroundColor: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 11,
              }}
            />
          </div>

          {/* Evidence-Only Fallback Checkbox */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={allowFallback}
                onChange={(e) => setAllowFallback(e.target.checked)}
                style={{ accentColor: '#38bdf8' }}
              />
              <span>Fallback on failure</span>
            </label>
          </div>

          {/* Launch Button */}
          <button
            onClick={handleLaunchInvestigation}
            disabled={loading || !txIdInput}
            className="btn btn-primary"
            style={{
              padding: '6px 14px',
              fontSize: 12,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginLeft: 'auto',
            }}
          >
            {loading ? (
              <>
                <Loader2 size={13} className="spin" />
                <span>Investigating...</span>
              </>
            ) : (
              <>
                <Play size={13} fill="currentColor" />
                <span>Launch Investigation</span>
              </>
            )}
          </button>
        </div>

        {/* Tab Navigation */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--border-subtle, #182234)',
            backgroundColor: 'rgba(15, 23, 42, 0.3)',
            padding: '0 24px',
          }}
        >
          <button
            className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
            onClick={() => setActiveTab('report')}
            style={{ fontSize: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <FileText size={14} /> Report & Stepper
          </button>

          <button
            className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
            onClick={() => setActiveTab('evidence')}
            style={{ fontSize: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <Database size={14} /> Evidence ({report?.provenance_chain?.length || 0})
          </button>

          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
            style={{ fontSize: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <MessageSquare size={14} /> Follow-Up Chat
          </button>

          <button
            className={`tab-btn ${activeTab === 'feedback' ? 'active' : ''}`}
            onClick={() => setActiveTab('feedback')}
            style={{ fontSize: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <Shield size={14} /> Analyst Review
          </button>

          <button
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
            style={{ fontSize: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <History size={14} /> Session History
          </button>
        </div>

        {/* Main Scrollable Content Area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          {error && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 14px',
                borderRadius: 'var(--radius-sm, 6px)',
                backgroundColor: 'rgba(244, 63, 94, 0.12)',
                border: '1px solid rgba(244, 63, 94, 0.35)',
                color: '#f43f5e',
                fontSize: 12.5,
                marginBottom: 16,
              }}
            >
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* TAB 1: Report & Stepper */}
          {activeTab === 'report' && (
            <div>
              {/* Stepper */}
              <InvestigationWorkflowStepper events={events} currentStatus={status} />

              {/* Report Viewer */}
              {report ? (
                <>
                  <InvestigationReportViewer report={report} onSelectCitation={handleSelectCitation} />
                  <ModelComparisonPanel
                    predictions={report.model_predictions}
                    overallProb={report.model_probability}
                    overallDisagreement={report.model_disagreement}
                    overallSpread={report.probability_spread}
                  />
                  <InteractiveGraphEvidenceViewer
                    transactionId={report.target_transaction_id}
                    snapshotId={report.snapshot_id}
                    graphEvidence={report.graph_evidence}
                    gnnExplainerEvidence={report.gnnexplainer_evidence}
                    onNavigateToGraph={() => {
                      onClose();
                      if (onNavigateToGraph) onNavigateToGraph();
                    }}
                  />
                </>
              ) : (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted, #64748b)' }}>
                  <Sparkles size={32} style={{ margin: '0 auto 12px auto', opacity: 0.5, color: '#38bdf8' }} />
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)' }}>
                    Ready to Investigate Transaction {txIdInput}
                  </div>
                  <div style={{ fontSize: 12, marginTop: 4 }}>
                    Click "Launch Investigation" above to execute the LangGraph multi-step state machine.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Evidence Explorer */}
          {activeTab === 'evidence' && (
            <EvidenceExplorer
              provenanceChain={report?.provenance_chain || []}
              researchCitations={report?.research_citations || []}
              selectedCitationId={selectedCitationId}
              onSelectCitation={(cid) => setSelectedCitationId(cid)}
            />
          )}

          {/* TAB 3: Interactive Chat */}
          {activeTab === 'chat' && (
            <>
              {investigationId ? (
                <InvestigationChat
                  investigationId={investigationId}
                  transactionId={txIdInput}
                  onSelectCitation={handleSelectCitation}
                />
              ) : (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted, #64748b)', fontSize: 13 }}>
                  Please launch an investigation first before opening the interactive Q&A copilot.
                </div>
              )}
            </>
          )}

          {/* TAB 4: Analyst Review */}
          {activeTab === 'feedback' && (
            <>
              {investigationId ? (
                <AnalystFeedbackForm
                  investigationId={investigationId}
                  transactionId={txIdInput}
                  availableCitations={report?.provenance_chain || []}
                />
              ) : (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted, #64748b)', fontSize: 13 }}>
                  Please launch an investigation first to submit review feedback.
                </div>
              )}
            </>
          )}

          {/* TAB 5: Session History */}
          {activeTab === 'history' && (
            <InvestigationHistory
              onSelectInvestigation={handleLoadInvestigationFromHistory}
              activeInvestigationId={investigationId || undefined}
            />
          )}
        </div>

        {/* Action / Export Footer */}
        {report && (
          <div
            style={{
              padding: '12px 24px',
              borderTop: '1px solid var(--border-color, #1e293b)',
              backgroundColor: 'var(--bg-header, rgba(15, 23, 42, 0.94))',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
              Report Status: <strong style={{ color: '#10b981' }}>{report.status}</strong> • Provenance Hash: {report.provenance_hash?.slice(0, 10)}...
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn btn-secondary"
                onClick={handleDownloadJson}
                style={{ fontSize: 12, padding: '6px 12px', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Download size={13} />
                <span>Download JSON</span>
              </button>

              {investigationId && (
                <a
                  href={getInvestigationPdfUrl(investigationId)}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-primary"
                  style={{
                    fontSize: 12,
                    padding: '6px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    textDecoration: 'none',
                  }}
                >
                  <Printer size={13} />
                  <span>Export Official PDF</span>
                </a>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
