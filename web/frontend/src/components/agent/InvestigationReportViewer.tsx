import type React from 'react';
import {
  FileText,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  Hash,
  Info,
} from 'lucide-react';
import type { ForensicInvestigationReport } from '../../types/agent';
import { AgentProviderStatus } from './AgentProviderStatus';

interface Props {
  report: ForensicInvestigationReport;
  onSelectCitation?: (citationId: string) => void;
}

export const InvestigationReportViewer: React.FC<Props> = ({
  report,
  onSelectCitation,
}) => {
  const isElevated =
    report.consensus_verdict.includes('HIGH_RISK') ||
    report.consensus_verdict.includes('SUSPICIOUS') ||
    (report.model_probability !== undefined && report.model_probability >= 0.50);

  const isAbstained = report.status === 'ABSTAINED' || report.investigation_status === 'ABSTAINED';

  const verdictColor = isAbstained ? '#f59e0b' : isElevated ? '#f43f5e' : '#10b981';

  return (
    <div className="investigation-report-viewer card" style={{ padding: '20px 24px', marginBottom: 20 }}>
      {/* Report Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileText size={20} color="var(--accent-cyan, #38bdf8)" />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0, color: 'var(--text-primary, #f8fafc)' }}>
              Forensic Investigation Dossier
            </h2>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted, #64748b)', marginTop: 4 }}>
            ID: <span style={{ fontFamily: 'var(--font-mono, monospace)', color: '#38bdf8' }}>{report.investigation_id}</span> • Tx: <span style={{ fontFamily: 'var(--font-mono, monospace)', color: '#f8fafc' }}>{report.target_transaction_id}</span> • Snapshot: {report.snapshot_id}
            {report.evidence_completeness && (
              <> • Completeness: <span style={{ color: report.evidence_completeness === 'COMPLETE' ? '#10b981' : '#f59e0b', fontWeight: 600 }}>{report.evidence_completeness}</span></>
            )}
            {report.workflow_stages_completed && report.workflow_stages_completed.length > 0 && (
              <> • Stages: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{report.workflow_stages_completed.length}/11</span></>
            )}
            {report.token_usage?.total_tokens !== undefined && (
              <> • Tokens: <span style={{ fontFamily: 'var(--font-mono, monospace)' }}>{report.token_usage.total_tokens}</span></>
            )}
          </div>
        </div>

        <AgentProviderStatus
          generationMode={report.generation_mode}
          providerType={report.provider_type}
          providerModelId={report.provider_model_id}
          liveVerified={report.live_llm_verified}
        />
      </div>

      {/* Consensus Verdict Box */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 18px',
          borderRadius: 'var(--radius-md, 10px)',
          backgroundColor: isAbstained
            ? 'rgba(245, 158, 11, 0.1)'
            : isElevated
            ? 'rgba(244, 63, 94, 0.1)'
            : 'rgba(16, 185, 129, 0.1)',
          border: `1px solid ${verdictColor}50`,
          marginBottom: 16,
        }}
      >
        <div>
          <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-muted, #64748b)', fontWeight: 600 }}>
            Consensus Calibrated Verdict
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, color: verdictColor, marginTop: 2 }}>
            {report.consensus_verdict}
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
            Statistical Probability
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-mono, monospace)', color: verdictColor, marginTop: 2 }}>
            {report.model_probability !== undefined ? report.model_probability.toFixed(4) : 'N/A'}
          </div>
        </div>
      </div>

      {/* Narrative Summary */}
      <div style={{ marginBottom: 18 }}>
        <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted, #64748b)', marginBottom: 6 }}>
          1. Forensic Executive Summary
        </h4>
        <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text-primary, #f8fafc)', backgroundColor: 'rgba(15, 23, 42, 0.3)', padding: '12px 16px', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle, #182234)' }}>
          {report.narrative_summary || 'No narrative generated.'}
        </div>
      </div>

      {/* Supported Findings List */}
      <div style={{ marginBottom: 18 }}>
        <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted, #64748b)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
          <ShieldCheck size={15} color="#10b981" />
          <span>2. Supported Analytical Findings ({report.supported_findings.length})</span>
        </h4>

        {report.supported_findings.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {report.supported_findings.map((f, idx) => {
              const findingText = f.finding;
              const cites = f.evidence_citations && f.evidence_citations.length > 0 ? f.evidence_citations : (f.evidence_id ? [f.evidence_id] : []);

              return (
                <div
                  key={idx}
                  style={{
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-sm, 6px)',
                    backgroundColor: 'rgba(15, 23, 42, 0.25)',
                    border: '1px solid var(--border-subtle, #182234)',
                  }}
                >
                  <div style={{ fontSize: 12.5, fontWeight: 500, color: 'var(--text-primary, #f8fafc)', lineHeight: 1.5 }}>
                    {findingText}
                  </div>

                  {cites.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
                      <span style={{ fontSize: 10, color: 'var(--text-muted, #64748b)' }}>Citations:</span>
                      {cites.map((cid, cIdx) => (
                        <button
                          key={cIdx}
                          onClick={() => onSelectCitation && onSelectCitation(cid)}
                          className="citation-pill"
                          style={{
                            fontSize: 10.5,
                            fontFamily: 'var(--font-mono, monospace)',
                            padding: '2px 8px',
                            borderRadius: 12,
                            backgroundColor: 'rgba(56, 189, 248, 0.12)',
                            color: '#38bdf8',
                            border: '1px solid rgba(56, 189, 248, 0.35)',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                          }}
                          title={`Click to inspect evidence record ${cid}`}
                        >
                          [{cid}]
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: 'var(--text-muted, #64748b)', fontStyle: 'italic' }}>
            No individual claims supported; investigation abstained or baseline verified.
          </div>
        )}
      </div>

      {/* Hypothesis Reasoning Evaluations (Phase 5D/5E) */}
      {report.hypothesis_evaluations && report.hypothesis_evaluations.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted, #64748b)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <ShieldCheck size={15} color="#38bdf8" />
            <span>2B. Structured Hypothesis Evaluation ({report.hypothesis_evaluations.length})</span>
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {report.hypothesis_evaluations.map((hyp, hIdx) => {
              const badgeColor =
                hyp.classification === 'SUPPORTED'
                  ? '#10b981'
                  : hyp.classification === 'PARTIALLY_SUPPORTED'
                  ? '#f59e0b'
                  : hyp.classification === 'UNSUPPORTED'
                  ? '#f43f5e'
                  : '#94a3b8';
              return (
                <div
                  key={hIdx}
                  style={{
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-sm, 6px)',
                    backgroundColor: 'rgba(15, 23, 42, 0.25)',
                    border: '1px solid var(--border-subtle, #182234)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-primary, #f8fafc)' }}>
                      {hyp.hypothesis}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 700,
                        padding: '2px 8px',
                        borderRadius: 4,
                        backgroundColor: `${badgeColor}20`,
                        color: badgeColor,
                        border: `1px solid ${badgeColor}50`,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {hyp.classification}
                    </span>
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-secondary, #94a3b8)', lineHeight: 1.4 }}>
                    {hyp.rationale}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Uncertain or Degraded Findings */}
      {report.uncertain_findings && report.uncertain_findings.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', color: '#f59e0b', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <AlertTriangle size={15} color="#f59e0b" />
            <span>3. Uncertain or Degraded Assertions ({report.uncertain_findings.length})</span>
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {report.uncertain_findings.map((uf, idx) => (
              <div
                key={idx}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm, 6px)',
                  backgroundColor: 'rgba(245, 158, 11, 0.08)',
                  border: '1px solid rgba(245, 158, 11, 0.25)',
                  fontSize: 12,
                  color: 'var(--text-secondary, #94a3b8)',
                  lineHeight: 1.4,
                }}
              >
                {uf}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Limitations & Boundaries */}
      <div style={{ marginBottom: 16 }}>
        <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted, #64748b)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
          <HelpCircle size={15} color="var(--accent-cyan, #38bdf8)" />
          <span>4. Methodological Limitations & Epistemic Boundaries</span>
        </h4>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {(report.evidence_limitations || report.limitations || []).map((lim, idx) => (
            <div key={idx} style={{ fontSize: 11.5, color: 'var(--text-secondary, #94a3b8)', lineHeight: 1.4 }}>
              • {lim}
            </div>
          ))}
        </div>
      </div>

      {/* Epistemic Notice */}
      <div
        style={{
          padding: '10px 14px',
          borderRadius: 'var(--radius-sm, 6px)',
          backgroundColor: 'rgba(15, 23, 42, 0.4)',
          border: '1px solid var(--border-subtle, #182234)',
          fontSize: 11,
          color: 'var(--text-muted, #64748b)',
          lineHeight: 1.4,
          display: 'flex',
          alignItems: 'flex-start',
          gap: 8,
        }}
      >
        <Info size={14} style={{ marginTop: 2, flexShrink: 0 }} />
        <span>
          <strong>Epistemic Notice:</strong> {report.epistemic_notice || 'Model probability represents raw statistical output, NOT a calibrated Bayesian confidence. Statistical risk predictions do not constitute judicial proof or legal evidence of financial crime.'}
        </span>
      </div>

      {/* Cryptographic Audit Hash */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 14, paddingTop: 10, borderTop: '1px solid var(--border-subtle, #182234)', fontSize: 10, color: 'var(--text-muted, #64748b)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Hash size={12} />
          <span>SHA256 Audit: <span style={{ fontFamily: 'var(--font-mono, monospace)' }}>{report.provenance_hash || `audit_${report.target_transaction_id}`}</span></span>
        </div>
        <div>
          ETGNN-Agent Verified Forensic Pipeline
        </div>
      </div>
    </div>
  );
};
