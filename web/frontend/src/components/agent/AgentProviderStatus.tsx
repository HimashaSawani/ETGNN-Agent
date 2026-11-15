import type React from 'react';
import { AlertTriangle, FileText, CheckCircle2 } from 'lucide-react';
import type { GenerationMode } from '../../types/agent';

interface Props {
  generationMode?: GenerationMode;
  providerType?: string;
  providerModelId?: string;
  liveVerified?: boolean;
}

export const AgentProviderStatus: React.FC<Props> = ({
  generationMode = 'SIMULATED',
  providerType = 'mock',
  providerModelId = 'deterministic_mock_v3',
  liveVerified = false,
}) => {
  const isLive = generationMode === 'LIVE' && liveVerified;
  const isLiveUnverified = generationMode === 'LIVE' && !liveVerified;
  const isEvidenceOnly = generationMode === 'DETERMINISTIC_EVIDENCE_ONLY';

  return (
    <div className="agent-provider-status" style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
      padding: '4px 10px',
      borderRadius: 'var(--radius-sm, 6px)',
      fontSize: 12,
      fontWeight: 500,
      backgroundColor: isLive
        ? 'rgba(16, 185, 129, 0.12)'
        : isLiveUnverified
        ? 'rgba(239, 68, 68, 0.12)'
        : isEvidenceOnly
        ? 'rgba(56, 189, 248, 0.12)'
        : 'rgba(245, 158, 11, 0.12)',
      border: `1px solid ${
        isLive
          ? 'rgba(16, 185, 129, 0.35)'
          : isLiveUnverified
          ? 'rgba(239, 68, 68, 0.35)'
          : isEvidenceOnly
          ? 'rgba(56, 189, 248, 0.35)'
          : 'rgba(245, 158, 11, 0.35)'
      }`,
      color: isLive ? '#10b981' : isLiveUnverified ? '#ef4444' : isEvidenceOnly ? '#38bdf8' : '#f59e0b',
    }}>
      {isLive ? (
        <>
          <CheckCircle2 size={13} />
          <span>LIVE LLM: {providerModelId || providerType}</span>
          <span style={{ fontSize: 10, opacity: 0.8 }}>(Verified)</span>
        </>
      ) : isLiveUnverified ? (
        <>
          <AlertTriangle size={13} />
          <span>LIVE LLM: {providerModelId || providerType}</span>
          <span style={{ fontSize: 10, opacity: 0.8 }}>(Unverified Live Execution)</span>
        </>
      ) : isEvidenceOnly ? (
        <>
          <FileText size={13} />
          <span>EVIDENCE-ONLY: Deterministic Factual Mode</span>
          <span style={{ fontSize: 10, opacity: 0.8 }}>(Non-Generative Fallback)</span>
        </>
      ) : (
        <>
          <AlertTriangle size={13} />
          <span>SIMULATED: Deterministic Offline Mock</span>
          <span style={{ fontSize: 10, opacity: 0.8 }}>(Zero Hallucination)</span>
        </>
      )}
    </div>
  );
};
