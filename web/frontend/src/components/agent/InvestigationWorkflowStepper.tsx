import type React from 'react';
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2,
  MinusCircle,
  Activity,
} from 'lucide-react';
import type { AgentInvestigationEvent } from '../../types/agent';

interface Props {
  events: AgentInvestigationEvent[];
  currentStatus: string;
}

interface StepDefinition {
  id: string;
  name: string;
  description: string;
  eventMatches: string[];
}

const WORKFLOW_STEPS: StepDefinition[] = [
  {
    id: 'validate_request',
    name: '1. Validate Request',
    description: 'Verify transaction identity and enforce point-in-time snapshot boundaries.',
    eventMatches: ['INVESTIGATION_STARTED'],
  },
  {
    id: 'plan_investigation',
    name: '2. Plan Investigation',
    description: 'Evaluate query requirements and plan bounded tool execution steps.',
    eventMatches: ['INVESTIGATION_PLANNED'],
  },
  {
    id: 'execute_tools',
    name: '3. Execute Tools',
    description: 'Retrieve transaction record, extract 2-hop subgraph, and query model baselines.',
    eventMatches: ['TOOL_COMPLETED'],
  },
  {
    id: 'consolidate_evidence',
    name: '4. Consolidate Evidence',
    description: 'Aggregate predictions, calculate probability spread, and detect model divergence.',
    eventMatches: ['EVIDENCE_CONSOLIDATED'],
  },
  {
    id: 'generate_explanation',
    name: '5. Generate Explanation',
    description: 'Synthesize structured forensic findings strictly grounded in retrieved evidence.',
    eventMatches: ['EXPLANATION_STARTED', 'EXPLANATION_GENERATED'],
  },
  {
    id: 'validate_claims',
    name: '6. Validate Claims',
    description: 'Verify numeric assertions against ±0.01 probability and ±0.001 benchmark tolerances.',
    eventMatches: ['CLAIMS_VALIDATED'],
  },
  {
    id: 'validate_report',
    name: '7. Validate Report',
    description: 'Enforce topological distinctions, Snapshot 43 caveats, and GNNExplainer non-causality.',
    eventMatches: ['REPORT_VALIDATED'],
  },
  {
    id: 'finalize_investigation',
    name: '8. Finalize Investigation',
    description: 'Package tamper-evident forensic dossier with consensus verdict and audit checksum.',
    eventMatches: ['INVESTIGATION_COMPLETED', 'INVESTIGATION_ABSTAINED'],
  },
];

export const InvestigationWorkflowStepper: React.FC<Props> = ({
  events,
  currentStatus,
}) => {
  const isCompleted = currentStatus === 'COMPLETED';
  const isAbstained = currentStatus === 'ABSTAINED';
  const isFailed = currentStatus === 'FAILED';
  const isRunning = currentStatus === 'RUNNING' || currentStatus === 'VALIDATING_REQUEST';

  return (
    <div className="workflow-stepper card" style={{ padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Activity size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            LangGraph Investigation State Machine (8 Stages)
          </h3>
        </div>
        <span
          className="badge"
          style={{
            fontSize: 11,
            backgroundColor: isCompleted
              ? 'rgba(16, 185, 129, 0.15)'
              : isAbstained
              ? 'rgba(245, 158, 11, 0.15)'
              : isFailed
              ? 'rgba(244, 63, 94, 0.15)'
              : 'rgba(56, 189, 248, 0.15)',
            color: isCompleted
              ? '#10b981'
              : isAbstained
              ? '#f59e0b'
              : isFailed
              ? '#f43f5e'
              : '#38bdf8',
            border: '1px solid currentColor',
          }}
        >
          Status: {currentStatus}
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {WORKFLOW_STEPS.map((step, idx) => {
          // Find matching event for this step
          const matchingEvents = events.filter((e) => step.eventMatches.includes(e.event_type));
          const stepDone = isCompleted || matchingEvents.length > 0;
          const stepDuration = matchingEvents.reduce((acc, e) => acc + (e.duration_ms || 0), 0);
          const hasError = matchingEvents.some((e) => e.execution_status === 'FAILED');

          let statusIcon = <Clock size={14} color="var(--text-muted, #64748b)" />;
          let iconColor = 'var(--text-muted, #64748b)';
          let badgeText = 'Pending';

          if (hasError) {
            statusIcon = <AlertCircle size={14} color="#f43f5e" />;
            iconColor = '#f43f5e';
            badgeText = 'Failed';
          } else if (isAbstained && idx >= 1) {
            statusIcon = <MinusCircle size={14} color="#f59e0b" />;
            iconColor = '#f59e0b';
            badgeText = 'Abstained';
          } else if (stepDone) {
            statusIcon = <CheckCircle2 size={14} color="#10b981" />;
            iconColor = '#10b981';
            badgeText = stepDuration > 0 ? `${stepDuration} ms` : 'Done';
          } else if (isRunning && idx === 0) {
            statusIcon = <Loader2 size={14} className="spin" color="#38bdf8" />;
            iconColor = '#38bdf8';
            badgeText = 'Running';
          }

          return (
            <div
              key={step.id}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm, 6px)',
                backgroundColor: stepDone
                  ? 'rgba(15, 23, 42, 0.4)'
                  : 'rgba(15, 23, 42, 0.15)',
                border: `1px solid ${stepDone ? 'rgba(56, 189, 248, 0.2)' : 'var(--border-subtle, #182234)'}`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                <div style={{ marginTop: 2 }}>{statusIcon}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: iconColor }}>
                    {step.name}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                    {step.description}
                  </div>
                </div>
              </div>

              <div
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--font-mono, monospace)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  color: iconColor,
                  whiteSpace: 'nowrap',
                }}
              >
                {badgeText}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

