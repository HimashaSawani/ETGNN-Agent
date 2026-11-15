/**
 * ETGNN-Agent Phase 4 API Service Client.
 * Connects frontend investigation drawer and human feedback workflows
 * to verified FastAPI backend routes.
 */

import type {
  AgentInvestigateRequest,
  AgentInvestigateResponse,
  AgentInvestigationEvent,
  ChatRequest,
  ChatResponse,
  FeedbackListResponse,
  FeedbackRecord,
  FeedbackSubmission,
  InvestigationHistoryItem,
} from '../types/agent';

const API_BASE = '/api';

function formatErrorMessage(err: any, defaultMsg: string): string {
  if (!err) return defaultMsg;
  if (typeof err.detail === 'string') return err.detail;
  if (err.detail && typeof err.detail === 'object') {
    return err.detail.message || err.detail.error_code || JSON.stringify(err.detail);
  }
  if (typeof err.message === 'string') return err.message;
  return defaultMsg;
}

/**
 * Triggers an autonomous multi-step LangGraph forensic investigation.
 */
export async function runAgentInvestigation(
  req: AgentInvestigateRequest
): Promise<AgentInvestigateResponse> {
  const res = await fetch(`${API_BASE}/agent/investigate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(formatErrorMessage(err, `Investigation failed with HTTP ${res.status}`));
  }

  return res.json();
}

/**
 * Retrieves a full forensic investigation report and status by ID.
 */
export async function fetchInvestigation(
  investigationId: string
): Promise<AgentInvestigateResponse> {
  const res = await fetch(`${API_BASE}/agent/investigations/${investigationId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch investigation ${investigationId}`);
  }
  return res.json();
}

/**
 * Retrieves observable telemetry events for an investigation trace.
 */
export async function fetchInvestigationEvents(
  investigationId: string
): Promise<AgentInvestigationEvent[]> {
  const res = await fetch(`${API_BASE}/agent/investigations/${investigationId}/events`);
  if (!res.ok) {
    return [];
  }
  return res.json();
}

/**
 * Lists session-cached investigations.
 */
export async function fetchInvestigationList(): Promise<InvestigationHistoryItem[]> {
  const res = await fetch(`${API_BASE}/agent/investigations`);
  if (!res.ok) {
    return [];
  }
  return res.json();
}

/**
 * Sends a follow-up inquiry grounded in the verified investigation evidence bundle.
 */
export async function askInvestigationChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(formatErrorMessage(err, 'Follow-up query failed'));
  }

  return res.json();
}

/**
 * Submits human-in-the-loop analyst review feedback (stored separately in SQLite).
 */
export async function submitAnalystFeedback(
  submission: FeedbackSubmission
): Promise<FeedbackRecord> {
  const res = await fetch(`${API_BASE}/agent/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(submission),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(formatErrorMessage(err, 'Failed to submit feedback'));
  }

  return res.json();
}

/**
 * Retrieves analyst feedback records for an investigation.
 */
export async function fetchAnalystFeedback(
  investigationId?: string
): Promise<FeedbackRecord[]> {
  const url = investigationId
    ? `${API_BASE}/agent/feedback/${investigationId}`
    : `${API_BASE}/agent/feedback`;

  const res = await fetch(url);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  if (Array.isArray(data)) {
    return data;
  }
  if (data && Array.isArray(data.items)) {
    return data.items;
  }
  return [];
}

/**
 * Retrieves overall analyst feedback overview with metrics.
 */
export async function fetchAnalystFeedbackOverview(): Promise<FeedbackListResponse> {
  const res = await fetch(`${API_BASE}/agent/feedback`);
  if (!res.ok) {
    return {
      stats: {
        total_feedback_count: 0,
        average_rating: 0,
        category_distribution: {},
        storage_engine: 'sqlite_isolated',
        safety_invariant: 'read_only_isolation',
      },
      items: [],
    };
  }
  return res.json();
}

/**
 * Generates direct download URL for official forensic investigation PDF report.
 */
export function getInvestigationPdfUrl(investigationId: string): string {
  return `${API_BASE}/agent/investigations/${investigationId}/export/pdf`;
}
