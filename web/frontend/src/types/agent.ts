/**
 * ETGNN-Agent Phase 4 TypeScript Interfaces.
 * Strongly typed definitions matching backend LangGraph schemas,
 * forensic reports, telemetry events, chat, and feedback models.
 */

export type GenerationMode = 'SIMULATED' | 'DETERMINISTIC_EVIDENCE_ONLY' | 'LIVE' | 'UNKNOWN';

export interface SupportedFinding {
  finding: string;
  evidence_id?: string;
  evidence_citations?: string[];
  confidence?: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface ModelPredictionsSummary {
  random_forest_prob?: number;
  xgboost_prob?: number;
  canonical_etgnn_prob?: number;
  matched_gat_prob?: number;
  model_divergence_detected?: boolean;
  divergence_explanation?: string | null;
}

export interface HypothesisEvaluation {
  hypothesis_id: string;
  hypothesis: string;
  classification: 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'UNSUPPORTED' | 'NOT_TESTABLE' | string;
  evidence_ids: string[];
  rationale: string;
}

export interface GraphEvidenceSummary {
  incoming_edges_count: number;
  outgoing_edges_count: number;
  subgraph_nodes_2hop: number;
  illicit_neighbors_count: number;
  is_isolated_incoming?: boolean;
  is_fully_isolated?: boolean;
}

export interface GNNExplainerEvidenceSummary {
  top_feature_indices: number[];
  fidelity_drop: number;
  interpretation_disclaimer?: string;
}

export interface ForensicInvestigationReport {
  investigation_id: string;
  target_transaction_id: number;
  snapshot_id: number;
  status: string;
  investigation_status?: string;
  consensus_verdict: string;
  predicted_class?: number;
  model_probability?: number;
  model_disagreement?: boolean;
  probability_spread?: number;
  risk_indicator?: string;
  epistemic_notice?: string;
  provider_type?: string;
  provider_model_id?: string;
  generation_mode?: GenerationMode;
  live_llm_verified?: boolean;
  token_usage?: Record<string, number>;
  llm_latency_ms?: number;
  estimated_cost_usd?: number;
  evidence_completeness?: string;
  workflow_stages_completed?: string[];
  narrative_summary?: string;
  model_predictions?: ModelPredictionsSummary;
  graph_evidence?: GraphEvidenceSummary;
  gnnexplainer_evidence?: GNNExplainerEvidenceSummary | null;
  supported_findings: SupportedFinding[];
  hypothesis_evaluations?: HypothesisEvaluation[];
  uncertain_findings: string[];
  limitations: string[];
  evidence_limitations?: string[];
  research_citations: string[];
  provenance_chain: string[];
  provenance_hash?: string;
  generated_at?: string;
}

export interface AgentInvestigationEvent {
  event_id: string;
  investigation_id: string;
  timestamp: string;
  event_type: string;
  execution_status: 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILED' | 'SKIPPED';
  tool_name?: string | null;
  duration_ms: number;
  safe_input_summary?: Record<string, any> | null;
  safe_output_summary?: Record<string, any> | null;
  evidence_ref_ids: string[];
  error_code?: string | null;
  message?: string | null;
}

export interface AgentInvestigateRequest {
  transaction_id: number;
  question?: string;
  evidence_mode?: 'RETROSPECTIVE_INVESTIGATION' | 'HISTORICAL_EVALUATION';
  as_of_snapshot?: number;
  llm_provider?: 'mock' | 'evidence_only' | 'gemini' | 'anthropic' | 'ollama';
  allow_evidence_only_fallback?: boolean;
}

export interface AgentInvestigateResponse {
  investigation_id: string;
  status: string;
  report?: ForensicInvestigationReport | null;
  warnings: string[];
  errors: string[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  cited_evidence_ids?: string[];
  provider_type?: string;
  generation_mode?: GenerationMode;
  timestamp: string;
  limitations?: string[];
}

export interface ChatRequest {
  investigation_id: string;
  question: string;
  llm_provider?: string;
}

export interface ChatResponse {
  investigation_id: string;
  transaction_id: number;
  question: string;
  answer: string;
  cited_evidence_ids: string[];
  provider_type: string;
  generation_mode: GenerationMode;
  limitations: string[];
}

export interface FeedbackSubmission {
  investigation_id: string;
  transaction_id: number;
  feedback_category: 'helpful' | 'unhelpful' | 'citation_issue' | 'missing_evidence' | 'incorrect_numerical' | 'needs_further_review';
  rating: number;
  comment?: string;
  reviewer_id?: string;
  report_version?: string;
  evidence_references?: string[];
}

export interface FeedbackRecord extends FeedbackSubmission {
  feedback_id: string;
  created_at: string;
}

export interface FeedbackStats {
  total_feedback_count: number;
  average_rating: number;
  category_distribution: Record<string, number>;
  storage_engine: string;
  safety_invariant: string;
}

export interface FeedbackListResponse {
  stats: FeedbackStats;
  items: FeedbackRecord[];
}

export interface InvestigationHistoryItem {
  investigation_id: string;
  transaction_id: number;
  status: string;
  consensus_verdict?: string;
  risk_indicator?: string;
  model_disagreement?: boolean;
  probability_spread?: number;
  generation_mode?: GenerationMode;
  provider_type?: string;
  created_at?: string;
  session_note?: string;
}

