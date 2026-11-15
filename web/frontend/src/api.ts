export interface HealthResponse {
  status: string;
  system: string;
  dataset: string;
  models_loaded: string[];
  epistemic_notice: string;
}

export interface ModelMetadata {
  id: string;
  name: string;
  architecture: string;
  category: string;
  dataset?: string;
  feature_representation?: string;
  training_protocol?: string;
  input_dim: number;
  time_dim: number;
  hidden_dim?: number;
  num_heads?: number;
  num_layers?: number;
  dropout?: number;
  parameters_count: number | null;
  checkpoint: string;
  checkpoint_sha256?: string | null;
  held_out_auprc: number | null;
  validation_auprc?: number;
  epistemic_status: string;
}

export interface MetricItem {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  auprc: number;
}

export interface MetricsResponse {
  dataset: string;
  split: string;
  test_nodes_count: number;
  test_illicit_prevalence: number;
  metrics: MetricItem[];
  primary_ranking_metric: string;
  key_finding: string;
}

export interface TransactionSummary {
  node_index: number;
  transaction_id: number;
  time_step: number;
  actual_label: number | null;
  actual_label_name: 'illicit' | 'licit' | 'unknown';
  split: string;
  etgnn_prob: number;
  matched_gat_prob: number;
  has_frozen_explanation: boolean;
  has_intervention_record: boolean;
}

export interface TransactionsPageResponse {
  total: number;
  limit: number;
  offset: number;
  items: TransactionSummary[];
}

export interface PredictionDetail {
  model_id: string;
  display_name: string;
  illicit_probability: number;
  predicted_class: number;
  confidence: number;
}

export interface TransactionDetail {
  transaction_id: number;
  node_index: number;
  time_step: number;
  split: string;
  actual_label: number | null;
  actual_label_name: 'illicit' | 'licit' | 'unknown';
  graph_degree: {
    in_degree: number;
    out_degree: number;
    total_degree: number;
  };
  features_summary: {
    total_features: number;
    local_features_count: number;
    aggregate_features_count: number;
    local_mean: number;
    local_std: number;
    aggregate_mean: number;
    aggregate_std: number;
    sample_features: Record<string, number>;
  };
  predictions: {
    etgnn: PredictionDetail;
    matched_gat: PredictionDetail;
    residual_etgnn?: PredictionDetail;
    random_forest: PredictionDetail;
    xgboost: PredictionDetail;
  };
  explanation_status: {
    has_frozen_explanation: boolean;
    has_intervention_records: boolean;
  };
}

export interface SubgraphNode {
  id: string;
  subgraph_index: number;
  transaction_id: number;
  time_step: number;
  actual_label: number | null;
  actual_label_name: string;
  etgnn_probability: number;
  is_center: boolean;
}

export interface SubgraphEdge {
  id: string;
  source: string;
  target: string;
  source_subgraph_idx: number;
  target_subgraph_idx: number;
  is_incident_to_center: boolean;
}

export interface SubgraphResponse {
  center_transaction_id: number;
  center_node_index: number;
  num_hops: number;
  node_count: number;
  edge_count: number;
  is_truncated?: boolean;
  total_untruncated_nodes?: number;
  truncation_notice?: string | null;
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
}

export interface ExplanationTopEdge {
  source_node_index: number;
  target_node_index: number;
  mask_value: number;
}

export interface ExplanationResponse {
  transaction_id: number;
  node_index: number;
  case?: string;
  is_frozen_research_artifact: boolean;
  is_live_demonstration?: boolean;
  provenance: string;
  scientific_safety_notice: string;
  disclaimer?: string;
  subgraph_nodes?: number;
  subgraph_edges?: number;
  original_predicted_class_probability?: number;
  probability_after_intervention?: number;
  probability_after_removing_top_edges?: number;
  top_edges?: ExplanationTopEdge[];
  signed_drop?: number;
  budget_k?: number;
  candidate_edges_count?: number;
  discrete_deletion_pct?: number;
  predicted_class_before?: number;
  predicted_class_after?: number;
  class_changed?: boolean;
  intervention_protocol?: string;
  trained_model_frozen?: boolean;
  seed?: number;
}

export interface InterventionResponse {
  transaction_id: number;
  node_index: number;
  case?: string;
  model_id: string;
  checkpoint: string;
  trained_model_frozen: boolean;
  original_probability: number;
  original_predicted_class: number;
  intervened_probability: number;
  intervened_predicted_class: number;
  score_difference: number;
  class_changed: boolean;
  candidate_edges_count: number;
  budget_k: number;
  discrete_deletion_pct: number;
  discrete_pct_explanation?: string | null;
  deleted_edges?: Array<{
    source_node_index: number;
    target_node_index: number;
    mask_value?: number;
    direction?: string;
  }>;
  intervention_protocol: string;
  is_frozen_research_artifact: boolean;
  provenance: string;
  epistemic_distinction: string;
  message?: string;
}

export interface C1CohortItem {
  Cohort: string;
  N: number;
  delta_adj_mean: number;
  delta_adj_sample_sd: number;
  delta_adj_median: number;
  delta_adj_iqr: number;
  delta_raw_mean: number;
  delta_raw_sample_sd: number;
  delta_raw_median: number;
  adv_etgnn_mean: number;
  adv_gat_common_mean: number;
  note: string;
}

export interface C1Response {
  experiment: string;
  contrast: string;
  hypothesis: string;
  findings_summary: string;
  data: C1CohortItem[];
  scientific_takeaway: string;
}

export interface C2RegisterItem {
  name: string;
  mean_diff: number;
  ci_95_low: number;
  ci_95_high: number;
  bootstrap_se: number;
  ci_crosses_zero: boolean;
  interpretation: string;
}

export interface C2Response {
  experiment: string;
  registers_count: number;
  total_matched_pairs: number;
  data: any[];
  registers: C2RegisterItem[];
  scientific_takeaway: string;
}

export interface AblationsResponse {
  multi_seed_comparison: {
    etgnn: Array<{ model: string; seed: number; best_epoch: number; validation_auprc: number }>;
    matched_gat: Array<{ model: string; seed: number; best_epoch: number; validation_auprc: number }>;
    finding: string;
  };
  edge_ablation_anomaly: {
    phenomenon: string;
    explanation: string;
    data: Array<{ model: string; val_auprc: number; notes: string }>;
  };
  temporal_prevalence_volatility: Array<{
    time_step: number;
    labelled_count: number;
    illicit_count: number;
    licit_count: number;
    illicit_prevalence: number;
    etgnn_auprc: number;
    no_edge_auprc: number;
  }>;
}

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelMetadata[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchTransactions(params: {
  limit?: number;
  offset?: number;
  split?: string;
  label?: number;
  time_step?: number;
  search?: string;
}): Promise<TransactionsPageResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set('limit', params.limit.toString());
  if (params.offset !== undefined) query.set('offset', params.offset.toString());
  if (params.split) query.set('split', params.split);
  if (params.label !== undefined) query.set('label', params.label.toString());
  if (params.time_step !== undefined) query.set('time_step', params.time_step.toString());
  if (params.search) query.set('search', params.search);

  const res = await fetch(`${API_BASE}/transactions?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch transactions: ${res.statusText}`);
  return res.json();
}

export async function fetchTransaction(txId: number): Promise<TransactionDetail> {
  const res = await fetch(`${API_BASE}/transactions/${txId}`);
  if (!res.ok) throw new Error(`Transaction ${txId} not found`);
  return res.json();
}

export async function fetchSubgraph(txId: number, hops: number = 2): Promise<SubgraphResponse> {
  const res = await fetch(`${API_BASE}/transactions/${txId}/subgraph?hops=${hops}`);
  if (!res.ok) throw new Error(`Failed to fetch subgraph for ${txId}`);
  return res.json();
}

export async function predictTransaction(
  txId: number,
  featuresOverride?: Record<string, number>
): Promise<any> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transaction_id: txId,
      features_override: featuresOverride || null,
    }),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}

export async function fetchExplanation(txId: number): Promise<ExplanationResponse> {
  const res = await fetch(`${API_BASE}/explanations/${txId}`);
  if (!res.ok) throw new Error(`Failed to fetch explanation for ${txId}`);
  return res.json();
}

export async function fetchIntervention(txId: number): Promise<InterventionResponse> {
  const res = await fetch(`${API_BASE}/transactions/${txId}/intervention`);
  if (!res.ok) throw new Error(`Failed to fetch intervention for ${txId}`);
  return res.json();
}

export async function fetchResearchC1(): Promise<C1Response> {
  const res = await fetch(`${API_BASE}/research/c1`);
  if (!res.ok) throw new Error(`Failed to fetch C1 research: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchC2(): Promise<C2Response> {
  const res = await fetch(`${API_BASE}/research/c2`);
  if (!res.ok) throw new Error(`Failed to fetch C2 research: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchAblations(): Promise<AblationsResponse> {
  const res = await fetch(`${API_BASE}/research/ablations`);
  if (!res.ok) throw new Error(`Failed to fetch ablations: ${res.statusText}`);
  return res.json();
}

export interface Phase4ConditionItem {
  condition_id: string;
  display_name: string;
  feature_mode: string;
  input_dim: number;
  param_count: number;
  n_seeds: number;
  auprc_mean: number;
  auprc_sd: number;
  f1_mean: number;
  f1_sd: number;
  precision_mean: number;
  recall_mean: number;
  roc_auc_mean: number;
}

export interface Phase4ContrastItem {
  contrast: string;
  cond_1: string;
  cond_2: string;
  mean_diff: number;
  sd_diff: number;
  ci_95_low: number;
  ci_95_high: number;
  t_stat: number;
  p_unadjusted: number;
  p_holm_6: number;
  positive_seeds: number;
  total_seeds: number;
  epistemic_verdict: string;
}

export interface Phase4SummaryResponse {
  experiment: string;
  evaluation_split: string;
  total_trials: number;
  matrix_sha256: string | null;
  conditions: Phase4ConditionItem[];
  paired_contrasts: Phase4ContrastItem[];
  scientific_takeaway: string;
}

export interface XaiValidationResponse {
  experiment: string;
  evaluation_split: string;
  summary: {
    cohort_size: number;
    mean_e_cand: number;
    fidelity_k1_abs_shift: {
      exact_oracle_mean: number;
      top1_ge_mean: number;
      rand1_mean: number;
      deg1_mean: number;
      bot1_ge_mean: number;
      top1_vs_rand1_diff: number;
      top1_vs_rand1_p: number;
      top1_vs_bot1_diff: number;
      top1_vs_bot1_p: number;
    };
    ego_residual_protection_bound: {
      empty_subgraph_mean_abs_shift: number;
      empty_subgraph_max_abs_shift: number;
    };
    oracle_alignment: {
      mean_spearman_ge_vs_oracle_abs: number;
      top1_oracle_hit_rate: number;
    };
    sparsity: {
      mean_normalized_entropy: number;
      mean_gini_sparsity: number;
    };
    stability: {
      intra_model_spearman_mean: number;
      intra_model_top2_jaccard_mean: number;
      inter_model_spearman_mean: number;
      inter_model_top2_jaccard_mean: number;
    };
    computational_cost: {
      mean_gnnexplainer_ms: number;
      mean_exact_oracle_ms: number;
    };
  };
  scientific_takeaway: string;
}

export interface ExternalValidationResponse {
  experiment: string;
  provenance_classification: string;
  dataset_manifest: {
    dataset: string;
    feature_dim: number;
    total_observations: number;
    unique_addresses: number;
    total_intra_step_edges: number;
    stages?: Record<string, {
      total_nodes: number;
      total_edges: number;
      labelled_nodes: number;
      unseen_labelled_nodes: number;
      illicit_prevalence: number;
    }>;
  };
  validation_summary: {
    split: string;
    seeds: number[];
    conditions: Record<string, {
      condition: string;
      description: string;
      param_count: number;
      metrics: Record<string, {
        mean: number;
        sd: number;
        ci95_low: number;
        ci95_high: number;
      }>;
    }>;
    contrasts_auprc: Array<{
      contrast_id: string;
      comparison: string;
      description: string;
      mean_diff: number;
      sd_diff: number;
      ci95_low: number;
      ci95_high: number;
      t_stat: number;
      p_raw: number;
      p_holm: number | null;
      statistically_significant_holm_005: boolean;
      sign_concordance: string;
    }>;
  };
  test_summary: {
    split: string;
    seeds: number[];
    conditions: Record<string, {
      condition: string;
      description: string;
      param_count: number;
      metrics: Record<string, {
        mean: number;
        sd: number;
        ci95_low: number;
        ci95_high: number;
      }>;
    }>;
    contrasts_auprc: Array<{
      contrast_id: string;
      comparison: string;
      description: string;
      mean_diff: number;
      sd_diff: number;
      ci95_low: number;
      ci95_high: number;
      t_stat: number;
      p_raw: number;
      p_holm: number | null;
      statistically_significant_holm_005: boolean;
      sign_concordance: string;
    }>;
  };
  external_xai_summary: {
    n_evaluated_nodes: number;
    mean_candidate_edges: number;
    ego_residual_empty_subgraph_bound: {
      mean_abs_dp: number;
      max_abs_dp: number;
    };
    fidelity_abs_dp: {
      loo_top1_mean: number;
      gnnexp_top1_mean: number;
      rand_top1_mean: number;
    };
    hypothesis_tests: {
      loo_top1_vs_rand1: {
        t_stat: number;
        p_value: number;
      };
      gnnexp_top1_vs_rand1: {
        t_stat: number;
        p_value: number;
      };
    };
    sparsity_and_latency: {
      gnnexp_entropy_mean: number;
      loo_latency_ms_mean: number;
      gnnexp_latency_ms_mean: number;
    };
  };
  scientific_verdict: string;
}

export async function fetchResearchPhase4(): Promise<Phase4SummaryResponse> {
  const res = await fetch(`${API_BASE}/research/phase4`);
  if (!res.ok) throw new Error(`Failed to fetch Phase 4 summary: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchXaiValidation(): Promise<XaiValidationResponse> {
  const res = await fetch(`${API_BASE}/research/xai_validation`);
  if (!res.ok) throw new Error(`Failed to fetch XAI validation summary: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchExternalValidation(): Promise<ExternalValidationResponse> {
  const res = await fetch(`${API_BASE}/research/external_validation`);
  if (!res.ok) throw new Error(`Failed to fetch External validation summary: ${res.statusText}`);
  return res.json();
}

export interface CensusRecord {
  Cohort: string;
  N_Nodes: number;
  Illicit_Count: number;
  Illicit_Pct: number;
  E_cand_0_Count: number;
  E_cand_0_Pct: number;
  E_cand_1_Count: number;
  E_cand_1_Pct: number;
  E_cand_2_Count: number;
  E_cand_2_Pct: number;
  E_cand_ge2_Count: number;
  E_cand_ge2_Pct: number;
  Mean_E_cand: number;
  Median_E_cand: number;
  Mean_Abs_Shift_Empty_ET165: number;
  Mean_Abs_Shift_Empty_ET165_ActiveOnly: number;
}

export interface StabilityRecord {
  Stratum: string;
  Metric_Key: string;
  Metric_Description: string;
  N_Pre: number;
  N_Post: number;
  Pre_Mean: number;
  Pre_CI95_Low: number;
  Pre_CI95_High: number;
  Post_Mean: number;
  Post_CI95_Low: number;
  Post_CI95_High: number;
  Diff_Post_minus_Pre: number;
  Diff_CI95_Low: number;
  Diff_CI95_High: number;
  MannWhitney_P: number;
  Permutation_P: number;
}

export interface TemporalStabilityResponse {
  experiment: string;
  evaluation_split: string;
  census_summary: {
    total_nodes: number;
    preshift_nodes: number;
    postshift_nodes: number;
    preshift_illicit_count: number;
    postshift_illicit_count: number;
    preshift_illicit_zero_edge_pct: number;
    postshift_illicit_zero_edge_pct: number;
    preshift_licit_zero_edge_pct: number;
    postshift_licit_zero_edge_pct: number;
    census_records: CensusRecord[];
  };
  stability_summary: {
    strata_records: StabilityRecord[];
    stratum_2_matched_pairs: number;
    tost_equivalence_result: string;
  };
  scientific_takeaway: string;
}

export interface MitigationRecord {
  Period: string;
  Strategy_ID: string;
  Strategy_Description: string;
  Family: string;
  Threshold_Used: number;
  Routing_Fraction: number;
  AP_Primary: number;
  PR_AUC_Trapz: number;
  ROC_AUC: number;
  Precision: number;
  Recall: number;
  F1: number;
  FPR: number;
  Brier: number;
  ECE_10: number;
  TP: number;
  FP: number;
  TN: number;
  FN: number;
  Delta_FP_vs_Base: number;
  Delta_TP_vs_Base: number;
}

export interface BootstrapContrastRecord {
  Contrast: string;
  Delta_AP: number;
  AP_CI95_Low: number;
  AP_CI95_High: number;
  AP_Exact_P: number;
  Delta_F1: number;
  F1_CI95_Low: number;
  F1_CI95_High: number;
  F1_Exact_P: number;
  Delta_FPR: number;
  FPR_CI95_Low: number;
  FPR_CI95_High: number;
  AP_Holm_Adjusted_P: number;
  'AP_Holm_Significant_0.05': boolean;
}

export interface ErrorMitigationResponse {
  experiment: string;
  evaluation_split: string;
  strategies_count: number;
  mitigation_results: MitigationRecord[];
  paired_bootstrap_contrasts: BootstrapContrastRecord[];
  scientific_takeaway: string;
}

export async function fetchResearchTemporalStability(): Promise<TemporalStabilityResponse> {
  const res = await fetch(`${API_BASE}/research/temporal_stability`);
  if (!res.ok) throw new Error(`Failed to fetch Temporal Stability: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchErrorMitigation(): Promise<ErrorMitigationResponse> {
  const res = await fetch(`${API_BASE}/research/error_mitigation`);
  if (!res.ok) throw new Error(`Failed to fetch Error Mitigation: ${res.statusText}`);
  return res.json();
}

// Phase 5 Research Monitoring API functions
export async function fetchMonitoringDrift(): Promise<any> {
  const res = await fetch(`${API_BASE}/monitoring/drift`);
  if (!res.ok) throw new Error(`Failed to fetch drift monitoring report: ${res.statusText}`);
  return res.json();
}

export async function fetchMonitoringEvaluation(): Promise<any> {
  const res = await fetch(`${API_BASE}/monitoring/evaluation`);
  if (!res.ok) throw new Error(`Failed to fetch agent evaluation report: ${res.statusText}`);
  return res.json();
}

export async function fetchMonitoringCandidates(): Promise<any> {
  const res = await fetch(`${API_BASE}/monitoring/candidates`);
  if (!res.ok) throw new Error(`Failed to fetch candidate models: ${res.statusText}`);
  return res.json();
}

export async function fetchMonitoringFeedbackStats(): Promise<any> {
  const res = await fetch(`${API_BASE}/monitoring/feedback-stats`);
  if (!res.ok) throw new Error(`Failed to fetch feedback governance stats: ${res.statusText}`);
  return res.json();
}


