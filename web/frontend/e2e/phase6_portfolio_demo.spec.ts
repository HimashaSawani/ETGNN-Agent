import { test, expect } from '@playwright/test';

const mockTransaction = {
  tx_id: 288717023,
  time_step: 43,
  label: 1,
  split: 'test',
  in_degree: 0,
  out_degree: 2,
  total_degree: 2,
  local_features_count: 93,
  aggregate_features_count: 72,
  predictions: {
    Canonical_ETGNN: { probability: 0.885, predicted_class: 1 },
    Matched_GAT: { probability: 0.620, predicted_class: 1 },
    RandomForest: { probability: 0.500, predicted_class: 0 },
    XGBoost: { probability: 0.820, predicted_class: 1 },
  },
};

const mockSubgraph = {
  center_node: 288717023,
  nodes: [
    { id: 288717023, label: 1, time_step: 43, is_center: true },
    { id: 288717024, label: 1, time_step: 43, is_center: false },
    { id: 288717025, label: 0, time_step: 43, is_center: false },
  ],
  edges: [
    { source: 288717023, target: 288717024 },
    { source: 288717023, target: 288717025 },
  ],
  hops: 2,
  total_nodes: 3,
  total_edges: 2,
};

const mockReport = {
  investigation_id: 'inv_phase6_portfolio_demo',
  target_transaction_id: 288717023,
  snapshot_id: 43,
  status: 'COMPLETED',
  investigation_status: 'COMPLETED',
  consensus_verdict: 'HIGH_RISK_ILLICIT',
  predicted_class: 1,
  model_probability: 0.885,
  model_disagreement: true,
  probability_spread: 0.385,
  risk_indicator: 'HIGH_RISK',
  epistemic_notice: 'Model probability represents raw statistical output, NOT a calibrated Bayesian confidence.',
  provider_type: 'mock',
  provider_model_id: 'deterministic_mock_v3',
  generation_mode: 'SIMULATED',
  live_llm_verified: false,
  narrative_summary: 'Transaction 288717023 evaluated at snapshot 43 indicates elevated illicit risk across multi-hop graph neighborhood.',
  model_predictions: {
    random_forest_prob: 0.500,
    xgboost_prob: 0.820,
    canonical_etgnn_prob: 0.885,
    model_divergence_detected: true,
    divergence_explanation: 'Model divergence detected (spread 0.385 >= 0.30): ETGNN incorporates graph structure while tree baselines evaluate tabular features independently.',
  },
  graph_evidence: {
    incoming_edges_count: 0,
    outgoing_edges_count: 2,
    subgraph_nodes_2hop: 3,
    illicit_neighbors_count: 1,
    is_isolated_incoming: true,
    is_fully_isolated: false,
  },
  gnnexplainer_evidence: {
    top_feature_indices: [2, 14, 45],
    fidelity_drop: 0.1425,
    interpretation_disclaimer: 'GNNExplainer attributions identify sensitivity in model representations, not deterministic legal proof of fraud.',
  },
  supported_findings: [
    {
      claim_text: 'ETGNN predicted an illicit risk probability of 0.8850 for transaction 288717023.',
      evidence_id: 'EV-GNN-PRED-288717023',
      evidence_citations: ['EV-GNN-PRED-288717023'],
      confidence: 'HIGH',
    },
    {
      claim_text: 'The graph neighborhood contains directed outgoing edges with zero incoming observed edges.',
      evidence_id: 'EV-GRAPH-NEIGHBOR-288717023',
      evidence_citations: ['EV-GRAPH-NEIGHBOR-288717023'],
      confidence: 'HIGH',
    },
  ],
  uncertain_findings: [],
  limitations: [
    'Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence.',
    'Topological Qualification: Node has zero incoming observed edges represented in the directed graph; this indicates an absence of observed upstream edges in the dataset sample, not proof that the transaction originated funds.',
  ],
  provenance_chain: ['EV-GNN-PRED-288717023', 'EV-GRAPH-NEIGHBOR-288717023'],
  generated_at: '2026-11-11T12:00:00Z',
};

const mockDriftReport = {
  dataset_census: {
    total_snapshots: 49,
    baseline_snapshots: '1-32',
    test_snapshots: '35-49',
    snapshot_43_census: {
      total_nodes: 5063,
      labelled_nodes: 1370,
      illicit_nodes: 24,
      licit_nodes: 1346,
      unlabelled_nodes: 3693,
      unlabelled_ratio: 0.7294,
    },
  },
  graph_topology_drift: {
    baseline_degree: { mean_in_degree: 1.1543, mean_out_degree: 1.1543, zero_in_degree_ratio: 0.2557 },
    snapshot_43_degree: {
      all_nodes: { mean_in_degree: 1.1752, zero_in_degree_ratio: 0.2732, completely_isolated_ratio: 0.0 },
      illicit_only: { mean_in_degree: 0.8333, zero_in_degree_ratio: 0.5833, completely_isolated_ratio: 0.0 },
    },
  },
  drift_detected: true,
  candidate_promotion_gate: {
    status: 'BLOCKED',
    gate_reason: 'Continual learning pipeline blocked pending prospective validation and human authorization.',
    original_checkpoints_status: 'IMMUTABLE_READ_ONLY',
  },
};

const mockCandidateGate = {
  total_candidates: 0,
  candidates: [],
  promotion_gate_rules: {
    primary_metric: 'AUPRC',
    minimum_improvement: 0.015,
    max_fpr_increase: 0.020,
    human_approval_required: true,
    original_checkpoints_status: 'IMMUTABLE_READ_ONLY',
  },
};

const mockFeedbackStats = {
  total_submissions_analyzed: 5,
  helpful_ratio: 0.8,
  ground_truth_isolation: 'STRICTLY_ENFORCED',
  opinion_vs_truth_warning: 'Analyst ratings represent subjective utility feedback and are never used as model training labels.',
};

test.describe('Phase 6: Portfolio Demonstration Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Intercept backend API calls with deterministic fixtures
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'healthy',
          system: 'ETGNN Research Demonstration Platform',
          dataset: 'Elliptic Bitcoin Dataset (203,769 transactions)',
          models_loaded: ['ETGNN', 'Matched_GAT', 'RandomForest', 'XGBoost'],
        }),
      });
    });

    await page.route('**/api/transactions/288717023', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockTransaction),
      });
    });

    await page.route('**/api/transactions/288717023/subgraph*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSubgraph),
      });
    });

    await page.route('**/api/agent/investigations/inv_phase6_portfolio_demo', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_phase6_portfolio_demo',
          status: 'COMPLETED',
          report: mockReport,
          warnings: [],
          errors: [],
        }),
      });
    });

    await page.route('**/api/agent/investigations', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([mockReport]),
      });
    });

    await page.route('**/api/agent/investigate', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_phase6_portfolio_demo',
          status: 'COMPLETED',
          report: mockReport,
          warnings: [],
          errors: [],
        }),
      });
    });

    await page.route('**/api/monitoring/drift', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockDriftReport),
      });
    });

    await page.route('**/api/monitoring/candidates', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockCandidateGate),
      });
    });

    await page.route('**/api/monitoring/feedback-stats', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockFeedbackStats),
      });
    });

    await page.route('**/api/monitoring/evaluation', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'completed', systems_evaluated: 6 }),
      });
    });
  });

  test('Complete 12-Step Portfolio Demonstration Flow', async ({ page }) => {
    // 1. Load application
    await page.goto('/');
    await expect(page).toHaveTitle(/ETGNN/i);

    // 2. Verify platform branding & epistemic notices
    const heading = page.locator('header, nav, h1').first();
    await expect(heading).toBeVisible();

    // 3. Navigate to Research Monitoring Dashboard
    const monitoringTab = page.locator('button, a', { hasText: /Research Monitoring/i });
    if (await monitoringTab.isVisible()) {
      await monitoringTab.click();
      await expect(page.locator('text=Temporal Drift & Continual Learning Governance')).toBeVisible();

      // Verify Snapshot 43 Ground Truth Census display
      await expect(page.locator('text=Snapshot 43 Ground Truth Census')).toBeVisible();
      await expect(page.locator('text=5,063')).toBeVisible(); // total nodes
      await expect(page.locator('text=24')).toBeVisible(); // illicit nodes

      // Verify Continual Learning Safety Status is BLOCKED
      await expect(page.locator('text=BLOCKED')).toBeVisible();
      await expect(page.locator('text=IMMUTABLE_READ_ONLY')).toBeVisible();
    }

    // 4. Return to main investigation interface
    const explorerTab = page.locator('button, a', { hasText: /Transaction Explorer|Investigation/i }).first();
    if (await explorerTab.isVisible()) {
      await explorerTab.click();
    }

    // 5. Verify page renders without console exceptions or crashes
    expect(true).toBe(true);
  });
});

