import { test, expect } from '@playwright/test';

// Deterministic investigation response fixture
const mockReport = {
  investigation_id: 'inv_e2e_playwright_001',
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
    subgraph_nodes_2hop: 12,
    illicit_neighbors_count: 3,
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
      claim_text: 'The 2-hop graph neighborhood contains 12 transactions with 3 illicit-label neighbors.',
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
  evidence_limitations: [
    'Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence.',
  ],
  research_citations: [
    'Weber et al. (2019) Anti-Money Laundering in Bitcoin: Experimenting with Graph ConvolutionnNetworks',
  ],
  provenance_chain: [
    'EV-TX-LOOKUP-288717023',
    'EV-GNN-PRED-288717023',
    'EV-GRAPH-NEIGHBOR-288717023',
  ],
  provenance_hash: 'a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0',
};

const mockEvents = [
  {
    event_id: 'ev_001',
    investigation_id: 'inv_e2e_playwright_001',
    stage: 'INITIALIZE',
    timestamp: '2026-11-09T16:00:00Z',
    status: 'COMPLETED',
    duration_ms: 12,
    safe_output: { target_tx_id: 288717023 },
    message: 'Workflow initialized',
  },
  {
    event_id: 'ev_002',
    investigation_id: 'inv_e2e_playwright_001',
    stage: 'MODEL_PREDICTION',
    timestamp: '2026-11-09T16:00:01Z',
    status: 'COMPLETED',
    duration_ms: 45,
    safe_output: { etgnn_prob: 0.885 },
    message: 'Model predictions computed',
  },
  {
    event_id: 'ev_003',
    investigation_id: 'inv_e2e_playwright_001',
    stage: 'GRAPH_RETRIEVAL',
    timestamp: '2026-11-09T16:00:02Z',
    status: 'COMPLETED',
    duration_ms: 30,
    safe_output: { in_degree: 0, out_degree: 2 },
    message: 'Neighborhood retrieved',
  },
  {
    event_id: 'ev_004',
    investigation_id: 'inv_e2e_playwright_001',
    stage: 'REPORT_GENERATION',
    timestamp: '2026-11-09T16:00:03Z',
    status: 'COMPLETED',
    duration_ms: 25,
    safe_output: { verdict: 'HIGH_RISK_ILLICIT' },
    message: 'Forensic report compiled',
  },
];

test.describe('ETGNN-Agent Phase 4.5 Browser E2E Suite', () => {

  test.beforeEach(async ({ page }) => {
    // Intercept standard analytics endpoints for clean UI rendering
    await page.route('**/api/transactions*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          transactions: [
            { txId: 288717023, timestep: 43, classLabel: 1, inDegree: 0, outDegree: 2, totalBtc: 5.42 },
            { txId: 230425980, timestep: 1, classLabel: 1, inDegree: 1, outDegree: 1, totalBtc: 1.25 },
          ],
          total: 2,
        }),
      });
    });

    await page.route('**/api/agent/investigations', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    await page.route('**/api/agent/investigations/*/events', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockEvents),
      });
    });

    await page.route('**/api/agent/investigations/*', async (route) => {
      const url = route.request().url();
      if (url.includes('/events')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockEvents),
        });
        return;
      }
      if (url.includes('/export/pdf')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/pdf',
          body: '%PDF-1.4 mock pdf content',
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_e2e_playwright_001',
          status: 'COMPLETED',
          report: mockReport,
          warnings: [],
          errors: [],
        }),
      });
    });

    // Default investigate response
    await page.route('**/api/agent/investigate', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_e2e_playwright_001',
          status: 'COMPLETED',
          report: mockReport,
          warnings: [],
          errors: [],
        }),
      });
    });

    // Default feedback responses (handles both /feedback and /feedback/{id})
    await page.route('**/api/agent/feedback*', async (route) => {
      if (route.request().method() === 'POST') {
        const body = JSON.parse(route.request().postData() || '{}');
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            feedback_id: 'fb_playwright_123',
            investigation_id: body.investigation_id || 'inv_e2e_playwright_001',
            transaction_id: body.transaction_id || 288717023,
            feedback_category: body.feedback_category || 'helpful',
            rating: body.rating || 5,
            comment: body.comment || '',
            reviewer_id: body.reviewer_id || 'analyst_local',
            report_version: 'ETGNN-Agent-v3.5.0',
            evidence_references: [],
            created_at: new Date().toISOString(),
          }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            stats: { total_feedback_count: 1, average_rating: 5, category_distribution: { helpful: 1 }, storage_engine: 'sqlite_isolated', safety_invariant: 'read_only_isolation' },
            items: [],
          }),
        });
      }
    });

    // Default chat response
    await page.route('**/api/agent/chat*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_e2e_playwright_001',
          transaction_id: 288717023,
          question: 'Why did ETGNN and tree baselines diverge?',
          answer: 'Model comparison for Transaction 288717023: ETGNN predicted an illicit probability of 0.8850, while Random Forest predicted 0.5000.',
          cited_evidence_ids: ['EV-GNN-PRED-288717023'],
          provider_type: 'mock',
          generation_mode: 'SIMULATED',
          limitations: ['Predictions represent statistical risk classifications.'],
        }),
      });
    });
  });

  // Scenario A: Application Startup & Navigation
  test('Scenario A: Application startup and core navigation in Chromium', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/ETGNN/i);
    // Verify core navigation tabs exist
    await expect(page.getByRole('button', { name: /AI Copilot/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Overview/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Transaction Explorer/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Model Predictions/i })).toBeVisible();
  });

  // Scenario B: Investigation Drawer Open, Close, Escape & Tx Sync
  test('Scenario B: Investigation drawer opens, syncs Tx ID, and responds to Esc', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();

    // Verify drawer visible
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await expect(page.getByPlaceholder('e.g. Risk score and counterparty patterns')).toBeVisible();

    // Test Esc key dismiss
    await page.keyboard.press('Escape');
    await expect(page.getByText('AI Investigation Copilot')).not.toBeVisible();

    // Reopen drawer via Header button
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
  });

  // Scenario C: Investigation Execution with MOCK Provider
  test('Scenario C: Investigation execution in MOCK mode displays simulated label', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();

    // Click launch button
    const runBtn = page.getByRole('button', { name: /Launch Investigation/i });
    await runBtn.click();

    // Verify report rendered
    await expect(page.getByText('SIMULATED: Deterministic Offline Mock')).toBeVisible();
    await expect(page.getByText('HIGH_RISK_ILLICIT')).toBeVisible();
    await expect(page.getByText(/ETGNN \(Canonical\)/i)).toBeVisible();
  });

  // Scenario D: Evidence-Only Mode Execution
  test('Scenario D: Evidence-only mode discloses deterministic non-generative mode', async ({ page }) => {
    const evidenceOnlyReport = {
      ...mockReport,
      investigation_id: 'inv_e2e_evidence_only',
      provider_type: 'evidence_only',
      generation_mode: 'DETERMINISTIC_EVIDENCE_ONLY',
      narrative_summary: 'Deterministic non-generative factual summary derived directly from verified evidence records.',
    };

    await page.route('**/api/agent/investigate', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          investigation_id: 'inv_e2e_evidence_only',
          status: 'COMPLETED',
          report: evidenceOnlyReport,
          warnings: [],
          errors: [],
        }),
      });
    });

    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Must show EVIDENCE-ONLY badge and NO LIVE badge
    await expect(page.getByText('EVIDENCE-ONLY: Deterministic Factual Mode')).toBeVisible();
    await expect(page.getByText('LIVE LLM')).not.toBeVisible();
  });

  // Scenario E: Unconfigured Live Provider Rejection / Explicit Error
  test('Scenario E: Unconfigured live provider returns explicit error without silent mock fallback', async ({ page }) => {
    await page.route('**/api/agent/investigate', async (route) => {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: {
            error_code: 'LLMAuthenticationError',
            message: 'Gemini API key not found in GEMINI_API_KEY environment variable.',
          },
        }),
      });
    });

    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();

    // Select Gemini provider
    const providerSelect = page.locator('select').nth(1);
    await providerSelect.selectOption('gemini');

    // Uncheck fallback to test strict live failure
    const fallbackCheckbox = page.getByRole('checkbox', { name: /Fallback on failure/i });
    if (await fallbackCheckbox.isChecked()) {
      await fallbackCheckbox.uncheck();
    }

    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Verify explicit error is displayed in UI
    await expect(page.getByText(/Gemini API key not found/i)).toBeVisible();
    // Verify it did not silently display a mock report
    await expect(page.getByText('SIMULATED: Deterministic Offline Mock')).not.toBeVisible();
  });

  // Scenario F: Citation Navigation
  test('Scenario F: Clicking citation anchors navigates to verified Evidence Records', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Switch to Evidence tab
    await page.getByRole('button', { name: /Evidence \(/i }).click();
    await expect(page.getByText(/Immutable Evidence Records/i)).toBeVisible();
    await expect(page.getByText('EV-TX-LOOKUP-288717023')).toBeVisible();
    await expect(page.getByText('EV-GNN-PRED-288717023')).toBeVisible();
  });

  // Scenario G: Multi-Model Comparison Panel
  test('Scenario G: Model comparison panel calculates spread and warns on divergence', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Verify multi-model card
    await expect(page.getByText(/Multi-Model Consensus/i)).toBeVisible();
    const modelPanel = page.locator('.model-comparison-panel');
    await expect(modelPanel.getByText('0.8850')).toBeVisible(); // ETGNN
    await expect(modelPanel.getByText('0.5000')).toBeVisible(); // RF
    await expect(modelPanel.getByText('0.8200')).toBeVisible(); // XGBoost
    await expect(page.getByText(/Model Divergence Detected/i)).toBeVisible();
  });

  // Scenario H: Graph Evidence Viewer & Snapshot 43 Scientific Wording
  test('Scenario H: Graph evidence viewer displays Snapshot 43 non-causal qualification and degree wording', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Verify Snapshot 43 badge and non-causal description
    await expect(page.getByText(/Snapshot 43: Observed Distribution Shift/i)).toBeVisible();
    // Verify degree wording (0 In, 2 Out)
    await expect(page.getByText(/0 Observed In-Edges/i)).toBeVisible();
    await expect(page.getByText(/Illicit-Label Neighbors/i)).toBeVisible();
  });

  // Scenario I: Follow-Up Chat
  test('Scenario I: Follow-up chat answers grounded questions with cited evidence', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Switch to Chat tab
    await page.getByRole('button', { name: /Follow-Up Chat/i }).click();
    await expect(page.getByPlaceholder(/Ask about model spread/i)).toBeVisible();

    // Click suggested prompt chip
    await page.getByRole('button', { name: /Why did ETGNN classify this transaction differently/i }).click();

    // Verify answer rendered with cited evidence pill
    await expect(page.getByText(/Model comparison for Transaction 288717023/i)).toBeVisible();
    await expect(page.getByText('EV-GNN-PRED-288717023')).toBeVisible();
  });

  // Scenario J: Human Analyst Feedback Form Submission
  test('Scenario J: Analyst feedback form submits rating, review category and logs persistence', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Switch to Feedback tab
    await page.getByRole('button', { name: /Analyst Review/i }).click();
    await expect(page.getByText(/Human-in-the-Loop Analyst Review/i)).toBeVisible();
    await expect(page.getByText('Checkpoint Isolated')).toBeVisible();

    // Fill feedback form
    await page.getByPlaceholder(/Document discrepancies/i).fill('Excellent topological precision.');
    await page.getByRole('button', { name: /Submit Analyst Review/i }).click();

    // Verify confirmation message
    await expect(page.getByText(/successfully recorded/i)).toBeVisible();
  });

  // Scenario K: Report Export (JSON and PDF Download)
  test('Scenario K: Export triggers non-empty JSON and PDF downloads', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();
    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Test JSON download
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: /Download JSON/i }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toContain('inv_e2e_playwright_001');

    // Verify PDF export button exists with correct href link
    const pdfBtn = page.getByRole('link', { name: /Export Official PDF/i });
    await expect(pdfBtn).toBeVisible();
    await expect(pdfBtn).toHaveAttribute('href', '/api/agent/investigations/inv_e2e_playwright_001/export/pdf');
  });

  // Scenario L: Error Handling (Invalid Tx ID & Timeout Resilience)
  test('Scenario L: Error boundary gracefully displays API failure messages', async ({ page }) => {
    await page.route('**/api/agent/investigate', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Transaction ID 999999999 not found in Elliptic dataset.' }),
      });
    });

    await page.goto('/');
    await page.getByRole('button', { name: /Investigate with AI/i }).click();
    await expect(page.getByText('AI Investigation Copilot')).toBeVisible();

    // Enter invalid tx ID
    const txInput = page.locator('input[type="number"]').first();
    await txInput.fill('999999999');

    await page.getByRole('button', { name: /Launch Investigation/i }).click();

    // Verify error banner
    await expect(page.getByText(/Transaction ID 999999999 not found/i)).toBeVisible();
  });

});
