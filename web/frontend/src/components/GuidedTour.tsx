import type React from 'react';
import { useState } from 'react';
import {
  Compass,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  BookOpen,
  CheckCircle2,
  TrendingUp,
} from 'lucide-react';

interface Props {
  onSelectTab: (tabId: string) => void;
}

interface TourStep {
  stepNumber: number;
  title: string;
  subtitle: string;
  duration: string;
  themeColor: string;
  targetTab: string;
  keyPoints: string[];
  metrics: Array<{ label: string; value: string; note: string; color?: string }>;
  takeaway: string;
  deepLinkText: string;
}

export const GuidedTour: React.FC<Props> = ({ onSelectTab }) => {
  const [currentStep, setCurrentStep] = useState<number>(1);

  const steps: TourStep[] = [
    {
      stepNumber: 1,
      title: 'Research Overview & The Fraud Problem',
      subtitle: 'Evaluating Graph Attention Networks for Financial Crime Detection',
      duration: '1.0 min',
      themeColor: '#38bdf8',
      targetTab: 'overview',
      keyPoints: [
        'Massive Bitcoin transaction network spanning 203,769 transactions and 234,355 payment flows.',
        'Strict chronological split across 49 discrete time-steps: Training (steps 1..32), Tuning (steps 33..40), and Held-Out Test (steps 41..49).',
        'Realistic extreme class imbalance: 5.25% illicit base rate in test set vs 10.47% in validation set.',
        'Out-of-time evaluation uncovers real-world macro-regime degradation that standard cross-validation hides.',
      ],
      metrics: [
        { label: 'Total Transactions', value: '203,769', note: 'Elliptic Bitcoin Dataset' },
        { label: 'Test illicit Base Rate', value: '5.25%', note: '524 illicit vs 9,449 licit nodes', color: '#f43f5e' },
        { label: 'Chronological Steps', value: '49 steps', note: 'Strict out-of-time split' },
      ],
      takeaway:
        'Standard cross-validation leaks future graph state. Strict chronological splitting is required to measure true financial crime detection efficacy.',
      deepLinkText: 'Explore Research Overview',
    },
    {
      stepNumber: 2,
      title: 'Dataset & Transaction Graph Topology',
      subtitle: 'The GNN Heterophily Dilemma in Financial Payment Networks',
      duration: '1.5 min',
      themeColor: '#fbbf24',
      targetTab: 'explorer',
      keyPoints: [
        'Hypothesis: Temporal Graph Attention Networks (ETGNN) will outperform tabular models by aggregating transaction neighborhood signals.',
        'Empirical Refutation: Pure tabular Random Forest achieves 0.6323 Average Precision on held-out test data, while 165-feature ETGNN achieves only 0.2250 AP (-0.4073 deficit).',
        'Ablation Anomaly: Stripping all graph edges from a 93-feature local model causes performance to surge from 0.2344 to 0.4812 AP (+0.2468 gain).',
        'Diagnosis: Financial transaction graphs are structurally heterophilic. Money launderers mix with legitimate addresses, causing standard GNN message passing to average away distinctive fraud signatures.',
      ],
      metrics: [
        { label: 'Random Forest (Tabular)', value: '0.6323 AP', note: 'Top overall model', color: '#fbbf24' },
        { label: 'ETGNN (Static Time-GAT)', value: '0.2250 AP', note: 'Suffers heterophilic dilution', color: '#38bdf8' },
        { label: 'Edge-Removal Gain', value: '+0.2468 AP', note: 'Ablating edges boosts local GNN to 0.4812', color: '#34d399' },
      ],
      takeaway:
        'Raw transaction edges connect licit and illicit accounts indiscriminately. Isotropic graph smoothing degrades detection unless ego-features are protected.',
      deepLinkText: 'Browse Transaction Explorer',
    },
    {
      stepNumber: 3,
      title: 'Model Predictions & Ego-Residual Solution',
      subtitle: 'Residual ETGNN Preserves Local Features While Learning From Neighbors',
      duration: '1.5 min',
      themeColor: '#34d399',
      targetTab: 'prediction',
      keyPoints: [
        'Architectural Solution: Residual ETGNN introduces explicit linear skip-projections across each GAT layer, allowing the ego-node to bypass neighborhood smoothing.',
        'Validation & Test Gains: Residual ETGNN achieves 0.7758 ± 0.0111 validation AP and 0.4223 held-out test AP (Seed 42), outperforming standard ETGNN (0.2250) by nearly 20 percentage points.',
        'Statistical Rigor: 5-seed factorial validation confirms statistically significant stabilization over degree-preserving rewired graphs (+0.0506, p_unadj=0.0245).',
        'Analytical Ego-Protection Bound: Guarantees that when candidate edges vanish entirely, the maximum output probability shift is bounded by Δ_empty ≤ 0.0773.',
      ],
      metrics: [
        { label: 'Residual ETGNN (93f)', value: '0.4223 AP', note: '+0.1973 over standard ETGNN (0.2250)', color: '#34d399' },
        { label: '5-Seed Validation AP', value: '0.7758', note: '± 0.0111 across seeds 42..46' },
        { label: 'Ego Protection Bound', value: 'Δ ≤ 0.0773', note: 'Theoretical resistance to edge loss', color: '#38bdf8' },
      ],
      takeaway:
        'Ego-residual skip projections resolve the GNN heterophily dilemma, preserving vital local transaction attributes while retaining graph learning capabilities.',
      deepLinkText: 'Test Live Model Predictions',
    },
    {
      stepNumber: 4,
      title: 'Model Comparisons & Benchmark Results',
      subtitle: 'Standardized Evaluation Across 11 Model Configurations',
      duration: '1.0 min',
      themeColor: '#38bdf8',
      targetTab: 'benchmarks',
      keyPoints: [
        'Standardized metric reporting: Average Precision (AP) is the primary ranking metric, complemented by Trapezoidal PR-AUC, ROC-AUC, Precision, Recall, and F1.',
        'Non-graph tabular dominance: Random Forest (0.6323 AP) and XGBoost (0.5472 AP) remain the strongest performers on held-out test data.',
        'GNN Baselines: Standard GCN (0.1306 AP) and standard GAT (0.1155 AP) underperform drastically, while GraphSAGE (0.2427 AP) retains partial signal via root concatenation.',
        'All reported metrics are loaded directly from verified experimental CSV outputs.',
      ],
      metrics: [
        { label: 'Random Forest', value: '0.6323 AP', note: 'F1: 0.6319 | ROC-AUC: 0.8735', color: '#fbbf24' },
        { label: 'XGBoost', value: '0.5472 AP', note: 'F1: 0.3849 | ROC-AUC: 0.8074' },
        { label: 'Residual ETGNN', value: '0.4223 AP', note: 'Top GNN architecture', color: '#34d399' },
      ],
      takeaway:
        'In financial transaction surveillance, tabular baseline ensembles must always be benchmarked against GNNs to ensure graph additions yield defensible value.',
      deepLinkText: 'Open Benchmark Comparison Table',
    },
    {
      stepNumber: 5,
      title: 'Explainability & Evidence Verification',
      subtitle: 'Counterfactual Auditing of Model Explanations',
      duration: '1.0 min',
      themeColor: '#c084fc',
      targetTab: 'c1c2',
      keyPoints: [
        'Counterfactual sensitivity testing: Deleting GNNExplainer top edges yielded inconclusive differences between ETGNN and zero-time Matched GAT (Hypothesis C1 95% CIs cross zero).',
        'Error vs correct attribution: Matched paired bootstrap analysis across 8 explanation registers on 420 matched pairs also showed 95% CIs crossing zero (Hypothesis C2).',
        'Fidelity verification: Leave-One-Out (LOO) edge attribution demonstrated higher fidelity than random edge deletion, but explanation masks exhibit high entropy in sparse regimes.',
        'Scientific disclosure: Post-hoc explanation masks highlight model sensitivity, but cannot be treated as causal proof of illicit behavior.',
      ],
      metrics: [
        { label: 'Hypothesis C1', value: 'Inconclusive', note: '95% CIs encompass zero', color: '#fbbf24' },
        { label: 'C2 Matched Pairs', value: '420 pairs', note: '8 registers tested via bootstrap' },
        { label: 'Case Study Exemplars', value: '1,500', note: 'Preserved research artifacts' },
      ],
      takeaway:
        'Standard post-hoc explanation masks are descriptive sensitivity measures rather than causal ground truth in high-stakes financial crime detection.',
      deepLinkText: 'Review Explanation Evidence',
    },
    {
      stepNumber: 6,
      title: 'Temporal Analysis & Trade-off Conclusions',
      subtitle: 'Candidate-Edge Censorship and 21 Prospective Mitigation Strategies',
      duration: '1.0 min',
      themeColor: '#f43f5e',
      targetTab: 'temporal',
      keyPoints: [
        'Topological Censorship Shock: Across the step 43 darknet marketplace shutdown, isolated illicit transactions surged from 13.52% to 76.92% (a 63.4 percentage-point jump).',
        'Explanation-guided mitigations fail: Explanation-based pruning strategies (M1a..M3b) underperformed random edge controls (e.g. M1a 0.2721 vs C1a 0.2915 AP, Holm p=0.0140).',
        'Architectural Controls: A predicted-risk cascade (Strategy 13_C2b, 0.4808 AP) and a 100% local veto (Strategy 19_C3c, 0.4442 AP) provide practical stabilization.',
        'Mandatory Operational Trade-Off: In post-shift data, Strategy 13_C2b loses 2 of 5 true positives vs default baseline (-165 false positives); Strategy 19_C3c loses 3 of 5 true positives vs default (-284 false positives) and 9 of 11 true positives vs standalone local model.',
      ],
      metrics: [
        { label: 'Pre-Shift Isolated Illicit', value: '13.52%', note: 'Steps 41..42', color: '#38bdf8' },
        { label: 'Post-Shift Isolated Illicit', value: '76.92%', note: 'Steps 43..49 (+63.4% jump)', color: '#f43f5e' },
        { label: 'Top Cascade 13_C2b', value: '0.4808 AP', note: 'Discloses -2 post-shift TP trade-off', color: '#34d399' },
      ],
      takeaway:
        'Post-shock performance degradation in transaction GNNs is driven by structural edge censorship. Cascades stabilize precision at the cost of true-positive loss.',
      deepLinkText: 'Explore Temporal Analysis',
    },
  ];

  const current = steps[currentStep - 1];

  return (
    <div className="guided-tour-view">
      {/* Tour Header Banner */}
      <div
        className="card"
        style={{
          marginBottom: 20,
          background: 'linear-gradient(135deg, #0b1222 0%, #171d33 100%)',
          border: '1px solid #334155',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div
              style={{
                background: 'rgba(56, 189, 248, 0.15)',
                padding: 10,
                borderRadius: 10,
                border: '1px solid rgba(56, 189, 248, 0.35)',
              }}
            >
              <Compass size={24} color="#38bdf8" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="badge badge-dataset" style={{ fontWeight: 700 }}>
                  INTERACTIVE GUIDED TOUR
                </span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Target Pace: 5–8 minutes</span>
              </div>
              <h1 style={{ fontSize: 20, fontWeight: 800, color: '#f8fafc', margin: '4px 0 0 0' }}>
                ETGNN Research Walkthrough & Findings Summary
              </h1>
            </div>
          </div>

          {/* Stepper Buttons */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {steps.map((s) => (
              <button
                key={s.stepNumber}
                onClick={() => setCurrentStep(s.stepNumber)}
                style={{
                  background: currentStep === s.stepNumber ? s.themeColor : '#1e293b',
                  color: currentStep === s.stepNumber ? '#090d16' : '#94a3b8',
                  border: 'none',
                  borderRadius: 6,
                  padding: '6px 12px',
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                Part {s.stepNumber}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Slide Panel */}
      <div className="card" style={{ marginBottom: 20, borderTop: `4px solid ${current.themeColor}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span className="badge" style={{ background: `${current.themeColor}22`, color: current.themeColor, fontWeight: 700 }}>
                PART {current.stepNumber} OF 6
              </span>
              <span style={{ fontSize: 12, color: '#64748b' }}>Estimated: {current.duration}</span>
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 800, color: '#f8fafc', margin: '4px 0' }}>
              {current.title}
            </h2>
            <div style={{ fontSize: 13, color: '#94a3b8' }}>{current.subtitle}</div>
          </div>

          <button
            className="action-btn"
            onClick={() => onSelectTab(current.targetTab)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: '#1e293b',
              color: '#38bdf8',
              border: '1px solid #334155',
              padding: '7px 14px',
              fontSize: 12,
            }}
          >
            <span>{current.deepLinkText}</span>
            <ExternalLink size={14} />
          </button>
        </div>

        {/* Primary Metrics Grid */}
        <div className="grid-3" style={{ marginBottom: 20 }}>
          {current.metrics.map((m, idx) => (
            <div key={idx} className="stat-card" style={{ borderColor: m.color ? `${m.color}55` : undefined }}>
              <div className="stat-label">{m.label}</div>
              <div className="stat-value" style={{ color: m.color || '#f8fafc' }}>
                {m.value}
              </div>
              <div className="stat-subtext">{m.note}</div>
            </div>
          ))}
        </div>

        {/* Key Presentation Points */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BookOpen size={16} color={current.themeColor} />
            Key Insights & Evidence
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {current.keyPoints.map((point, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 12,
                  background: '#0b1120',
                  padding: '12px 16px',
                  borderRadius: 8,
                  border: '1px solid #1e293b',
                }}
              >
                <div style={{ color: current.themeColor, fontWeight: 700, fontSize: 13, marginTop: 2 }}>
                  {idx + 1}.
                </div>
                <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.55 }}>{point}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Core Takeaway Banner */}
        <div
          style={{
            background: `${current.themeColor}12`,
            border: `1px solid ${current.themeColor}44`,
            borderRadius: 8,
            padding: 16,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 12,
            marginBottom: 20,
          }}
        >
          <TrendingUp size={20} color={current.themeColor} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: current.themeColor, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Core Scientific Takeaway
            </div>
            <div style={{ fontSize: 13, color: '#f8fafc', fontWeight: 600, marginTop: 3 }}>
              {current.takeaway}
            </div>
          </div>
        </div>

        {/* Navigation Footer */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid #1e293b', paddingTop: 16 }}>
          <button
            className="action-btn"
            disabled={currentStep === 1}
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            style={{
              opacity: currentStep === 1 ? 0.35 : 1,
              cursor: currentStep === 1 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <ChevronLeft size={16} />
            Previous
          </button>

          <div style={{ fontSize: 12, color: '#64748b' }}>
            Part {currentStep} of {steps.length}
          </div>

          <button
            className="action-btn primary"
            disabled={currentStep === steps.length}
            onClick={() => setCurrentStep((prev) => Math.min(steps.length, prev + 1))}
            style={{
              opacity: currentStep === steps.length ? 0.35 : 1,
              cursor: currentStep === steps.length ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* Overview Context Card */}
      <div className="card" style={{ background: '#0b1120', border: '1px solid #1e293b' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <CheckCircle2 size={16} color="#34d399" />
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc', margin: 0 }}>
            Research Verification Standards
          </h3>
        </div>
        <div className="grid-3" style={{ gap: 12 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', background: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
            <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: 4 }}>Live Model Inference</div>
            Predictions are generated via real forward passes across PyTorch Geometric and scikit-learn models.
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', background: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
            <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: 4 }}>Standardized Metrics</div>
            Average Precision (AP) is calculated strictly on out-of-time chronological test steps 41..49.
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', background: '#0f172a', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
            <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: 4 }}>Transparent Findings</div>
            All negative results (tabular baseline superiority, inconclusive explanation hypotheses) are disclosed.
          </div>
        </div>
      </div>
    </div>
  );
};

