import type React from 'react';
import { ShieldAlert, BookOpen, AlertCircle, FileText, CheckCircle2 } from 'lucide-react';

export const MethodologyBoundaries: React.FC = () => {
  return (
    <div className="methodology-boundaries">
      {/* Top Banner Notice */}
      <div className="safety-banner danger" style={{ marginBottom: 24 }}>
        <ShieldAlert size={22} color="#f43f5e" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div className="safety-title" style={{ color: '#f43f5e', fontSize: 14 }}>
            ACADEMIC INTEGRITY & REGULATORY NON-ENFORCEMENT DISCLOSURE
          </div>
          <div className="safety-desc">
            This platform is an academic research demonstration developed for undergraduate thesis defense and peer review.
            Under no circumstances should the models, predictions, or explanations rendered on this platform be utilized
            for autonomous account blocking, financial surveillance, legal prosecution, or commercial anti-money laundering (AML) decisions.
          </div>
        </div>
      </div>

      {/* Disclosures Grid */}
      <div className="grid-2">
        {/* Disclosure 1: Static GAT Architecture */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <BookOpen size={18} color="#38bdf8" />
              1. Architectural Classification: Static Time-Encoded GAT
            </div>
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>
            <p style={{ marginBottom: 10 }}>
              Although designated as <strong>ETGNN</strong>, the model operates as a static 2-layer Graph Attention Network (GAT)
              augmented with a 16-dimensional sinusoidal time-step vector concatenated to input features.
            </p>
            <p style={{ marginBottom: 10 }}>
              The model does <em>not</em> implement continuous dynamic temporal graph learning (such as Temporal Graph Networks (TGN)
              or Memory-based GNNs). Because all edges in the Elliptic dataset exist strictly within isolated 2-week time steps,
              no inter-step temporal message passing occurs.
            </p>
            <p>
              In multi-seed ablation, ETGNN and Matched GAT (which receives zeroed time vectors) performed identically (~0.551 vs ~0.550 validation AUPRC),
              proving that sinusoidal time encoding provided no dynamic temporal adaptation.
            </p>
          </div>
        </div>

        {/* Disclosure 2: Baseline Dominance */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <FileText size={18} color="#fbbf24" />
              2. Baseline Dominance & Topological Noise
            </div>
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>
            <p style={{ marginBottom: 10 }}>
              On held-out chronological test steps 41–49, <strong>Random Forest</strong> achieved an AUPRC of <strong>0.6323</strong>,
              substantially outperforming ETGNN (<strong>0.2250</strong>) and all evaluated GNN baselines (GCN: 0.1306, GAT: 0.1155, GraphSAGE: 0.2427).
            </p>
              <p style={{ marginBottom: 10 }}>
                Furthermore, ablation experiments showed that deleting all graph edges from the original local ETGNN increased validation AUPRC by <strong>+0.2468</strong> (from 0.5401 to 0.7869).
                In Phase 4 (40 controlled trials), introducing an explicit ego-residual projection + LayerNorm (<code>ResidualETGNNClassifier</code>) stabilized 93-feature validation AUPRC at <strong>0.7758 ± 0.0111</strong> versus <strong>0.7252 ± 0.0420</strong> on degree-preserving rewired graphs (unadjusted p = 0.0245, Holm-adjusted p = 0.1223 — <em>exploratory stabilization evidence</em>), while matching the 93f Deep MLP (<strong>0.7616 ± 0.0563</strong>, p = 0.6503). On 165 features, the Deep MLP (<strong>0.7189 ± 0.0016</strong>) significantly outperformed the True-Graph Residual ETGNN (<strong>0.5812 ± 0.0639</strong>, Holm-adjusted p = 0.0498).
              </p>
            <p>
              This demonstrates the <em>Topological Noise phenomenon</em>: in transaction graphs without transaction amount/type filtering,
              message passing forces illicit and licit feature mixing, which degrades model representations below tabular tree ensembles.
            </p>
          </div>
        </div>

        {/* Disclosure 3: Explainability Limits */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <AlertCircle size={18} color="#f43f5e" />
              3. Explainability: Prediction Sensitivity ≠ Causal Ground Truth
            </div>
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>
            <p style={{ marginBottom: 10 }}>
              GNNExplainer edge importance values identify which edges in the local computation graph cause the largest drop
              in model prediction probability when pruned.
            </p>
            <p style={{ marginBottom: 10 }}>
              <strong>Edge masks reflect model gradient sensitivity under edge removal, not real-world criminal conspiracy.</strong>
              High edge importance between two transactions does not constitute legal or empirical proof of money laundering or collusion.
            </p>
            <p>
              Additionally, hypothesis C1 (temporal explanation sensitivity) was statistically inconclusive (95% CIs cross zero),
              and C2 evaluates sensitivity differences under matched conditions rather than causal explanation correctness.
              In Phase H (N=30 multi-edge validation nodes on the frozen 93f <code>ResidualETGNNClassifier</code>), Exact Leave-One-Edge-Out (LOO) discrete occlusion captured <strong>3.65x higher sensitivity</strong> (0.0547 vs 0.0150 |Δp|) and executed <strong>19.0x faster</strong> (13.7 ms vs 260.5 ms/node) than continuous mask relaxation (<code>GNNExplainer</code>), whose gradient masks were near-uniform (normalized entropy 0.9378) because the ego-residual pathway bounds edge perturbation gradients.
            </p>
          </div>
        </div>

        {/* Disclosure 4: Dataset & Experimental Provenance */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <CheckCircle2 size={18} color="#10b981" />
              4. Experimental Provenance & Reproducibility
            </div>
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>
            <p style={{ marginBottom: 10 }}>
              All metrics, models, checkpoints, and tables presented across this platform are loaded directly from frozen repository artifacts:
            </p>
            <ul style={{ paddingLeft: 20, marginBottom: 10 }}>
              <li><code>results/tables/final_test_results.csv</code></li>
              <li><code>results/tables/c1_response_summary.csv</code></li>
              <li><code>results/tables/c2_registers_bootstrap_summary.csv</code></li>
              <li><code>results/checkpoints/*.pt, *.joblib, *.json</code></li>
              <li><code>results/explanations/connected_*.json</code></li>
            </ul>
            <p>
              No checkpoints have been retrained, no results have been fabricated, and the official thesis findings are preserved without embellishment.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
