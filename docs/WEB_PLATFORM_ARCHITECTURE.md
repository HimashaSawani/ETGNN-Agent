# ETGNN Web Research Demonstration Platform — Architecture & System Design

## 1. System Overview

The **ETGNN Web Research Demonstration Platform** is a full-stack, research-grade demonstration system built to showcase the completed empirical research on Explainable Temporal Graph Neural Networks for Bitcoin fraud detection using the Elliptic dataset.

The system is designed with strict adherence to research integrity:
- **No Model Retraining**: All models are executed using frozen repository checkpoints.
- **No Result Fabrication**: All metrics, benchmark tables, and C1/C2 statistics are loaded directly from repository CSV/JSON artifacts.
- **Exact Mathematical Equivalence**: GNN message passing over 2-hop neighborhoods is mathematically identical to whole-graph receptive field computation.
- **Epistemic Clarity**: Clear visual boundaries distinguish ground-truth labels, model predictions, and explanation-derived edge attribution weights.

---

## 2. Architecture Diagram

```mermaid
graph TD
    Client["React Frontend (SPA)<br/>TypeScript, Vite, Vanilla CSS<br/>Port: 5173 / Production Build"]
    API["FastAPI Backend Server<br/>Uvicorn / Port: 8000"]

    subgraph Data & Checkpoints
        NodesCSV["data/processed/nodes_with_masks.csv<br/>(203,769 transactions)"]
        FeaturesNPY["data/processed/scaled_features.npy<br/>(Memory-mapped float32)"]
        GraphPT["data/processed/test_graph.pt<br/>(PyG Graph Tensor)"]
        Checkpoints["results/checkpoints/<br/>etgnn.pt, matched_gat.pt<br/>random_forest.joblib, xgboost.json"]
        TablesCSV["results/tables/<br/>final_test_results.csv<br/>c1_response_summary.csv<br/>c2_registers_bootstrap_summary.csv"]
        ExemplarsJSON["results/explanations/<br/>connected_{tp,fp,tn,fn}.json"]
        InterventionsCSV["results/main_c1_c2_results/<br/>detailed_intervention_records.csv"]
    end

    Client -->|REST API Requests| API
    API -->|O(1) Memory Cache| DataLoader["ResearchDataLoader Singleton"]
    DataLoader --> NodesCSV
    DataLoader --> FeaturesNPY
    DataLoader --> GraphPT
    DataLoader --> Checkpoints
    DataLoader --> TablesCSV
    DataLoader --> ExemplarsJSON
    DataLoader --> InterventionsCSV
```

---

## 3. Backend Architecture (`src/web/`)

### 3.1. Singleton Data Loader (`src/web/data_loader.py`)
- **Memory Efficiency**: Memory-maps `scaled_features.npy` ($203,769 \times 165$) to minimize resident memory usage.
- **Fast Lookup**: Precomputes full test-graph probabilities for ETGNN and Matched GAT on startup (~800 KB RAM), providing $O(1)$ response times (< 2 ms) for single-transaction lookups.
- **Exact Neighborhood Forward Pass**: When feature overrides are supplied in `POST /api/predict`, the loader extracts the exact 2-hop neighborhood using `torch_geometric.utils.k_hop_subgraph` and executes the PyG forward pass on CPU, matching full-graph mathematical output to zero floating-point error.
- **XAI Pipeline**:
  - Tier 1: Returns frozen exemplar JSON files (`results/explanations/connected_{tp,fp,tn,fn}.json`).
  - Tier 2: Returns frozen C1/C2 detailed intervention records (`results/main_c1_c2_results/detailed_intervention_records.csv`).
  - Tier 3: For arbitrary transactions, computes live demonstration explanations via PyG `Explainer(algorithm=GNNExplainer(epochs=35))` explicitly tagged as demonstration output.

### 3.2. REST API Endpoints (`src/web/app.py`)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | System status, dataset scale, loaded models, epistemic notice |
| `/api/models` | GET | Specifications, layer count, parameters, and checkpoints of all models |
| `/api/metrics` | GET | Official held-out test evaluation results from `final_test_results.csv` |
| `/api/transactions` | GET | Paginated transaction register with split, label, and time-step filters |
| `/api/transactions/{id}` | GET | Full details, graph degree, features summary, and multi-model predictions |
| `/api/transactions/{id}/subgraph` | GET | 1-hop / 2-hop neighborhood graph extracted for canvas visualization |
| `/api/predict` | POST | Live forward pass across ETGNN, Matched GAT, RF, and XGBoost |
| `/api/explanations/{id}` | GET | GNNExplainer top edge masks, deletion drops, and provenance |
| `/api/research/c1` | GET | C1 cohort response summary (ETGNN vs Matched GAT) |
| `/api/research/c2` | GET | C2 8-register bootstrap summary with 95% confidence intervals |
| `/api/research/ablations` | GET | Multi-seed repeat ablation and the +0.2468 edge-removal anomaly |
| `/{full_path:path}` | GET | Serves the compiled React Single Page Application (SPA) |

---

## 4. Frontend Architecture (`web/frontend/`)

### 4.1. Technology Stack
- **Framework**: React 19 + TypeScript + Vite.
- **Design System**: Bespoke academic dark-mode CSS (`index.css`) featuring slate/indigo/cyan palettes, glassmorphism, responsive grid layouts, and high-contrast typography (Inter, Outfit, JetBrains Mono).
- **Icons**: Lucide React.
- **Graph Visualization**: HTML5 Canvas rendering engine with interactive force-radial layout, arrowheads, zoom/pan controls, and node hit detection.

### 4.2. Research Dashboards & Tabs
1. **Research Overview**: Executive summary of empirical results, core findings, dataset metrics, and quick-access exemplar cards (TP, FP, TN, FN).
2. **Transaction Explorer**: Searchable, filterable data table covering all 203,769 transactions with label tags, split indicators, and quick-action inspect triggers.
3. **Fraud Prediction Arena**: Side-by-side comparison gauges for ETGNN, Matched GAT, Random Forest, and XGBoost, with real-time feature sliders.
4. **Interactive Subgraph**: Radial neighborhood canvas showing directed Bitcoin transaction flows, color-coded node labels, and a live Node Inspector.
5. **Explanation Inspector**: Edge attribution rankings, mask weight progress bars, post-intervention probability drops ($\Delta P$), and non-causal safety notices.
6. **Model Benchmarks**: Visual bar charts and full sortable table of all 11 evaluated models on held-out test steps 41–49.
7. **C1/C2 Evidence Hub**:
   - Sub-tab C1: Cohort response summary demonstrating inconclusive evidence.
   - Sub-tab C2: Interactive Forest Plot displaying 95% bootstrap confidence intervals across all 8 registers.
8. **Ablation Analysis**: In-depth examination of the +0.2468 edge removal anomaly, 3-seed multi-seed ablation, and temporal prevalence volatility.
9. **Methodology & Limitations**: Full academic disclosure of static GAT limitations, tabular baseline dominance, and non-enforcement boundaries.
