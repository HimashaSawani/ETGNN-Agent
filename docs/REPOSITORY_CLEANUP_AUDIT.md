# ETGNN-Agent — Repository Cleanup & Architecture Audit Report

**Audit Date:** November 14, 2026  
**Audited Branch:** `release/clean-main` (Commit `2d695f1c26592e4e4968655d24317a47debd4068`)  
**Target Goal:** Transform the public GitHub repository into a polished, recruiter-friendly AI Engineer showcase without altering research integrity, runtime dependencies, or Git history.  
**Execution Safety Rule:** **PLAN ONLY** — No files will be moved, deleted, committed, or pushed without explicit user approval.

---

## 1. Executive Summary & Inventory Overview

A complete file-by-file audit of the 1,530 tracked files on `release/clean-main` reveals that while the core application (`src/`, `web/`, `tests/`) is well-modularized, **the repository root directory contains 80 files**, of which **72 are cluttering development reports, thesis Word documents, viva presentation slides, Colab notebooks, and evaluation ZIP bundles**.

### Tracked File Breakdown (1,530 Total Files)
| Category | File Count | Current Status & Description |
| :--- | :---: | :--- |
| **1. Application Source Code** | **186 files** | Core backend (`src/`), PyTorch GNNs, LangGraph agent, and React frontend (`web/frontend/src/`) |
| **2. Configuration & Dependencies** | **14 files** | `.gitignore`, `.env.example`, `requirements.txt`, `pytest.ini`, `package.json`, `vite.config.ts`, `configs/` |
| **3. Tests & Automation Scripts** | **60 files** | Backend unit/regression tests (`tests/`), automation scripts (`scripts/`), Playwright E2E (`e2e/`) |
| **4. Core Portfolio Docs & Screenshots** | **14 files** | `README.md`, `LICENSE`, `docs/screenshots/*.png` (9 images), `docs/architecture/`, verification reports |
| **5. Root Binary Clutter (ZIP/DOCX/PPTX)** | **21 files** | Historical thesis drafts, Colab zips, viva PPTX slides in repository root |
| **6. Root Internal Dev Reports (.md)** | **43 files** | Phase milestone reports, checkpoint audits, and Colab execution plans in root |
| **7. Root Colab Notebooks (.ipynb)** | **8 files** | Standalone Colab training/audit notebooks scattered in root |
| **8. Academic Thesis & Viva Archives** | **527 files** | `thesis/` and `research_extension/` chapters, DOCX files, and viva packages |
| **9. Historical Experiment Runs & Logs** | **657 files** | `experiments/`, `results/`, `reports/`, and intermediate evaluation CSV/JSONs |

---

## 2. GitHub Presentation & Default Branch Findings

* **Current Default Branch on GitHub:** `main` (tracked at commit `8c85c401222eb52ae6415a41f5f76ceb28812e50` / `15aec8d50e6b660f88bb63c86f507bca58ed3a1f`).
* **Clean Portfolio Release Branch:** `release/clean-main` (currently at `2d695f1c26592e4e4968655d24317a47debd4068`).
* **Landing Page Visibility Issue:** Because GitHub displays the default branch by default, visitors landing on `https://github.com/HimashaSawani/ETGNN-Fraud-Detection` currently see the original uncleaned `main` branch rather than the polished `release/clean-main` branch with the new enterprise README and screenshots.
* **Remediation Recommendation:** After completing repository cleanup on `release/clean-main`, the user should set `release/clean-main` as the default branch in GitHub Repository Settings (`Settings -> Branches -> Default branch`).

---

## 3. Detailed File Classification & Action Plan

### A. Root Clutter Files Recommended for Archival / Relocation (64 Files)

None of these files are required for application runtime, FastAPI endpoints, or automated test discovery. However, to preserve academic provenance, **valuable thesis and experiment logs should be cleanly archived into `archive/` rather than permanently destroyed**.

| File Group | Example Files | Current Location | Recommended Action & Destination |
| :--- | :--- | :--- | :--- |
| **Colab Notebooks (8 files)** | `ETGNN_Main_C1_C2_Colab.ipynb`, `ETGNN_Technical_Pilot_Colab.ipynb`, etc. | Root (`/`) | **MOVE** to `notebooks/colab_training/` |
| **Thesis Drafts & Documents (7 files)** | `ETGNN_Final_Thesis_Novelty_Revised.docx`, `ETGNN_FINAL_REPORT_SUBMISSION_CLEAN.docx`, etc. | Root (`/`) | **ARCHIVE** to `archive/thesis_drafts/` |
| **Presentation Slides (2 files)** | `ETGNN_NIBM_FINAL_VIVA_PRESENTATION.pptx`, `test_presentation.pptx` | Root (`/`) | **ARCHIVE** to `archive/presentations/` |
| **Internal Markdown Reports (43 files)** | `COLAB_EXECUTION_EVIDENCE.md`, `PHASE13_EXPERIMENT_DESIGN.md`, `CHECKPOINT_AND_LOG_VERIFICATION.md`, etc. | Root (`/`) | **ARCHIVE** to `archive/research_milestone_reports/` |
| **Evaluation ZIP Bundles (12 files)** | `ETGNN_Audit_Corrected_Results.zip`, `ETGNN_Colab_Bundle.zip`, `technical_pilot_results.zip`, etc. | Root (`/`) | **ARCHIVE** to `archive/legacy_bundles/` (or remove if redundant with `results/`) |

---

### B. Core Files to Retain at Repository Root (8 Files)

These 8 files provide a clean, professional, standardized GitHub root:

```text
D:\ETGNN-Fraud-Detection\
├── README.md               <-- Professional portfolio presentation with real screenshots
├── LICENSE                 <-- MIT open-source license
├── requirements.txt        <-- Python runtime dependencies
├── pytest.ini              <-- Test suite configuration
├── .gitignore              <-- Hardened build, cache, and large file exclusion rules
├── .env.example            <-- Documented environment variables template
├── manifest.sha256         <-- Cryptographic verification manifest
└── research_log.md         <-- Clean top-level research timeline log
```

---

## 4. Proposed Clean Directory Structure

```text
ETGNN-Fraud-Detection/
├── README.md                           # Main recruiter-friendly portfolio presentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Test configuration
├── .gitignore                          # Clean gitignore
├── .env.example                        # Environment variables template
├── manifest.sha256                     # Checkpoint and data manifest
├── research_log.md                     # High-level research log
│
├── src/                                # Core Python ML & Agent Subsystems
│   ├── models/                         # PyTorch GNNs (ETGNN, Residual, Matched GAT)
│   ├── agent/                          # LangGraph state graph, tools, RAG, XAI, validation
│   ├── data/                           # Elliptic dataset parsing and preprocessing
│   └── web/                            # FastAPI REST application, endpoints, data loaders
│
├── web/                                # Frontend Presentation Layer
│   └── frontend/                       # React 18 + Vite + TypeScript application
│       ├── src/                        # Enterprise components, visualizers, state
│       └── e2e/                        # Playwright browser integration tests
│
├── tests/                              # Comprehensive Python Automated Tests
│   ├── test_agent_evidence_security.py # Prompt injection & claim security tests
│   ├── test_claim_and_report_validation.py # Evidence and citation validation
│   └── test_evidence_schemas.py        # Schema immutability & serialization
│
├── scripts/                            # Operational & Benchmarking Scripts
│   ├── run_real_playwright_e2e.ps1     # Automated end-to-end browser test runner
│   ├── benchmark_models.py             # Out-of-time evaluation pipeline
│   └── verify_environment.py          # Pre-flight environment check
│
├── configs/                            # System & Model Configuration Files
│   ├── experiment_config.yaml          # Model hyperparameters & split definitions
│   └── phase5_frozen_manifest.json     # Frozen research model references
│
├── notebooks/                          # Interactive Jupyter & Google Colab Notebooks
│   ├── colab_training/                 # Historical Colab GPU training runbooks
│   └── exploration/                    # Data exploration & XAI inspection notebooks
│
├── docs/                               # Core Documentation & Visual Evidence
│   ├── screenshots/                    # 9 Verified high-res screenshots for README
│   ├── architecture/                   # System and agent workflow specifications
│   ├── LIVE_DEMO_VERIFICATION_REPORT.md # 8/8 Live demo test verification evidence
│   ├── FINAL_ENTERPRISE_UI_COMPLETION_REPORT.md # UI redesign verification
│   └── REPOSITORY_CLEANUP_AUDIT.md     # This repository cleanup plan
│
└── archive/                            # Organized Historical Academic Artifacts
    ├── thesis_drafts/                  # Historical Word & PDF thesis submissions
    ├── presentations/                  # Viva presentation slides (PPTX/MD)
    └── research_milestone_reports/     # Phase 1-13 internal engineering reports
```

---

## 5. Dependency & Safety Analysis

Before executing any file reorganization, the following dependency checks were performed:

1. **FastAPI Application Imports (`src/web/app.py`):**
   - References only `src.*` modules and `data/processed/`.
   - **0 dependencies** on root `.docx`, `.pptx`, `.zip`, or root `.md` files.
2. **Frontend Build & Test Discovery (`web/frontend/`):**
   - Isolated inside `web/frontend/` with its own `package.json` and `playwright.config.ts`.
   - **0 dependencies** on root milestone files.
3. **Pytest Test Discovery (`tests/`):**
   - Discovers test files strictly in `tests/test_*.py`.
   - Moving root notebooks and reports will **NOT affect test discovery or execution**.
4. **README Links & Asset Paths:**
   - All 9 screenshots are stored in `docs/screenshots/` and referenced relatively (`docs/screenshots/*.png`).
   - Moving root clutter files will **NOT break any README images or badges**.

---

## 6. Safe Implementation Plan (Upon Approval)

If you approve this plan, the reorganization will be carried out safely in the isolated `release-candidate` worktree:

1. **Step 1:** Create an organized archive structure (`archive/thesis_drafts/`, `archive/presentations/`, `archive/research_milestone_reports/`, `notebooks/colab_training/`).
2. **Step 2:** Use `git mv` to relocate root reports, Word documents, PowerPoint slides, and Colab notebooks into their designated subdirectories.
3. **Step 3:** Verify that all 14 Playwright tests, backend tests, and frontend build continue to pass with 0 regressions.
4. **Step 4:** Commit the clean organization to `release-candidate` and fast-forward push to `release/clean-main`.
5. **Step 5:** Recommend switching the default GitHub branch to `release/clean-main` in repository settings.

*No files have been moved, deleted, or committed.* Ready for your review and approval.
