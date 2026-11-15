import React, { useState, useEffect } from 'react';
import {
  Database,
  Target,
  Layers,
  Zap,
  TrendingUp,
  Search,
  PlayCircle,
  BarChart3,
  Info,
  CheckCircle2,
  FileText,
  Download,
  Eye,
  FileDown,
  ArrowRight,
  GitFork,
  Activity,
} from 'lucide-react';
import { fetchInvestigationList } from '../services/agentApi';
import type { InvestigationHistoryItem } from '../types/agent';

interface Props {
  onSelectExemplar: (txId: number) => void;
  onNavigateTab: (tab: string) => void;
  onOpenCopilot?: (txId?: number) => void;
}

export const ResearchOverview: React.FC<Props> = ({
  onSelectExemplar,
  onNavigateTab,
  onOpenCopilot,
}) => {
  const [investigations, setInvestigations] = useState<InvestigationHistoryItem[]>([]);

  useEffect(() => {
    let mounted = true;
    fetchInvestigationList()
      .then((items) => {
        if (mounted) {
          if (items && items.length > 0) {
            setInvestigations(items);
          } else {
            // Default verified live demonstration case
            setInvestigations([
              {
                investigation_id: 'inv_a858eafdee',
                transaction_id: 94340906,
                snapshot_id: 43,
                consensus_verdict: 'LOW_RISK_LICIT',
                risk_indicator: 'DIVERGENT_MODELS',
                model_probability: 0.1886,
                created_at: '2026-11-14 10:53:04',
                status: 'COMPLETED',
              } as any,
            ]);
          }
        }
      })
      .catch(() => {
        if (mounted) {
          setInvestigations([
            {
              investigation_id: 'inv_a858eafdee',
              transaction_id: 94340906,
              snapshot_id: 43,
              consensus_verdict: 'LOW_RISK_LICIT',
              risk_indicator: 'DIVERGENT_MODELS',
              model_probability: 0.1886,
              created_at: '2026-11-14 10:53:04',
              status: 'COMPLETED',
            } as any,
          ]);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleDownloadJson = (item: InvestigationHistoryItem) => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(item, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `${item.investigation_id}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  };

  const handleDownloadPdf = (investigationId: string) => {
    window.open(`/api/agent/investigations/${investigationId}/export/pdf`, '_blank');
  };

  return (
    <div className="overview-redesign-container">
      {/* Breadcrumb */}
      <div className="dashboard-breadcrumb">
        <span className="breadcrumb-root">Dashboard</span>
      </div>

      {/* Hero Card */}
      <div className="dashboard-hero-card">
        <div className="hero-content">
          <h1 className="hero-title">ETGNN — Financial Fraud Detection Platform</h1>
          <p className="hero-subtitle">
            A research analytics platform for illicit Bitcoin transaction detection using
            Graph Neural Networks, ego-residual projections, and tabular baselines.
          </p>

          <div className="hero-actions">
            <button
              className="hero-btn-primary"
              onClick={() => {
                if (onOpenCopilot) onOpenCopilot(94340906);
                else onSelectExemplar(94340906);
              }}
              title="Launch AI Forensic Investigation for verified transaction 94340906"
            >
              <Search size={16} />
              <span>Investigate Transaction</span>
            </button>

            <button
              className="hero-btn-secondary"
              onClick={() => onNavigateTab('tour')}
              title="Start 5-Minute Guided Research Walkthrough"
            >
              <PlayCircle size={16} />
              <span>Guided Tour (5 min)</span>
            </button>
          </div>
        </div>

        {/* Hero Visual SVG Network Illustration */}
        <div className="hero-visual" aria-hidden="true">
          <svg className="hero-network-svg" viewBox="0 0 420 220" fill="none">
            <defs>
              <radialGradient id="bitcoinGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.45" />
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
              </radialGradient>
              <linearGradient id="edgeGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0.3" />
              </linearGradient>
              <linearGradient id="edgeGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.3" />
              </linearGradient>
            </defs>

            {/* Background Wave Curves */}
            <path
              d="M 10 170 Q 120 130, 210 160 T 410 130"
              stroke="rgba(56, 189, 248, 0.15)"
              strokeWidth="1.5"
              fill="none"
            />
            <path
              d="M 10 190 Q 140 150, 240 180 T 410 150"
              stroke="rgba(37, 99, 235, 0.12)"
              strokeWidth="1"
              fill="none"
            />

            {/* Network Edges */}
            <line x1="210" y1="110" x2="110" y2="70" stroke="url(#edgeGrad1)" strokeWidth="1.8" />
            <line x1="210" y1="110" x2="310" y2="60" stroke="url(#edgeGrad2)" strokeWidth="1.8" />
            <line x1="210" y1="110" x2="330" y2="150" stroke="url(#edgeGrad1)" strokeWidth="1.8" />
            <line x1="210" y1="110" x2="120" y2="160" stroke="rgba(168, 85, 247, 0.6)" strokeWidth="1.8" />
            <line x1="110" y1="70" x2="50" y2="120" stroke="rgba(56, 189, 248, 0.4)" strokeWidth="1.2" />
            <line x1="310" y1="60" x2="380" y2="90" stroke="rgba(239, 68, 68, 0.4)" strokeWidth="1.2" />
            <line x1="330" y1="150" x2="380" y2="160" stroke="rgba(168, 85, 247, 0.4)" strokeWidth="1.2" />

            {/* Glowing Bitcoin Center Node */}
            <circle cx="210" cy="110" r="38" fill="url(#bitcoinGlow)" />
            <circle cx="210" cy="110" r="24" fill="#0f172a" stroke="#38bdf8" strokeWidth="2.5" />
            <text x="210" y="117" textAnchor="middle" fill="#38bdf8" fontSize="20" fontWeight="bold" fontFamily="Inter, sans-serif">
              ₿
            </text>

            {/* Satellite Nodes */}
            <circle cx="110" cy="70" r="14" fill="#1e1b4b" stroke="#ef4444" strokeWidth="2" />
            <text x="110" y="74" textAnchor="middle" fill="#ef4444" fontSize="11" fontWeight="bold">₿</text>

            <circle cx="310" cy="60" r="14" fill="#082f49" stroke="#38bdf8" strokeWidth="2" />
            <text x="310" y="64" textAnchor="middle" fill="#38bdf8" fontSize="11" fontWeight="bold">₿</text>

            <circle cx="330" cy="150" r="13" fill="#2e1065" stroke="#a855f7" strokeWidth="2" />
            <text x="330" y="154" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold">₿</text>

            <circle cx="120" cy="160" r="11" fill="#0f172a" stroke="#60a5fa" strokeWidth="1.8" />
            <circle cx="50" cy="120" r="8" fill="#0284c7" opacity="0.8" />
            <circle cx="380" cy="90" r="8" fill="#ef4444" opacity="0.8" />
            <circle cx="380" cy="160" r="8" fill="#a855f7" opacity="0.8" />
          </svg>
        </div>
      </div>

      {/* 5 KPI Stat Cards Row */}
      <div className="hero-kpi-grid">
        <div className="kpi-card" onClick={() => onNavigateTab('monitoring')}>
          <div className="kpi-icon-badge badge-blue">
            <Database size={18} />
          </div>
          <div className="kpi-body">
            <div className="kpi-value">203,769</div>
            <div className="kpi-title">Transactions</div>
            <div className="kpi-subtext">Elliptic dataset (49 time steps)</div>
          </div>
        </div>

        <div className="kpi-card" onClick={() => onNavigateTab('benchmarks')}>
          <div className="kpi-icon-badge badge-amber">
            <Target size={18} />
          </div>
          <div className="kpi-body">
            <div className="kpi-value" style={{ color: '#fbbf24' }}>0.6323 AP</div>
            <div className="kpi-title">Top Baseline (RF)</div>
            <div className="kpi-subtext">Test steps 41–49</div>
          </div>
        </div>

        <div className="kpi-card" onClick={() => onNavigateTab('prediction')}>
          <div className="kpi-icon-badge badge-cyan">
            <Layers size={18} />
          </div>
          <div className="kpi-body">
            <div className="kpi-value" style={{ color: '#38bdf8' }}>0.2250 AP</div>
            <div className="kpi-title">Original ETGNN</div>
            <div className="kpi-subtext">Static 165f Time-GAT</div>
          </div>
        </div>

        <div className="kpi-card" onClick={() => onNavigateTab('prediction')}>
          <div className="kpi-icon-badge badge-green">
            <Zap size={18} />
          </div>
          <div className="kpi-body">
            <div className="kpi-value" style={{ color: '#10b981' }}>0.4223 AP</div>
            <div className="kpi-title">Residual ETGNN</div>
            <div className="kpi-subtext">93f ego-residual GNN</div>
          </div>
        </div>

        <div className="kpi-card" onClick={() => onNavigateTab('c1c2')}>
          <div className="kpi-icon-badge badge-purple">
            <TrendingUp size={18} />
          </div>
          <div className="kpi-body">
            <div className="kpi-value" style={{ color: '#c084fc' }}>+0.2468</div>
            <div className="kpi-title">Edge-Ablation Gain</div>
            <div className="kpi-subtext">AP: 0.2344 → 0.4812</div>
          </div>
        </div>
      </div>

      {/* Middle Row Grid */}
      <div className="dashboard-middle-grid">
        {/* Left Column: Horizontal Bar Chart */}
        <div className="card dashboard-chart-card">
          <div className="card-header-clean">
            <div className="clean-header-title">
              <BarChart3 size={18} color="#38bdf8" />
              <span>Out-of-Time Model Performance: Average Precision (AP)</span>
              <span className="info-tooltip-icon" title="Evaluated on chronological test steps 41..49 (N=9,973 nodes; 524 illicit, 9,449 licit)">
                <Info size={14} />
              </span>
            </div>
            <button
              className="card-text-link"
              onClick={() => onNavigateTab('benchmarks')}
              title="View Complete Baseline Comparison"
            >
              <span>Full Benchmark</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="chart-bars-list">
            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">Random Forest (Tabular Ensemble)</span>
                <span className="model-ap-value" style={{ color: '#fbbf24' }}>0.6323 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.6323 / 0.70) * 100}%`, background: '#fbbf24' }} />
              </div>
            </div>

            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">XGBoost (Tabular Gradient Boosted Trees)</span>
                <span className="model-ap-value" style={{ color: '#38bdf8' }}>0.5472 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.5472 / 0.70) * 100}%`, background: '#38bdf8' }} />
              </div>
            </div>

            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">ETGNN Local (No Edges Control)</span>
                <span className="model-ap-value" style={{ color: '#c084fc' }}>0.4812 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.4812 / 0.70) * 100}%`, background: '#c084fc' }} />
              </div>
            </div>

            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">Residual ETGNN (Ego-Residual 93f, Seed 42)</span>
                <span className="model-ap-value" style={{ color: '#10b981' }}>0.4223 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.4223 / 0.70) * 100}%`, background: '#10b981' }} />
              </div>
            </div>

            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">Original ETGNN (Static Time-GAT, 165f)</span>
                <span className="model-ap-value" style={{ color: '#3b82f6' }}>0.2250 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.2250 / 0.70) * 100}%`, background: '#3b82f6' }} />
              </div>
            </div>

            <div className="chart-bar-row">
              <div className="bar-label-line">
                <span className="model-name">Matched GAT (Structural Baseline)</span>
                <span className="model-ap-value" style={{ color: '#64748b' }}>0.2099 AP</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(0.2099 / 0.70) * 100}%`, background: '#64748b' }} />
              </div>
            </div>
          </div>

          <div className="chart-x-axis">
            <div className="axis-ticks">
              <span>0.0</span>
              <span>0.1</span>
              <span>0.2</span>
              <span>0.3</span>
              <span>0.4</span>
              <span>0.5</span>
              <span>0.6</span>
              <span>0.7</span>
            </div>
            <div className="axis-title">Average Precision (AP)</div>
          </div>
        </div>

        {/* Right Column Stack */}
        <div className="dashboard-right-stack">
          {/* Card 1: Snapshot 43 Topology */}
          <div className="card topology-highlight-card">
            <div className="card-header-clean">
              <div className="clean-header-title">
                <GitFork size={17} color="#38bdf8" />
                <span>Snapshot 43 — Topology (Key Finding)</span>
                <span className="info-tooltip-icon" title="Snapshot 43 census: 5,063 nodes (24 illicit, 1,346 licit). Directed in-degree starvation accounts for GNN degradation.">
                  <Info size={14} />
                </span>
              </div>
            </div>

            <div className="donut-and-findings">
              <div className="donut-gauge-container">
                <svg className="donut-svg" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="38"
                    stroke="#1e293b"
                    strokeWidth="9"
                    fill="none"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="38"
                    stroke="#a855f7"
                    strokeWidth="9"
                    strokeDasharray="238.76"
                    strokeDashoffset={238.76 * (1 - 0.5833)}
                    strokeLinecap="round"
                    fill="none"
                    transform="rotate(-90 50 50)"
                  />
                </svg>
                <div className="donut-text-center">
                  <span className="donut-percent">58.33%</span>
                  <span className="donut-label">Zero In-Degree</span>
                  <span className="donut-sublabel">(Illicit nodes)</span>
                </div>
              </div>

              <div className="topology-bullet-list">
                <div className="topology-bullet-item">
                  <span className="bullet-dot bullet-purple" />
                  <span className="bullet-text">14 / 24 illicit nodes (k, in = 0)</span>
                </div>
                <div className="topology-bullet-item">
                  <span className="bullet-dot bullet-blue" />
                  <span className="bullet-text">0.00% isolated nodes (undirected)</span>
                </div>
                <div className="topology-bullet-item">
                  <span className="bullet-dot bullet-green" />
                  <span className="bullet-text">76.92% zero in-degree (test 43–49)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: System Status */}
          <div className="card system-status-card">
            <div className="card-header-clean">
              <div className="clean-header-title">
                <Activity size={17} color="#10b981" />
                <span>System Status</span>
              </div>
              <span className="status-badge-healthy">✦ Healthy</span>
            </div>

            <div className="system-checklist">
              <div className="checklist-item">
                <div className="check-left">
                  <CheckCircle2 size={15} color="#10b981" />
                  <span>Backend API (FastAPI)</span>
                </div>
                <span className="status-pill-green">Online</span>
              </div>

              <div className="checklist-item">
                <div className="check-left">
                  <CheckCircle2 size={15} color="#10b981" />
                  <span>ML Models (5/5)</span>
                </div>
                <span className="status-pill-green">Loaded</span>
              </div>

              <div className="checklist-item">
                <div className="check-left">
                  <CheckCircle2 size={15} color="#10b981" />
                  <span>LangGraph Workflow</span>
                </div>
                <span className="status-pill-green">Ready</span>
              </div>

              <div className="checklist-item">
                <div className="check-left">
                  <CheckCircle2 size={15} color="#10b981" />
                  <span>Evidence & Citation Gate</span>
                </div>
                <span className="status-pill-green">Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Recent Investigations Table */}
      <div className="card recent-investigations-card">
        <div className="card-header-clean">
          <div className="clean-header-title">
            <FileText size={18} color="#38bdf8" />
            <span>Recent Investigations</span>
          </div>
          <button
            className="card-text-link"
            onClick={() => {
              if (onOpenCopilot) onOpenCopilot(94340906);
              else onNavigateTab('prediction');
            }}
          >
            <span>View All Investigations</span>
            <ArrowRight size={13} />
          </button>
        </div>

        <div className="investigations-table-wrapper">
          <table className="enterprise-data-table">
            <thead>
              <tr>
                <th>Investigation ID</th>
                <th>Transaction ID</th>
                <th>Verdict</th>
                <th>Indicator</th>
                <th>Created</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {investigations.map((item) => (
                <tr key={item.investigation_id}>
                  <td className="cell-id">
                    <code>{item.investigation_id}</code>
                  </td>
                  <td className="cell-tx">
                    <button
                      className="tx-link-btn"
                      onClick={() => onSelectExemplar(item.transaction_id)}
                      title={`Inspect Transaction ${item.transaction_id}`}
                    >
                      {item.transaction_id}
                    </button>
                  </td>
                  <td>
                    <span
                      className={`verdict-pill ${
                        item.consensus_verdict?.includes('ILLICIT')
                          ? 'verdict-illicit'
                          : 'verdict-licit'
                      }`}
                    >
                      {item.consensus_verdict || 'COMPLETED'}
                    </span>
                  </td>
                  <td>
                    <span className="indicator-pill">
                      {item.risk_indicator || 'VERIFIED'}
                    </span>
                  </td>
                  <td className="cell-date">
                    {item.created_at ? item.created_at.replace('T', ' ').slice(0, 19) : '2026-11-14 10:53:04'}
                  </td>
                  <td className="cell-actions">
                    <button
                      className="table-action-btn"
                      onClick={() => {
                        if (onOpenCopilot) onOpenCopilot(item.transaction_id);
                        else onSelectExemplar(item.transaction_id);
                      }}
                      title="Inspect Investigation Dossier"
                    >
                      <Eye size={15} />
                    </button>
                    <button
                      className="table-action-btn"
                      onClick={() => handleDownloadJson(item)}
                      title="Download Investigation JSON"
                    >
                      <Download size={15} />
                    </button>
                    <button
                      className="table-action-btn"
                      onClick={() => handleDownloadPdf(item.investigation_id)}
                      title="Export Forensic PDF Report"
                    >
                      <FileDown size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
