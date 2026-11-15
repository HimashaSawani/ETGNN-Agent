import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Sliders,
  CheckCircle2,
  HelpCircle,
  Bell,
  Sun,
  Moon,
  User,
  Menu,
} from 'lucide-react';
import { fetchHealth } from '../api';

interface TopHeaderProps {
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onOpenCopilot: () => void;
  onStartTour: () => void;
  onToggleMobileSidebar?: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  theme,
  onToggleTheme,
  onOpenCopilot,
  onStartTour,
  onToggleMobileSidebar,
}) => {
  const [isHealthy, setIsHealthy] = useState<boolean>(true);
  const [showModeModal, setShowModeModal] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    fetchHealth()
      .then((res) => {
        if (mounted) setIsHealthy(res.status === 'healthy');
      })
      .catch(() => {
        if (mounted) setIsHealthy(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <header className="enterprise-top-header" role="banner">
      <div className="top-header-left">
        {onToggleMobileSidebar && (
          <button
            className="mobile-sidebar-toggle"
            onClick={onToggleMobileSidebar}
            aria-label="Toggle navigation menu"
          >
            <Menu size={20} />
          </button>
        )}
        <div className="header-context-text">
          <span className="context-primary">Temporal Graph Neural Networks for</span>
          <span className="context-secondary">Bitcoin Transaction Risk Analysis (Elliptic Dataset)</span>
        </div>
      </div>

      <div className="top-header-right">
        {/* Investigation Mode Pill */}
        <button
          className="header-mode-badge"
          onClick={() => setShowModeModal(!showModeModal)}
          title="Investigation Mode: Deterministic Offline Evidence Grounding (Zero API Cost)"
          aria-label="Investigation Mode"
        >
          <span className="mode-label">EVIDENCE_ONLY</span>
          <Sliders size={13} className="mode-icon" />
        </button>

        {/* System Online Status */}
        <div className="header-status-indicator" title="FastAPI Backend & Models Online">
          <span className={`status-dot ${isHealthy ? 'online' : 'offline'}`} />
          <span className="status-text">{isHealthy ? 'System Online' : 'Connecting...'}</span>
        </div>

        <div className="header-divider-v" />

        {/* Guided Tour / Help */}
        <button
          className="header-icon-btn"
          onClick={onStartTour}
          title="Start 5-Minute Research Guided Tour"
          aria-label="Guided Tour"
        >
          <HelpCircle size={18} />
        </button>

        {/* Notifications */}
        <button
          className="header-icon-btn"
          title="System Notifications: Baseline checkpoints loaded, 0 active alerts"
          aria-label="Notifications"
        >
          <Bell size={18} />
          <span className="notification-bubble" />
        </button>

        {/* Theme Toggle */}
        <button
          className="header-icon-btn"
          onClick={onToggleTheme}
          title={theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme'}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun size={18} className="theme-icon-sun" /> : <Moon size={18} className="theme-icon-moon" />}
        </button>

        {/* User Profile Avatar */}
        <div className="header-user-avatar" title="Analyst Session: Read-Only Verification Mode">
          <User size={16} />
        </div>

        {/* Primary Action Button (Matches Test Contract for /Investigate with AI/i) */}
        <button
          className="header-investigate-btn"
          onClick={onOpenCopilot}
          role="button"
          aria-label="Investigate with AI"
          title="Launch AI Forensic Investigation Copilot"
        >
          <Sparkles size={14} />
          <span>Investigate with AI</span>
        </button>
      </div>

      {/* Mode Explanation Popover */}
      {showModeModal && (
        <div className="mode-popover" role="dialog" aria-modal="true">
          <div className="popover-header">
            <CheckCircle2 size={16} color="#10b981" />
            <span className="popover-title">EVIDENCE_ONLY Governance Mode</span>
          </div>
          <p className="popover-desc">
            All investigation workflows evaluate real PyTorch forward passes and query deterministic
            historical evidence citations. Paid external LLM API keys are disabled, ensuring 100% reproducible,
            zero-cost forensic analysis.
          </p>
          <button className="popover-close-btn" onClick={() => setShowModeModal(false)}>
            Close
          </button>
        </div>
      )}
    </header>
  );
};
