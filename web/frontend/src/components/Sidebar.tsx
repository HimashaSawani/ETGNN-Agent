import React from 'react';
import {
  LayoutDashboard,
  Search,
  Sparkles,
  Cpu,
  GitFork,
  HelpCircle,
  BarChart3,
  GitCompare,
  Database,
  Download,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Shield,
  Code2,
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  onOpenCopilot: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  isAction?: boolean;
  badge?: string;
  badgeColor?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  onOpenCopilot,
  isCollapsed,
  onToggleCollapse,
}) => {
  const mainNavItems: NavItem[] = [
    { id: 'overview', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'explorer', label: 'Transaction Explorer', icon: Search },
    { id: 'copilot_action', label: 'AI Investigation', icon: Sparkles, isAction: true },
    { id: 'prediction', label: 'Model Predictions', icon: Cpu },
    { id: 'graph', label: 'Subgraph Visualizer', icon: GitFork },
    { id: 'explanation', label: 'Explanation Inspector', icon: HelpCircle },
    { id: 'benchmarks', label: 'Benchmark Comparison', icon: BarChart3 },
    { id: 'c1c2', label: 'Explanation Evidence', icon: GitCompare },
    { id: 'monitoring', label: 'Dataset & Statistics', icon: Database },
  ];

  const secondaryNavItems: NavItem[] = [
    { id: 'reports', label: 'Reports & Export', icon: Download, isAction: true },
    { id: 'methodology', label: 'Documentation', icon: BookOpen },
  ];

  const handleItemClick = (item: NavItem) => {
    if (item.id === 'copilot_action' || item.id === 'reports') {
      onOpenCopilot();
    } else {
      onSelectTab(item.id);
    }
  };

  return (
    <aside className={`enterprise-sidebar ${isCollapsed ? 'collapsed' : ''}`} aria-label="Main Navigation">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-logo-container" onClick={() => onSelectTab('overview')} title="ETGNN-Agent">
          <div className="brand-shield-icon">
            <Shield size={20} className="shield-icon-svg" />
            <div className="shield-node-glow" />
          </div>
          {!isCollapsed && (
            <div className="brand-text-block">
              <span className="brand-app-name">ETGNN-Agent</span>
              <span className="brand-app-tagline">Financial Fraud Detection Platform</span>
            </div>
          )}
        </div>
        <button
          className="sidebar-collapse-toggle"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Primary Navigation List */}
      <nav className="sidebar-nav">
        <div className="nav-group">
          {!isCollapsed && <div className="nav-group-heading">Platform</div>}
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = !item.isAction && activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`sidebar-nav-item ${isActive ? 'active' : ''} ${item.isAction ? 'action-item' : ''}`}
                onClick={() => handleItemClick(item)}
                title={isCollapsed ? item.label : undefined}
                role="button"
                aria-current={isActive ? 'page' : undefined}
              >
                <div className="nav-item-icon">
                  <Icon size={18} />
                </div>
                {!isCollapsed && <span className="nav-item-label">{item.label}</span>}
                {!isCollapsed && item.badge && (
                  <span className="nav-item-badge" style={{ backgroundColor: item.badgeColor || '#2563eb' }}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        <div className="sidebar-divider" />

        <div className="nav-group">
          {!isCollapsed && <div className="nav-group-heading">Research & Exports</div>}
          {secondaryNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = !item.isAction && activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                onClick={() => handleItemClick(item)}
                title={isCollapsed ? item.label : undefined}
                role="button"
                aria-current={isActive ? 'page' : undefined}
              >
                <div className="nav-item-icon">
                  <Icon size={18} />
                </div>
                {!isCollapsed && <span className="nav-item-label">{item.label}</span>}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Sidebar Footer Link */}
      <div className="sidebar-footer">
        <a
          href="https://github.com/HimashaSawani/ETGNN-Fraud-Detection/tree/release/clean-main"
          target="_blank"
          rel="noopener noreferrer"
          className="sidebar-github-card"
          title="View Clean Verified Release on GitHub"
        >
          <Code2 size={16} className="github-icon" />
          {!isCollapsed && (
            <div className="github-card-text">
              <span className="github-card-title">ETGNN-Agent</span>
              <span className="github-card-sub">v1.0.0 (Research)</span>
            </div>
          )}
          {!isCollapsed && <ExternalLink size={12} className="github-ext-icon" />}
        </a>
      </div>
    </aside>
  );
};
