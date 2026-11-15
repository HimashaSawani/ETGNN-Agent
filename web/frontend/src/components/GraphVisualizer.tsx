import type React from 'react';
import { useState, useEffect, useRef } from 'react';
import { GitFork, ZoomIn, ZoomOut, RotateCcw, AlertCircle, Info } from 'lucide-react';
import { fetchSubgraph } from '../api';
import type { SubgraphResponse, SubgraphNode, SubgraphEdge } from '../api';

interface Props {
  transactionId: number;
  onSelectNode: (txId: number) => void;
}

export const GraphVisualizer: React.FC<Props> = ({ transactionId, onSelectNode }) => {
  const [hops, setHops] = useState<number>(2);
  const [subgraph, setSubgraph] = useState<SubgraphResponse | null>(null);
  const [, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [hoveredNode, setHoveredNode] = useState<SubgraphNode | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  // Load subgraph
  useEffect(() => {
    if (!transactionId) return;
    setLoading(true);
    setError(null);
    setHoveredNode(null);
    fetchSubgraph(transactionId, hops)
      .then((res) => {
        setSubgraph(res);
        computeLayout(res);
      })
      .catch((err) => setError(err.message || 'Failed to load subgraph'))
      .finally(() => setLoading(false));
  }, [transactionId, hops]);

  // Compute radial layout: Center node at (0,0), 1-hop in inner circle, 2-hop in outer circle
  const computeLayout = (data: SubgraphResponse) => {
    const positions = new Map<string, { x: number; y: number }>();
    const centerNode = data.nodes.find((n) => n.is_center);
    if (!centerNode) return;

    positions.set(centerNode.id, { x: 0, y: 0 });

    const otherNodes = data.nodes.filter((n) => !n.is_center);
    const n = otherNodes.length;

    // Arrange in concentric circles
    const innerRadius = 130;
    const outerRadius = 240;

    otherNodes.forEach((node, i) => {
      const isIncident = data.edges.some(
        (e) =>
          (e.source === centerNode.id && e.target === node.id) ||
          (e.target === centerNode.id && e.source === node.id)
      );
      const radius = isIncident ? innerRadius : outerRadius;
      const angle = (2 * Math.PI * i) / Math.max(1, n);
      positions.set(node.id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    });

    positionsRef.current = positions;
    drawCanvas();
  };

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !subgraph) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + pan.x;
    const centerY = height / 2 + pan.y;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.scale(zoom, zoom);

    // 1. Draw Edges
    subgraph.edges.forEach((edge: SubgraphEdge) => {
      const srcPos = positionsRef.current.get(edge.source);
      const dstPos = positionsRef.current.get(edge.target);
      if (!srcPos || !dstPos) return;

      ctx.beginPath();
      ctx.moveTo(srcPos.x, srcPos.y);
      ctx.lineTo(dstPos.x, dstPos.y);
      ctx.strokeStyle = edge.is_incident_to_center
        ? 'rgba(56, 189, 248, 0.45)'
        : 'rgba(51, 65, 85, 0.35)';
      ctx.lineWidth = edge.is_incident_to_center ? 1.8 : 1.0;
      ctx.stroke();

      // Arrow head
      const headlen = 6;
      const angle = Math.atan2(dstPos.y - srcPos.y, dstPos.x - srcPos.x);
      const targetRadius = 14;
      const arrowX = dstPos.x - targetRadius * Math.cos(angle);
      const arrowY = dstPos.y - targetRadius * Math.sin(angle);

      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(
        arrowX - headlen * Math.cos(angle - Math.PI / 6),
        arrowY - headlen * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        arrowX - headlen * Math.cos(angle + Math.PI / 6),
        arrowY - headlen * Math.sin(angle + Math.PI / 6)
      );
      ctx.fillStyle = edge.is_incident_to_center
        ? 'rgba(56, 189, 248, 0.7)'
        : 'rgba(71, 85, 105, 0.5)';
      ctx.fill();
    });

    // 2. Draw Nodes
    subgraph.nodes.forEach((node: SubgraphNode) => {
      const pos = positionsRef.current.get(node.id);
      if (!pos) return;

      const isHovered = hoveredNode && hoveredNode.id === node.id;
      const radius = node.is_center ? 18 : 11;

      // Outer glow for center or hovered node
      if (node.is_center || isHovered) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 5, 0, Math.PI * 2);
        ctx.fillStyle = node.is_center
          ? 'rgba(245, 158, 11, 0.3)'
          : 'rgba(56, 189, 248, 0.35)';
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

      // Color by label
      if (node.actual_label === 1) {
        ctx.fillStyle = '#f43f5e'; // Illicit
      } else if (node.actual_label === 0) {
        ctx.fillStyle = '#10b981'; // Licit
      } else {
        ctx.fillStyle = '#64748b'; // Unknown
      }
      ctx.fill();

      // Border
      ctx.lineWidth = node.is_center ? 2.5 : 1.5;
      ctx.strokeStyle = node.is_center ? '#fbbf24' : '#1e293b';
      ctx.stroke();

      // Text label for center
      if (node.is_center) {
        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'center';
        ctx.fillText('TARGET', pos.x, pos.y - radius - 8);
      }
    });

    ctx.restore();
  };

  useEffect(() => {
    drawCanvas();
  }, [subgraph, zoom, pan, hoveredNode]);

  // Mouse interaction handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !subgraph) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
      return;
    }

    // Hit test for nodes
    const centerX = canvas.width / 2 + pan.x;
    const centerY = canvas.height / 2 + pan.y;
    const worldX = (mouseX - centerX) / zoom;
    const worldY = (mouseY - centerY) / zoom;

    let found: SubgraphNode | null = null;
    for (const node of subgraph.nodes) {
      const pos = positionsRef.current.get(node.id);
      if (pos) {
        const dx = pos.x - worldX;
        const dy = pos.y - worldY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= (node.is_center ? 22 : 14)) {
          found = node;
          break;
        }
      }
    }
    setHoveredNode(found);
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleClick = () => {
    if (hoveredNode && !hoveredNode.is_center) {
      onSelectNode(hoveredNode.transaction_id);
    }
  };

  return (
    <div className="graph-visualizer">
      {/* Header Bar */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <GitFork size={20} color="#38bdf8" />
            <div>
              <div style={{ fontSize: 17, fontWeight: 700, color: '#f8fafc' }}>
                Interactive Transaction Neighborhood
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Center Tx: <span className="mono" style={{ color: '#38bdf8' }}>{transactionId}</span> |{' '}
                {subgraph ? `${subgraph.node_count} nodes, ${subgraph.edge_count} directed edges` : 'Loading...'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: 4, background: '#0d1322', padding: 3, borderRadius: 6, border: '1px solid #1e293b' }}>
              <button
                className={`tab-btn ${hops === 1 ? 'active' : ''}`}
                style={{ padding: '4px 10px', fontSize: 11 }}
                onClick={() => setHops(1)}
              >
                1-Hop
              </button>
              <button
                className={`tab-btn ${hops === 2 ? 'active' : ''}`}
                style={{ padding: '4px 10px', fontSize: 11 }}
                onClick={() => setHops(2)}
              >
                2-Hop
              </button>
            </div>

            <div style={{ display: 'flex', gap: 4 }}>
              <button className="btn" style={{ padding: '6px 10px' }} onClick={() => setZoom((z) => Math.min(2.5, z * 1.2))}>
                <ZoomIn size={14} />
              </button>
              <button className="btn" style={{ padding: '6px 10px' }} onClick={() => setZoom((z) => Math.max(0.4, z * 0.8))}>
                <ZoomOut size={14} />
              </button>
              <button className="btn" style={{ padding: '6px 10px' }} onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
                <RotateCcw size={14} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="safety-banner danger">
          <AlertCircle size={20} color="#f43f5e" />
          <div className="safety-desc">{error}</div>
        </div>
      )}

      {subgraph?.is_truncated && (
        <div className="safety-banner danger" style={{ marginBottom: 18 }}>
          <AlertCircle size={20} color="#f43f5e" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="safety-title" style={{ color: '#f43f5e' }}>
              INCOMPLETE GRAPH VISUALIZATION (Neighborhood Truncated)
            </div>
            <div className="safety-desc">
              The complete {subgraph.num_hops}-hop neighborhood contains <strong>{subgraph.total_untruncated_nodes}</strong> transactions.
              To maintain client-side browser performance, this visualization is capped at {subgraph.node_count} nodes.
              <strong> This display does NOT represent the full neighborhood graph.</strong>
            </div>
          </div>
        </div>
      )}

      {/* Main Canvas & Details Panel */}
      <div className="grid-3" style={{ gridTemplateColumns: '2.4fr 1fr' }}>
        <div className="card" style={{ padding: 0, overflow: 'hidden', position: 'relative' }}>
          <canvas
            ref={canvasRef}
            width={850}
            height={560}
            style={{ width: '100%', height: 560, display: 'block', background: '#0b101d', cursor: isDragging ? 'grabbing' : 'grab' }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
          />

          {/* Legend Overlay */}
          <div
            style={{
              position: 'absolute',
              bottom: 14,
              left: 14,
              background: 'rgba(16, 23, 38, 0.88)',
              border: '1px solid #1e293b',
              borderRadius: 8,
              padding: '10px 14px',
              fontSize: 11,
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
              backdropFilter: 'blur(8px)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#fbbf24', border: '1px solid #f59e0b' }} />
              <span style={{ color: '#f8fafc' }}>Center Target Node</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#f43f5e' }} />
              <span style={{ color: '#f8fafc' }}>Illicit Transaction (1)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#10b981' }} />
              <span style={{ color: '#f8fafc' }}>Licit Transaction (0)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#64748b' }} />
              <span style={{ color: '#94a3b8' }}>Unlabelled Transaction (-1)</span>
            </div>
          </div>
        </div>

        {/* Hover / Node Details Panel */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div className="card-title" style={{ fontSize: 15 }}>
              <Info size={16} color="#38bdf8" />
              Node Inspector
            </div>
          </div>

          {hoveredNode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ background: '#0d1322', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>Transaction ID</div>
                <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc' }}>
                  {hoveredNode.transaction_id}
                </div>
              </div>

              <div style={{ background: '#0d1322', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>Actual Class</div>
                <div style={{ marginTop: 4 }}>
                  {hoveredNode.actual_label === 1 ? (
                    <span className="badge" style={{ background: 'var(--illicit-bg)', color: 'var(--illicit-red)' }}>
                      Illicit (1)
                    </span>
                  ) : hoveredNode.actual_label === 0 ? (
                    <span className="badge" style={{ background: 'var(--licit-bg)', color: 'var(--licit-green)' }}>
                      Licit (0)
                    </span>
                  ) : (
                    <span className="badge" style={{ background: 'var(--unknown-bg)', color: 'var(--unknown-gray)' }}>
                      Unlabelled (-1)
                    </span>
                  )}
                </div>
              </div>

              <div style={{ background: '#0d1322', padding: 12, borderRadius: 6, border: '1px solid #1e293b' }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>ETGNN Predicted Probability</div>
                <div className="mono" style={{ fontSize: 18, fontWeight: 800, color: hoveredNode.etgnn_probability >= 0.5 ? '#f43f5e' : '#10b981', marginTop: 2 }}>
                  {(hoveredNode.etgnn_probability * 100).toFixed(1)}%
                </div>
              </div>

              {!hoveredNode.is_center && (
                <button
                  className="btn btn-primary"
                  style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
                  onClick={() => onSelectNode(hoveredNode.transaction_id)}
                >
                  Make This Center Node
                </button>
              )}
            </div>
          ) : (
            <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: '40px 10px' }}>
              Hover over or click any node on the graph canvas to inspect its parameters.
            </div>
          )}

          <div style={{ marginTop: 'auto', paddingTop: 18, borderTop: '1px solid #1e293b', fontSize: 11, color: '#64748b' }}>
            Note: In the Elliptic dataset, all graph edges exist strictly within the same 2-week time step. There are no inter-step edges.
          </div>
        </div>
      </div>
    </div>
  );
};
