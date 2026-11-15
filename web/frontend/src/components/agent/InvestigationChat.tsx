import type React from 'react';
import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import type { ChatMessage } from '../../types/agent';
import { askInvestigationChat } from '../../services/agentApi';

interface Props {
  investigationId: string;
  transactionId: number;
  onSelectCitation?: (citationId: string) => void;
}

const SUGGESTED_QUESTIONS = [
  'Why did ETGNN classify this transaction differently from Random Forest?',
  'What graph evidence supports this prediction?',
  'What does GNNExplainer indicate about this transaction?',
  'Why is Snapshot 43 important for this evaluation?',
  'What scientific limitations affect this investigation?',
];

export const InvestigationChat: React.FC<Props> = ({
  investigationId,
  transactionId,
  onSelectCitation,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'agent',
      text: `Hello! I am your AI Forensic Copilot. I have analyzed Transaction ${transactionId} with verified graph, baseline models, and GNNExplainer evidence. You can ask follow-up questions grounded strictly in this transaction's evidence bundle.`,
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || inputText;
    if (!textToSend.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setLoading(true);
    setError(null);

    try {
      const resp = await askInvestigationChat({
        investigation_id: investigationId,
        question: textToSend,
      });

      const agentMsg: ChatMessage = {
        id: `agent_${Date.now()}`,
        sender: 'agent',
        text: resp.answer,
        cited_evidence_ids: resp.cited_evidence_ids,
        provider_type: resp.provider_type,
        generation_mode: resp.generation_mode,
        limitations: resp.limitations,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, agentMsg]);
    } catch (err: any) {
      setError(err.message || 'Follow-up query failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="investigation-chat card" style={{ padding: '16px 20px', marginBottom: 20, display: 'flex', flexDirection: 'column', height: 480 }}>
      {/* Chat Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, borderBottom: '1px solid var(--border-subtle, #182234)', paddingBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Sparkles size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Evidence-Grounded Investigation Chat
          </h3>
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
          Scoped strictly to Tx {transactionId}
        </span>
      </div>

      {/* Suggested Questions Chips */}
      <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 8, marginBottom: 8 }}>
        {SUGGESTED_QUESTIONS.map((q, i) => (
          <button
            key={i}
            className="btn btn-secondary"
            onClick={() => handleSend(q)}
            disabled={loading}
            style={{
              fontSize: 11,
              whiteSpace: 'nowrap',
              padding: '4px 10px',
              borderRadius: 14,
              backgroundColor: 'rgba(56, 189, 248, 0.08)',
              color: 'var(--text-secondary, #94a3b8)',
              border: '1px solid rgba(56, 189, 248, 0.2)',
              cursor: 'pointer',
            }}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Message Thread */}
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: 6, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((m) => {
          const isUser = m.sender === 'user';
          return (
            <div
              key={m.id}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
              }}
            >
              {!isUser && (
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                >
                  <Bot size={16} color="#38bdf8" />
                </div>
              )}

              <div
                style={{
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-md, 10px)',
                  backgroundColor: isUser ? 'var(--accent-blue, #2563eb)' : 'rgba(15, 23, 42, 0.45)',
                  border: isUser ? 'none' : '1px solid var(--border-subtle, #182234)',
                  color: '#f8fafc',
                  fontSize: 12.5,
                  lineHeight: 1.5,
                }}
              >
                <div>{m.text}</div>

                {m.cited_evidence_ids && m.cited_evidence_ids.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                    <span style={{ fontSize: 10, color: '#94a3b8' }}>Evidence Citations:</span>
                    {m.cited_evidence_ids.map((cid, cIdx) => (
                      <button
                        key={cIdx}
                        onClick={() => onSelectCitation && onSelectCitation(cid)}
                        style={{
                          fontSize: 10,
                          fontFamily: 'var(--font-mono, monospace)',
                          padding: '1px 6px',
                          borderRadius: 8,
                          backgroundColor: 'rgba(56, 189, 248, 0.15)',
                          color: '#38bdf8',
                          border: '1px solid rgba(56, 189, 248, 0.3)',
                          cursor: 'pointer',
                        }}
                      >
                        [{cid}]
                      </button>
                    ))}
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 6, fontSize: 9.5, color: isUser ? 'rgba(255, 255, 255, 0.7)' : '#64748b' }}>
                  <span>{m.timestamp}</span>
                  {m.generation_mode && (
                    <span style={{ fontFamily: 'var(--font-mono, monospace)' }}>
                      {m.provider_type} ({m.generation_mode})
                    </span>
                  )}
                </div>
              </div>

              {isUser && (
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginTop: 2,
                  }}
                >
                  <User size={16} color="#f8fafc" />
                </div>
              )}
            </div>
          );
        })}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#94a3b8', fontSize: 12 }}>
            <Loader2 size={16} className="spin" color="#38bdf8" />
            <span>Consulting verified transaction evidence records...</span>
          </div>
        )}

        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#f43f5e', fontSize: 12 }}>
            <AlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        style={{ display: 'flex', gap: 8, marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border-subtle, #182234)' }}
      >
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask about model spread, graph neighbors, GNNExplainer, or Snapshot 43..."
          disabled={loading}
          style={{
            flex: 1,
            padding: '8px 12px',
            borderRadius: 'var(--radius-sm, 6px)',
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid var(--border-color, #1e293b)',
            color: '#f8fafc',
            fontSize: 12.5,
          }}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!inputText.trim() || loading}
          style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 6 }}
        >
          <span>Ask</span>
          <Send size={13} />
        </button>
      </form>
    </div>
  );
};

