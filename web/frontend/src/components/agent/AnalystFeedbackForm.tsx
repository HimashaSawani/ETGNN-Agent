import type React from 'react';
import { useState, useEffect } from 'react';
import { MessageSquarePlus, Star, ShieldCheck, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import type { FeedbackRecord, FeedbackSubmission } from '../../types/agent';
import { submitAnalystFeedback, fetchAnalystFeedback } from '../../services/agentApi';

interface Props {
  investigationId: string;
  transactionId: number;
  availableCitations?: string[];
}

export const AnalystFeedbackForm: React.FC<Props> = ({
  investigationId,
  transactionId,
  availableCitations = [],
}) => {
  const [category, setCategory] = useState<FeedbackSubmission['feedback_category']>('helpful');
  const [rating, setRating] = useState<number>(5);
  const [comment, setComment] = useState<string>('');
  const [reviewerId, setReviewerId] = useState<string>('analyst_local');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [previousFeedback, setPreviousFeedback] = useState<FeedbackRecord[]>([]);

  const loadPreviousFeedback = async () => {
    try {
      const records = await fetchAnalystFeedback(investigationId);
      setPreviousFeedback(records);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadPreviousFeedback();
  }, [investigationId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setSuccessMessage(null);
    setErrorMessage(null);

    try {
      await submitAnalystFeedback({
        investigation_id: investigationId,
        transaction_id: transactionId,
        feedback_category: category,
        rating,
        comment: comment.trim() || undefined,
        reviewer_id: reviewerId.trim() || 'analyst_local',
        report_version: 'ETGNN-Agent-v3.5.0',
        evidence_references: availableCitations,
      });

      setSuccessMessage('Analyst review feedback successfully recorded in isolated database.');
      setComment('');
      await loadPreviousFeedback();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="analyst-feedback-form card" style={{ padding: '16px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <MessageSquarePlus size={18} color="var(--accent-cyan, #38bdf8)" />
          <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Human-in-the-Loop Analyst Review
          </h3>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#10b981' }}>
          <ShieldCheck size={14} />
          <span>Checkpoint Isolated</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Rating, Category, and Reviewer Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
          {/* Reviewer ID */}
          <div>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 4 }}>
              Analyst ID / Call-sign
            </label>
            <input
              type="text"
              value={reviewerId}
              onChange={(e) => setReviewerId(e.target.value)}
              placeholder="analyst_local"
              style={{
                width: '100%',
                padding: '7px 10px',
                borderRadius: 'var(--radius-sm, 6px)',
                backgroundColor: 'rgba(15, 23, 42, 0.5)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 12,
              }}
            />
          </div>

          {/* Category */}
          <div>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 4 }}>
              Review Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as any)}
              style={{
                width: '100%',
                padding: '7px 10px',
                borderRadius: 'var(--radius-sm, 6px)',
                backgroundColor: 'rgba(15, 23, 42, 0.5)',
                border: '1px solid var(--border-color, #1e293b)',
                color: '#f8fafc',
                fontSize: 12,
              }}
            >
              <option value="helpful">Helpful Explanation</option>
              <option value="unhelpful">Unhelpful Explanation</option>
              <option value="citation_issue">Citation Issue</option>
              <option value="missing_evidence">Missing Evidence</option>
              <option value="incorrect_numerical">Incorrect Numerical Statement</option>
              <option value="needs_further_review">Needs Further Review</option>
            </select>
          </div>

          {/* Rating */}
          <div>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 4 }}>
              Quality Rating (1–5)
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, paddingTop: 4 }}>
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  onClick={() => setRating(star)}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    padding: 2,
                    color: star <= rating ? '#fbbf24' : 'var(--text-muted, #64748b)',
                    transition: 'transform 0.1s ease',
                  }}
                >
                  <Star size={18} fill={star <= rating ? '#fbbf24' : 'none'} />
                </button>
              ))}
              <span style={{ fontSize: 12, color: 'var(--text-muted, #64748b)', marginLeft: 6 }}>
                ({rating}/5)
              </span>
            </div>
          </div>
        </div>

        {/* Written Comment */}
        <div>
          <label style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 4 }}>
            Analyst Review Notes (Optional)
          </label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Document discrepancies, suggest alternative evidence, or verify accuracy..."
            maxLength={2000}
            rows={3}
            style={{
              width: '100%',
              padding: '8px 10px',
              borderRadius: 'var(--radius-sm, 6px)',
              backgroundColor: 'rgba(15, 23, 42, 0.5)',
              border: '1px solid var(--border-color, #1e293b)',
              color: '#f8fafc',
              fontSize: 12,
              fontFamily: 'inherit',
              resize: 'vertical',
            }}
          />
        </div>

        {/* Safety Disclaimer */}
        <div
          style={{
            fontSize: 11,
            color: 'var(--text-muted, #64748b)',
            lineHeight: 1.4,
            backgroundColor: 'rgba(15, 23, 42, 0.25)',
            padding: '8px 12px',
            borderRadius: 6,
            border: '1px solid var(--border-subtle, #182234)',
          }}
        >
          <strong>Safety & Scientific Boundary Guarantee:</strong> Feedback is persisted in a standalone SQLite database (`data/analyst_feedback.db`). It does <em>not</em> automatically retrain ETGNN, modify model weights, or alter ground-truth Elliptic labels.
        </div>

        {/* Status Messages */}
        {successMessage && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#10b981', fontSize: 12 }}>
            <CheckCircle2 size={14} />
            <span>{successMessage}</span>
          </div>
        )}
        {errorMessage && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#f43f5e', fontSize: 12 }}>
            <AlertCircle size={14} />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Submit Button */}
        <div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting}
            style={{ padding: '8px 16px', display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}
          >
            {submitting ? (
              <>
                <Loader2 size={14} className="spin" />
                <span>Recording Review...</span>
              </>
            ) : (
              <span>Submit Analyst Review</span>
            )}
          </button>
        </div>
      </form>

      {/* Previous Feedback List */}
      {previousFeedback.length > 0 && (
        <div style={{ marginTop: 14, paddingTop: 10, borderTop: '1px solid var(--border-subtle, #182234)' }}>
          <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted, #64748b)', marginBottom: 8 }}>
            Previous Analyst Reviews ({previousFeedback.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {previousFeedback.map((fb) => (
              <div
                key={fb.feedback_id}
                style={{
                  padding: '6px 10px',
                  borderRadius: 4,
                  backgroundColor: 'rgba(15, 23, 42, 0.3)',
                  border: '1px solid var(--border-subtle, #182234)',
                  fontSize: 11,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <span style={{ fontWeight: 600, color: '#38bdf8' }}>{fb.feedback_category}</span>{' '}
                  <span style={{ color: '#fbbf24' }}>({fb.rating}★)</span>: {fb.comment || 'No written comment'}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted, #64748b)' }}>
                  {new Date(fb.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
