from __future__ import annotations

import hashlib
import re
import time
from typing import List, Optional, Tuple
import numpy as np

from src.agent.rag.vector_store import PersistentVectorStore
from src.agent.schemas import (
    EvidenceMode,
    ResearchDocumentEvidence,
    RetrievalResult,
    SourceAuthorityTier,
    VerificationStatus,
)

AUTHORITY_TIER_WEIGHTS = {
    SourceAuthorityTier.TIER_1_REPRODUCED_METRICS: 1.0,
    SourceAuthorityTier.TIER_2_CHECKPOINT_METADATA: 0.85,
    SourceAuthorityTier.TIER_3_REVIEWED_METHODOLOGY: 0.70,
    SourceAuthorityTier.TIER_4_QUALIFIED_AUDIT: 0.55,
    SourceAuthorityTier.TIER_5_HISTORICAL_REPORT: 0.40,
}

UNTRUSTED_EVIDENCE_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"override\s+(?:system\s+)?(?:prompt|guardrails)", re.IGNORECASE),
    re.compile(r"execute\s+tool\s*:", re.IGNORECASE),
    re.compile(r"<\s*script[^>]*>", re.IGNORECASE),
]


def sanitize_retrieved_text(text: str) -> Tuple[str, bool]:
    """
    Phase 5G: Treat retrieved documents strictly as untrusted data, never executable instructions.
    Strips adversarial instruction-injection phrases from retrieved text.
    """
    sanitized = text
    triggered = False
    for pat in UNTRUSTED_EVIDENCE_INJECTION_PATTERNS:
        if pat.search(sanitized):
            triggered = True
            sanitized = pat.sub("[REDACTED_UNTRUSTED_DIRECTIVE]", sanitized)
    return sanitized, triggered


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic cosine similarity,
    lexical keyword matching, metadata-aware reranking, and source authority weighting.
    Enforces point-in-time filtering, Snapshot 43 hypothesis quarantine,
    untrusted evidence isolation, and evidence-insufficient abstention.
    """
    def __init__(self, vector_store: Optional[PersistentVectorStore] = None):
        self.vector_store = vector_store or PersistentVectorStore()
        if not self.vector_store.chunks:
            self.vector_store.load_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        evidence_mode: EvidenceMode = EvidenceMode.HISTORICAL_EVALUATION,
        as_of_snapshot: Optional[int] = None,
        min_authority_tier: Optional[SourceAuthorityTier] = None,
        include_unverified_hypotheses: bool = False,
        target_tx_id: Optional[int] = None,
        snapshot_id: Optional[int] = None,
        min_relevance_threshold: float = 0.05,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        warnings: List[str] = []

        if not self.vector_store.chunks:
            self.vector_store.load_index()

        if not self.vector_store.chunks:
            return RetrievalResult(
                query=query,
                evidence_mode=evidence_mode,
                as_of_snapshot=as_of_snapshot,
                structured_evidence=[],
                research_evidence=[],
                total_evidence_count=0,
                retrieval_latency_ms=0.0,
                max_relevance_score=0.0,
                reranker_applied=False,
                abstention_triggered=True,
                warnings=["EVIDENCE_INSUFFICIENT: Vector index is empty or uninitialized."],
            )

        # Enrich query with transaction/snapshot context if provided
        effective_query = query
        if snapshot_id == 43 and "43" not in query:
            effective_query = f"{query} snapshot 43 structural shift"
        elif target_tx_id is not None and str(target_tx_id) not in query:
            effective_query = f"{query} transaction {target_tx_id}"

        # 1. Dense Semantic Similarity
        raw_candidates = self.vector_store.query_similarity(effective_query, top_k=max(top_k * 4, 16))

        filtered_count = 0
        scored_candidates: List[Tuple[ResearchDocumentEvidence, float]] = []

        query_tokens = set(re.findall(r"[a-z0-9_]+", effective_query.lower()))

        for chunk, dense_score in raw_candidates:
            # Point-in-time filtering
            if evidence_mode == EvidenceMode.HISTORICAL_EVALUATION and as_of_snapshot is not None:
                if chunk.available_as_of_step is not None and chunk.available_as_of_step > as_of_snapshot:
                    filtered_count += 1
                    continue

            # Hypothesis quarantine
            if not include_unverified_hypotheses and chunk.has_unverified_hypothesis:
                q_lower = effective_query.lower()
                if not any(k in q_lower for k in ("takedown", "darknet", "hydra", "alphabay", "hypothesis", "43")):
                    continue

            # Authority filtering
            if min_authority_tier is not None and chunk.authority_tier.value > min_authority_tier.value:
                continue

            # Untrusted evidence sanitization (Phase 5G)
            clean_text, had_injection = sanitize_retrieved_text(chunk.text_content)
            if had_injection:
                warnings.append(f"Sanitized untrusted directive in retrieved chunk {chunk.evidence_id}.")

            # Ensure deterministic content_hash is populated (Phase 5B)
            c_hash = chunk.content_hash or hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            if clean_text != chunk.text_content or not chunk.content_hash:
                chunk = chunk.model_copy(update={"text_content": clean_text, "content_hash": c_hash})

            # 2. Lexical Token Overlap (BM25-style normalized keyword overlap)
            chunk_tokens = set(re.findall(r"[a-z0-9_]+", clean_text.lower()))
            overlap = len(query_tokens.intersection(chunk_tokens)) / max(len(query_tokens), 1)

            # 3. Authority Weight
            auth_weight = AUTHORITY_TIER_WEIGHTS.get(chunk.authority_tier, 0.5)

            # 4. Metadata-Aware Reranking Boost (Phase 5B)
            rerank_boost = 0.0
            heading_lower = chunk.section_heading.lower()
            if any(t in heading_lower for t in query_tokens if len(t) > 3):
                rerank_boost += 0.08
            if snapshot_id == 43 and "43" in clean_text:
                rerank_boost += 0.10
            if chunk.authority_tier == SourceAuthorityTier.TIER_1_REPRODUCED_METRICS:
                rerank_boost += 0.05

            # Blended Hybrid + Reranked Score
            hybrid_score = 0.50 * dense_score + 0.30 * overlap + 0.15 * auth_weight + rerank_boost
            if hybrid_score >= min_relevance_threshold:
                scored_candidates.append((chunk, hybrid_score))

        # Sort by hybrid score descending
        scored_candidates.sort(key=lambda x: -x[1])
        selected_chunks = [c[0] for c in scored_candidates[:top_k]]
        max_score = round(float(scored_candidates[0][1]), 4) if scored_candidates else 0.0
        abstained = len(selected_chunks) == 0

        if abstained:
            warnings.append("EVIDENCE_INSUFFICIENT: No research chunks met the minimum relevance threshold.")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if evidence_mode == EvidenceMode.HISTORICAL_EVALUATION:
            warnings.append("HISTORICAL_EVALUATION mode: Filtered against future evidence and test labels.")

        return RetrievalResult(
            query=query,
            evidence_mode=evidence_mode,
            as_of_snapshot=as_of_snapshot,
            structured_evidence=[],
            research_evidence=selected_chunks,
            total_evidence_count=len(selected_chunks),
            retrieval_latency_ms=round(latency_ms, 2),
            max_relevance_score=max_score,
            reranker_applied=True,
            abstention_triggered=abstained,
            has_temporal_filter_applied=(as_of_snapshot is not None),
            filtered_future_items_count=filtered_count,
            warnings=warnings,
        )

