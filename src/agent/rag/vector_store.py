from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agent.schemas import ResearchDocumentEvidence

INDEX_SCHEMA_VERSION = "1.0.0"
EMBEDDING_MODEL_IDENTIFIER = "local-tfidf-sublinear-v1"

class PersistentVectorStore:
    """
    Lightweight, persistent vector store designed for research document retrieval.
    Generates reproducible L2-normalized dense embeddings, persists metadata and source hashes,
    and supports sub-millisecond cosine similarity searches without external network dependencies.
    """
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or Path("data/processed/research_index")
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[ResearchDocumentEvidence] = []
        self.source_checksums: Dict[str, str] = {}

    def is_built(self) -> bool:
        return (
            (self.index_dir / "index_metadata.json").exists()
            and (self.index_dir / "embeddings.npy").exists()
            and (self.index_dir / "chunks.json").exists()
            and (self.index_dir / "vectorizer.joblib").exists()
        )

    def build_index(
        self,
        chunks: List[ResearchDocumentEvidence],
        source_checksums: Dict[str, str],
    ) -> None:
        """Build and persist the vector index from research chunks."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self.source_checksums = source_checksums

        texts = [c.text_content for c in chunks]

        # Fit TF-IDF with sublinear scaling (logarithmic term frequency)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            max_features=5000,
        )
        sparse_matrix = self.vectorizer.fit_transform(texts)
        self.embeddings = sparse_matrix.toarray().astype(np.float32)

        # Persist artifacts
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        joblib.dump(self.vectorizer, self.index_dir / "vectorizer.joblib")

        chunks_data = [c.model_dump() for c in chunks]
        (self.index_dir / "chunks.json").write_text(
            json.dumps(chunks_data, indent=2), encoding="utf-8"
        )

        metadata = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": EMBEDDING_MODEL_IDENTIFIER,
            "total_chunks": len(chunks),
            "embedding_dim": int(self.embeddings.shape[1]),
            "source_checksums": source_checksums,
        }
        (self.index_dir / "index_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def load_index(self) -> bool:
        """Load persistent index from disk."""
        if not self.is_built():
            return False

        meta = json.loads((self.index_dir / "index_metadata.json").read_text(encoding="utf-8"))
        if meta.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            return False

        self.source_checksums = meta.get("source_checksums", {})
        self.embeddings = np.load(self.index_dir / "embeddings.npy")
        self.vectorizer = joblib.load(self.index_dir / "vectorizer.joblib")

        chunks_data = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [ResearchDocumentEvidence(**c) for c in chunks_data]
        return True

    def query_similarity(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[ResearchDocumentEvidence, float]]:
        """Compute cosine similarity between query and all indexed chunks."""
        if self.vectorizer is None or self.embeddings is None or not self.chunks:
            if not self.load_index():
                return []

        query_vec = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        # Cosine similarity for L2-normalized vectors is dot product
        scores = np.dot(self.embeddings, query_vec.T).flatten()

        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))
        return results

    def verify_integrity(self, current_source_checksums: Dict[str, str]) -> bool:
        """Check whether any underlying source files have drifted relative to the index."""
        if not self.is_built():
            return False
        meta = json.loads((self.index_dir / "index_metadata.json").read_text(encoding="utf-8"))
        indexed_checksums = meta.get("source_checksums", {})
        return indexed_checksums == current_source_checksums

