#!/usr/bin/env python
"""
Build and verify the persistent research knowledge vector index for ETGNN-Agent.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.rag.document_loader import ApprovedDocumentLoader
from src.agent.rag.chunker import SectionAwareChunker
from src.agent.rag.vector_store import PersistentVectorStore

def main():
    print("=" * 70)
    print("ETGNN-Agent: Building Persistent Research Knowledge Index")
    print("=" * 70)

    loader = ApprovedDocumentLoader(root_dir=ROOT)
    docs = loader.load_approved_documents()
    checksums = loader.get_source_checksums()

    print(f"Loaded {len(docs)} approved research documents:")
    for d in docs:
        print(f"  [{d.authority_tier.name}] {d.title} ({d.path})")

    chunker = SectionAwareChunker()
    all_chunks = []
    for d in docs:
        chunks = chunker.chunk_document(d)
        all_chunks.extend(chunks)

    print(f"\nGenerated {len(all_chunks)} section-aware chunks.")

    index_dir = ROOT / "data/processed/research_index"
    store = PersistentVectorStore(index_dir=index_dir)
    store.build_index(all_chunks, checksums)

    print(f"\nIndex successfully built and persisted to: {index_dir}")
    print(f"  Embedding shape: {store.embeddings.shape}")
    print(f"  Schema version: 1.0.0")

    # Quick test query
    test_queries = [
        "Why does Random Forest outperform GNN on Elliptic?",
        "Snapshot 43 topological shift and zero-incoming edges",
        "Ablation study removing edges from ETGNN",
    ]

    print("\nExecuting validation test queries:")
    for q in test_queries:
        res = store.query_similarity(q, top_k=2)
        print(f"\nQuery: '{q}'")
        for chunk, score in res:
            print(f"  -> Score: {score:.4f} | [{chunk.section_heading}] from {chunk.source_path}")

    print("\nIndex verification COMPLETE and operational.")

if __name__ == "__main__":
    main()

