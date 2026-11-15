import pytest
from src.agent.rag.retriever import HybridRetriever
from src.agent.schemas import EvidenceMode, SourceAuthorityTier

@pytest.fixture(scope="module")
def retriever():
    r = HybridRetriever()
    return r

def test_semantic_retrieval_rf_heterophily(retriever):
    res = retriever.retrieve("Why does Random Forest beat GNNs on Elliptic?", top_k=3)
    assert res.total_evidence_count > 0
    
    # Assert top chunks discuss heterophily or tabular performance
    combined_text = " ".join([c.text_content for c in res.research_evidence]).lower()
    assert "heterophily" in combined_text or "random forest" in combined_text or "smearing" in combined_text

def test_snapshot_43_qualification_in_retrieval(retriever):
    res = retriever.retrieve("What happened in Snapshot 43?", top_k=3)
    assert res.total_evidence_count > 0

    combined_text = " ".join([c.text_content for c in res.research_evidence]).lower()
    # Must preserve the exact qualified wording
    assert "76.92" in combined_text or "zero-incoming" in combined_text or "structural" in combined_text

def test_authority_tier_filtering(retriever):
    # Retrieve only Tier 1 (authoritative reproduced benchmarks)
    res = retriever.retrieve(
        "AUPRC metrics",
        top_k=5,
        min_authority_tier=SourceAuthorityTier.TIER_1_REPRODUCED_METRICS,
    )
    for c in res.research_evidence:
        assert c.authority_tier == SourceAuthorityTier.TIER_1_REPRODUCED_METRICS

