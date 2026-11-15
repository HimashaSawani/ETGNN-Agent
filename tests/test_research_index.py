import pytest
from pathlib import Path
from src.agent.rag.document_loader import ApprovedDocumentLoader, APPROVED_SOURCE_ALLOWLIST
from src.agent.rag.chunker import SectionAwareChunker
from src.agent.rag.vector_store import PersistentVectorStore

def test_document_loader_allowlist_enforcement():
    loader = ApprovedDocumentLoader()
    docs = loader.load_approved_documents()

    assert len(docs) > 0
    loaded_paths = {d.path for d in docs}
    
    # Assert every loaded path is in the approved allowlist
    for p in loaded_paths:
        assert p in APPROVED_SOURCE_ALLOWLIST

    # Assert no secrets, checkpoints or raw files are loaded
    for p in loaded_paths:
        assert not p.endswith(".pt")
        assert not p.endswith(".joblib")
        assert not p.endswith(".env")

def test_section_aware_chunker():
    loader = ApprovedDocumentLoader()
    docs = loader.load_approved_documents()
    chunker = SectionAwareChunker()

    all_chunks = []
    for d in docs:
        chunks = chunker.chunk_document(d)
        all_chunks.extend(chunks)

    assert len(all_chunks) >= 50
    for c in all_chunks:
        assert c.evidence_id.startswith("EV-DOC-")
        assert len(c.text_content) > 0
        assert c.document_title != ""
        assert c.section_heading != ""

def test_vector_store_persistence_and_checksum_verification(tmp_path):
    loader = ApprovedDocumentLoader()
    docs = loader.load_approved_documents()[:3]
    checksums = {d.path: d.checksum for d in docs}

    chunker = SectionAwareChunker()
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    store = PersistentVectorStore(index_dir=tmp_path)
    store.build_index(chunks, checksums)

    assert store.is_built() is True
    assert (tmp_path / "index_metadata.json").exists()
    assert (tmp_path / "embeddings.npy").exists()

    # Verify reload
    reloaded_store = PersistentVectorStore(index_dir=tmp_path)
    assert reloaded_store.load_index() is True
    assert len(reloaded_store.chunks) == len(chunks)

    # Checksum integrity
    assert reloaded_store.verify_integrity(checksums) is True
    # Mutate checksum to test drift detection
    mutated = dict(checksums)
    mutated[docs[0].path] = "tampered_checksum"
    assert reloaded_store.verify_integrity(mutated) is False

