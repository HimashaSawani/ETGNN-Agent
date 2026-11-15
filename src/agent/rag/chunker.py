from __future__ import annotations

import hashlib
import re
from typing import List
from src.agent.rag.document_loader import ApprovedDocument
from src.agent.schemas import ResearchDocumentEvidence, SourceAuthorityTier, VerificationStatus

class SectionAwareChunker:
    """
    Splits approved research documents into section-aligned chunks.
    Preserves document titles, section headers, authority tiers, and empirical qualifiers.
    """
    def __init__(self, min_chunk_chars: int = 150, max_chunk_chars: int = 2500):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk_document(self, doc: ApprovedDocument) -> List[ResearchDocumentEvidence]:
        if doc.path.endswith(".csv"):
            return self._chunk_csv(doc)
        return self._chunk_markdown(doc)

    def _chunk_csv(self, doc: ApprovedDocument) -> List[ResearchDocumentEvidence]:
        chunks: List[ResearchDocumentEvidence] = []
        lines = [l.strip() for l in doc.raw_content.strip().split("\n") if l.strip()]
        if not lines:
            return chunks

        header = lines[0]
        rows = lines[1:]

        # Create one cohesive chunk for benchmark CSV
        text_lines = [
            f"# {doc.title}",
            f"Source: {doc.path}",
            f"Columns: {header}",
            "Verified test benchmark rows (snapshots 41-49, 9973 transactions):",
        ]
        for r in rows:
            text_lines.append(f"- {r}")

        chunk_text = "\n".join(text_lines)
        c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunks.append(
            ResearchDocumentEvidence(
                evidence_id=f"EV-DOC-CSV-FINAL-TEST-RESULTS",
                source_type="research_document",
                source_artifact=doc.path,
                source_checksum=doc.checksum,
                content_hash=c_hash,
                document_title=doc.title,
                section_heading="Benchmark Test Metric Table",
                source_path=doc.path,
                authority_tier=doc.authority_tier,
                text_content=chunk_text,
                verification_status=VerificationStatus.VERIFIED,
                confidence_level="HIGH",
                limitations=["Test partition evaluation on snapshots 41–49 at threshold 0.50."],
                citation="ETGNN Authoritative Benchmark Table (results/tables/final_test_results.csv)",
                has_unverified_hypothesis=False,
            )
        )
        return chunks

    def _chunk_markdown(self, doc: ApprovedDocument) -> List[ResearchDocumentEvidence]:
        chunks: List[ResearchDocumentEvidence] = []
        raw_lines = doc.raw_content.split("\n")

        current_heading = "Overview"
        current_lines: List[str] = []
        chunk_idx = 1

        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in raw_lines:
            match = heading_pattern.match(line.strip())
            if match:
                # If we have accumulated enough text, finalize the previous chunk
                if current_lines and len("\n".join(current_lines).strip()) >= self.min_chunk_chars:
                    chunk_text = "\n".join(current_lines).strip()
                    chunks.append(self._create_chunk(doc, current_heading, chunk_text, chunk_idx))
                    chunk_idx += 1
                    current_lines = []

                current_heading = match.group(2).strip()

            current_lines.append(line)

        # Final chunk
        if current_lines and len("\n".join(current_lines).strip()) >= 50:
            chunk_text = "\n".join(current_lines).strip()
            chunks.append(self._create_chunk(doc, current_heading, chunk_text, chunk_idx))

        return chunks

    def _create_chunk(
        self,
        doc: ApprovedDocument,
        section_heading: str,
        text_content: str,
        index: int,
    ) -> ResearchDocumentEvidence:
        # Check for unverified Snapshot 43 darknet marketplace takedown claims
        has_hypo = False
        text_lower = text_content.lower()
        if ("darknet" in text_lower or "takedown" in text_lower or "hydra" in text_lower or "alphabay" in text_lower) and "snapshot 43" in text_lower:
            if "cause of this change has not been independently established" not in text_lower:
                has_hypo = True
                text_content += (
                    "\n\n> [!NOTE]\n"
                    "> **Research Integrity Notice [UNVERIFIED_HYPOTHESIS]**: "
                    "Snapshot 43 exhibits a structural change in the evaluated transaction graph (zero-incoming candidate edges jumped from 13.52% to 76.92%). "
                    "Its historical cause has not been independently established."
                )

        clean_slug = re.sub(r"[^a-zA-Z0-9]+", "-", section_heading.lower()).strip("-")[:40]
        evidence_id = f"EV-DOC-{doc.path.split('/')[-1].replace('.md', '').upper()}-{clean_slug}-{index}"

        status = VerificationStatus.UNVERIFIED_HYPOTHESIS if has_hypo else VerificationStatus.VERIFIED
        c_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()

        return ResearchDocumentEvidence(
            evidence_id=evidence_id,
            source_type="research_document",
            source_artifact=doc.path,
            source_checksum=doc.checksum,
            content_hash=c_hash,
            document_title=doc.title,
            section_heading=section_heading,
            source_path=doc.path,
            authority_tier=doc.authority_tier,
            text_content=text_content,
            verification_status=status,
            confidence_level="MEDIUM" if has_hypo else "HIGH",
            limitations=["Approved document excerpt from validated research archive."],
            citation=f"{doc.title} ({doc.path})",
            has_unverified_hypothesis=has_hypo,
        )

