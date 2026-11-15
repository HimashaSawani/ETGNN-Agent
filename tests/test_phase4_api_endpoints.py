"""
Tests for Phase 4 Backend Endpoints:
- Investigation list
- PDF report export
- Interactive Chat Q&A
- Analyst Feedback Store & Endpoints
"""

import pytest
from src.web.app import (
    api_agent_investigate,
    api_list_investigations,
    api_export_investigation_pdf,
    api_agent_chat,
    api_submit_feedback,
    api_list_feedback,
    api_get_investigation_feedback,
    AgentInvestigateRequest,
)
from src.agent.feedback.store import get_feedback_store


def test_phase4_backend_lifecycle():
    # 1. Run an investigation to populate runner cache
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        question="Initial test investigation for Phase 4",
        llm_provider="mock",
    )
    inv_res = api_agent_investigate(req)
    inv_id = inv_res["investigation_id"]
    assert inv_res["status"] == "COMPLETED"

    # 2. Test list investigations
    inv_list = api_list_investigations()
    assert isinstance(inv_list, list)
    assert any(i["investigation_id"] == inv_id for i in inv_list)

    # 3. Test PDF report export
    pdf_resp = api_export_investigation_pdf(inv_id)
    assert pdf_resp.status_code == 200
    assert pdf_resp.media_type == "application/pdf"
    assert len(pdf_resp.body) > 1000
    assert pdf_resp.body.startswith(b"%PDF")

    # 4. Test Interactive Chat follow-up
    chat_payload = {
        "investigation_id": inv_id,
        "question": "Why did ETGNN classify differently from Random Forest?",
        "llm_provider": "mock",
    }
    chat_resp = api_agent_chat(chat_payload)
    assert chat_resp["investigation_id"] == inv_id
    assert chat_resp["transaction_id"] == 230425980
    assert "spread" in chat_resp["answer"].lower()
    assert len(chat_resp["cited_evidence_ids"]) >= 1

    # 5. Test Analyst Feedback submission
    fb_payload = {
        "investigation_id": inv_id,
        "transaction_id": 230425980,
        "feedback_category": "helpful",
        "rating": 5,
        "comment": "Accurately noted multi-hop neighborhood and feature limitations.",
        "reviewer_id": "analyst_alice",
        "report_version": "ETGNN-Agent-v3.5.0",
        "evidence_references": chat_resp["cited_evidence_ids"],
    }
    fb_record = api_submit_feedback(fb_payload)
    assert fb_record["feedback_id"].startswith("fb_")
    assert fb_record["investigation_id"] == inv_id
    assert fb_record["rating"] == 5

    # 6. Test list feedback and query by investigation
    all_fb = api_list_feedback(limit=10, offset=0)
    assert all_fb["stats"]["total_feedback_count"] >= 1
    assert any(item["feedback_id"] == fb_record["feedback_id"] for item in all_fb["items"])

    inv_fb = api_get_investigation_feedback(inv_id)
    assert len(inv_fb) >= 1
    assert inv_fb[0]["feedback_id"] == fb_record["feedback_id"]

