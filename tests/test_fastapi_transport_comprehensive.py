import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.web.app import (
    app,
    AgentInvestigateRequest,
    AgentInvestigateContextRequest,
    AgentRetrieveRequest,
    api_agent_investigate,
    api_get_investigation,
    api_get_investigation_events,
)
from tests.conftest import LOOPBACK_SOCKETPAIR_AVAILABLE, ENVIRONMENT_BLOCKED_REASON


def test_agent_investigate_request_schema_validation():
    """Verify pydantic model validation on AgentInvestigateRequest."""
    # Negative transaction_id raises ValidationError
    with pytest.raises(ValidationError):
        AgentInvestigateRequest(transaction_id=-1)

    # Invalid snapshot_id (>49) raises ValidationError
    with pytest.raises(ValidationError):
        AgentInvestigateRequest(transaction_id=100, as_of_snapshot=50)

    # Valid request succeeds
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        as_of_snapshot=1,
        llm_provider="mock",
        allow_evidence_only_fallback=True,
    )
    assert req.transaction_id == 230425980
    assert req.llm_provider == "mock"


def test_direct_handler_api_agent_investigate():
    """Directly test the endpoint handler function without network/loopback socket overhead."""
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        question="Verify transaction integrity",
        llm_provider="mock",
    )
    res = api_agent_investigate(req)
    assert "investigation_id" in res
    assert res["status"] == "COMPLETED"
    assert res["report"]["target_transaction_id"] == 230425980
    assert res["report"]["generation_mode"] == "SIMULATED"
    assert "predicted_class" in res["report"]
    assert "model_probability" in res["report"]


def test_direct_handler_get_investigation_not_found():
    """Verify 404 HTTPException when investigation ID does not exist."""
    with pytest.raises(HTTPException) as exc_info:
        api_get_investigation("inv_nonexistent_99999999")
    assert exc_info.value.status_code == 404


def test_direct_handler_get_investigation_events():
    """Verify retrieval of observable telemetry events via endpoint handler."""
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        llm_provider="mock",
    )
    res = api_agent_investigate(req)
    inv_id = res["investigation_id"]

    events = api_get_investigation_events(inv_id)
    assert isinstance(events, list)
    assert len(events) >= 3
    event_types = [e["event_type"] for e in events]
    assert "INVESTIGATION_STARTED" in event_types
    assert "INVESTIGATION_COMPLETED" in event_types


@pytest.mark.skipif(not LOOPBACK_SOCKETPAIR_AVAILABLE, reason=ENVIRONMENT_BLOCKED_REASON)
def test_fastapi_asgi_testclient_investigate_transport():
    """Test full HTTP ASGI transport via TestClient (runs outside network-isolated sandbox)."""
    with TestClient(app) as client:
        payload = {
            "transaction_id": 230425980,
            "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
            "llm_provider": "mock",
        }
        res = client.post("/api/agent/investigate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "COMPLETED"
        assert data["report"]["target_transaction_id"] == 230425980

        # Query investigation
        inv_id = data["investigation_id"]
        res_get = client.get(f"/api/agent/investigations/{inv_id}")
        assert res_get.status_code == 200
        assert res_get.json()["investigation_id"] == inv_id

        # Query events
        res_events = client.get(f"/api/agent/investigations/{inv_id}/events")
        assert res_events.status_code == 200
        assert len(res_events.json()) >= 3


@pytest.mark.skipif(not LOOPBACK_SOCKETPAIR_AVAILABLE, reason=ENVIRONMENT_BLOCKED_REASON)
def test_fastapi_asgi_testclient_404_handling():
    """Test 404 response for nonexistent investigation over HTTP ASGI transport."""
    with TestClient(app) as client:
        res = client.get("/api/agent/investigations/inv_does_not_exist_at_all")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

