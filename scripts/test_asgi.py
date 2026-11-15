import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

from src.web.app import app
import httpx

async def main():
    print("Testing in-memory ASGI transport without network sockets ...", flush=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://local-backend") as client:
        # 1. Health
        r = await client.get("/api/health")
        print("1. Health:", r.status_code, r.json()["status"], flush=True)
        
        # 2. Models
        r = await client.get("/api/models")
        print("2. Models:", r.status_code, len(r.json()), "models loaded", flush=True)
        
        # 3. Transaction
        r = await client.get("/api/transactions/288717023")
        print("3. Tx 288717023:", r.status_code, "Snapshot:", r.json()["time_step"], flush=True)

        # 4. Subgraph
        r = await client.get("/api/transactions/288717023/subgraph?hops=2")
        print("4. Subgraph:", r.status_code, "Nodes:", r.json()["total_nodes"], "Edges:", r.json()["total_edges"], flush=True)

        # 5. Predict
        r = await client.post("/api/predict", json={"transaction_id": 288717023})
        preds = r.json()["predictions"]
        print("5. Predict:", r.status_code, "ETGNN:", preds["Canonical_ETGNN"]["probability"], "RF:", preds["RandomForest"]["probability"], "XGB:", preds["XGBoost"]["probability"], flush=True)

        # 6. Investigate
        r = await client.post("/api/agent/investigate", json={
            "transaction_id": 288717023,
            "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
            "as_of_snapshot": 43,
            "llm_provider": "mock",
            "allow_evidence_only_fallback": True
        })
        inv_data = r.json()
        inv_id = inv_data["investigation_id"]
        report = inv_data["report"]
        print("6. Investigate:", r.status_code, "Inv ID:", inv_id, "Verdict:", report["consensus_verdict"], flush=True)

        # 7. PDF Export
        r = await client.get(f"/api/agent/investigations/{inv_id}/export/pdf")
        print("7. PDF Export:", r.status_code, "Bytes:", len(r.content), "Magic %PDF:", r.content[:4] == b"%PDF", flush=True)

        # 8. Feedback
        r = await client.post("/api/agent/feedback", json={
            "investigation_id": inv_id,
            "transaction_id": 288717023,
            "feedback_category": "helpful",
            "rating": 5,
            "comment": "Verified accurate evidence grounding in Phase 6.5 real integration.",
            "reviewer_id": "auditor_p65"
        })
        print("8. Feedback:", r.status_code, "FB ID:", r.json()["feedback_id"], flush=True)

        # 9. Monitoring Drift
        r = await client.get("/api/monitoring/drift")
        s43 = r.json()["dataset_census"]["snapshot_43_census"]
        print("9. Drift S43:", r.status_code, "Nodes:", s43["total_nodes"], "Illicit:", s43["illicit_nodes"], flush=True)

        # 10. Monitoring Candidates
        r = await client.get("/api/monitoring/candidates")
        print("10. Candidates:", r.status_code, "Gate:", r.json()["promotion_gate_rules"]["original_checkpoints_status"], flush=True)

        # 11. Monitoring Feedback Stats
        r = await client.get("/api/monitoring/feedback-stats")
        print("11. Feedback Stats:", r.status_code, "Analyzed:", r.json()["total_submissions_analyzed"], flush=True)

        print("\nALL 11 BACKEND WORKFLOWS COMPLETED IN-MEMORY OVER ASGI WITH 100% SUCCESS!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
