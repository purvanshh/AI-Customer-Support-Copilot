from __future__ import annotations


def test_upload_query_feedback_analytics(client):
    csv_content = """Ticket ID,Customer query,Response,Timestamp,Tags
T1,How do I reset my password?,To reset your password, go to Settings > Security and click Reset Password.,2026-01-01T00:00:00Z,"Technical issue"
T2,My subscription was charged twice,Sorry about that. Please verify your billing history and ensure only one payment is pending. If you still see duplicates, contact billing support with your receipt.,2026-01-02T00:00:00Z,"Billing,Refund"
"""

    r = client.post(
        "/upload-tickets",
        files={"file": ("tickets.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["index_rebuilt"] is True
    assert payload["tickets_ingested"] == 2

    q = client.post(
        "/query",
        json={"customer_query": "I got charged twice and I want a refund", "top_k": 2},
    )
    assert q.status_code == 200, q.text
    qpayload = q.json()
    assert "suggestion_text" in qpayload
    assert qpayload["generation_id"] > 0
    assert 0.0 <= qpayload["confidence"] <= 1.0

    gen_id = qpayload["generation_id"]
    fb = client.post("/feedback", json={"generation_id": gen_id, "user_action": "accepted"})
    assert fb.status_code == 200, fb.text

    a = client.get("/analytics")
    assert a.status_code == 200, a.text
    ap = a.json()
    assert ap["total_queries"] >= 1
    assert ap["accuracy"] >= 0.0


def test_query_without_upload_returns_400(client):
    # New test process starts with empty DB/vector store for this fixture.
    q = client.post("/query", json={"customer_query": "Hello?"})
    assert q.status_code == 400

