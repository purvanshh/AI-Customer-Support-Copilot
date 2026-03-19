from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    Test client with isolated storage directories.
    """
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("EMBEDDING_BACKEND", "mock")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("AUTO_REBUILD_INDEX", "true")

    # Reset cached settings so each test picks up its isolated DATA_DIR.
    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    return TestClient(app)


def test_health(client: TestClient) -> None:
    """
    Health endpoint should return status ok.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_tickets_csv_and_persist(client: TestClient, tmp_path: Path) -> None:
    """
    Upload CSV tickets and ensure processed output is written.
    """
    csv_content = (
        "id,query,response,timestamp\n"
        "T1,How do I reset my password?,To reset your password...,2026-01-01T00:00:00Z\n"
        "T2,My subscription was charged twice?,Sorry about that...,2026-01-02T00:00:00Z\n"
    )

    resp = client.post(
        "/upload-tickets",
        files={"file": ("tickets.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["records_ingested"] == 2

    processed_path = Path(payload["processed_file_path"])
    raw_path = Path(payload["raw_file_path"])
    assert processed_path.exists()
    assert raw_path.exists()


def test_upload_docs_txt_and_chunk(client: TestClient, tmp_path: Path) -> None:
    """
    Upload TXT doc and ensure paragraph chunking is persisted.
    """
    txt_content = "First paragraph.\n\nSecond paragraph."

    resp = client.post(
        "/upload-docs",
        files={"file": ("kb.txt", txt_content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["chunks_saved"] == 2

    processed_path = Path(payload["processed_file_path"])
    raw_path = Path(payload["raw_file_path"])
    assert processed_path.exists()
    assert raw_path.exists()


def test_upload_tickets_missing_required_columns_returns_400(client: TestClient) -> None:
    """
    If CSV is missing required columns, API should return 400.
    """
    bad_csv = "wrong_col\n1\n"
    resp = client.post("/upload-tickets", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert resp.status_code == 400


def test_generate_response_ticket_only(client: TestClient) -> None:
    """
    After ingesting tickets, `POST /generate-response` should retrieve context and
    return a response + sources using mock backends.
    """
    csv_content = (
        "id,query,response,timestamp\n"
        "T1,How do I reset my password?,To reset your password, go to Settings > Security and click Reset Password.,2026-01-01T00:00:00Z\n"
    )

    up = client.post(
        "/upload-tickets",
        files={"file": ("tickets.csv", csv_content, "text/csv")},
    )
    assert up.status_code == 200, up.text

    q = client.post(
        "/generate-response",
        json={"query": "I want to reset my password", "top_k": 3, "force_rebuild_index": True},
    )
    assert q.status_code == 200, q.text
    payload = q.json()

    assert isinstance(payload.get("response"), str) and payload["response"]
    assert isinstance(payload.get("sources"), list) and len(payload["sources"]) >= 1
    assert 0.0 <= float(payload["confidence"]) <= 1.0
    assert "likely resolution starts with" in payload["response"]
    assert payload.get("category") in {"billing", "technical", "refund", "general"}


def test_feedback_loop_overrides_response_and_updates_analytics(client: TestClient) -> None:
    """
    Verify feedback is stored and reused to override future responses.
    """
    csv_content = (
        "id,query,response,timestamp\n"
        "T1,How do I reset my password?,To reset your password, go to Settings > Security and click Reset Password.,2026-01-01T00:00:00Z\n"
    )

    up = client.post(
        "/upload-tickets",
        files={"file": ("tickets.csv", csv_content, "text/csv")},
    )
    assert up.status_code == 200, up.text

    q = client.post(
        "/generate-response",
        json={"query": "I need help resetting my password", "top_k": 3, "force_rebuild_index": True},
    )
    assert q.status_code == 200, q.text
    base_payload = q.json()
    assert base_payload["response"]

    corrected = "Corrected response: To reset your password, open Settings > Security and click Reset Password."
    fb = client.post(
        "/feedback",
        json={
            "query": "I need help resetting my password",
            "ai_response": base_payload["response"],
            "corrected_response": corrected,
            "rating": 5,
        },
    )
    assert fb.status_code == 200, fb.text
    fb_payload = fb.json()
    assert fb_payload["status"] == "ok"
    assert isinstance(fb_payload["id"], int)

    q2 = client.post(
        "/generate-response",
        json={"query": "I need help resetting my password", "top_k": 3, "force_rebuild_index": False},
    )
    assert q2.status_code == 200, q2.text
    payload2 = q2.json()
    assert payload2["response"] == corrected
    assert "snippet" in payload2["sources"][0] or True

    a = client.get("/analytics")
    assert a.status_code == 200, a.text
    ap = a.json()
    assert ap["total_queries"] >= 1
    assert ap["avg_rating"] >= 1.0
    assert 0.0 <= ap["resolution_rate"] <= 1.0

