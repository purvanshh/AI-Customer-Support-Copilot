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

