"""Contract smoke test for the demo API (backend.md §9.8). Needs runs/p9/catalog.sqlite."""
from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_api():
    assert c.get("/api/health").json()["status"] == "ok"
    ex = c.get("/api/examples").json()
    assert len(ex) == 20

    r = c.get("/api/search", params={"q": ex[0]["query"]}).json()
    assert r["ground_truth"] and r["items"] and r["reranker"] is None
    assert any(h["human_label"] for h in r["items"])
    assert c.get("/api/search", params={"q": "zzzqqq xxjjkw"}).json()["items"] == []
    assert c.get("/api/search", params={"q": 'a" OR NEAR(*'}).status_code == 200  # FTS syntax neutralized

    for bad in ("", " ", "x" * 201):
        r = c.get("/api/search", params={"q": bad})
        assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"

    pid = c.get("/api/search", params={"q": ex[0]["query"]}).json()["items"][0]["product_id"]
    assert c.get(f"/api/products/{pid}").json()["product_id"] == pid
    assert c.get(f"/api/products/{pid}/looks-like").status_code == 200
    assert c.get(f"/api/products/{pid}/alternatives").json()["error"]["code"] == "MODEL_NOT_READY"
    assert c.get("/api/products/NOPE").json()["error"]["code"] == "PRODUCT_NOT_FOUND"
    assert c.get("/api/nope").status_code == 404
    assert c.get("/api/models").json() == []
