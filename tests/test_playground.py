"""Tests for the playground API (chainforge.playground.app)."""
import pytest
from fastapi.testclient import TestClient

from chainforge.playground import app as playground_app
from chainforge.playground.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate saved-chain persistence per test
    monkeypatch.setattr(playground_app, "CHAINS_FILE", tmp_path / "chains.json")
    return TestClient(app)


class TestProviders:
    def test_lists_providers_with_availability(self, client):
        data = client.get("/api/providers").json()
        assert data["mock"]["available"] is True
        assert "models" in data["openai"]
        assert "models" in data["anthropic"]


class TestRun:
    def test_multi_link_run_uses_input_and_chains_outputs(self, client):
        resp = client.post("/api/run", json={
            "chain_config": {"links": [
                {"name": "extract", "prompt": "Extract facts: {{input}}"},
                {"name": "summarize", "prompt": "Summarize: {{extract.output}}"},
            ]},
            "input": {"input": "UNIQUE_MARKER_42"},
            "provider": "mock",
            "model": "mock-model",
        }).json()
        assert "error" not in resp
        assert len(resp["link_results"]) == 2
        # The user's input reached step 1's rendered prompt
        assert "UNIQUE_MARKER_42" in resp["link_results"][0]["output"]
        # Step 2 received step 1's output (chaining works)
        assert "Extract facts" in resp["link_results"][1]["output"]
        assert resp["token_usage"]["total_tokens"] > 0

    def test_openai_without_key_returns_clear_error(self, client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        resp = client.post("/api/run", json={
            "chain_config": {"links": [{"name": "s1", "prompt": "{{input}}"}]},
            "input": {"input": "hi"},
            "provider": "openai",
            "model": "gpt-4o-mini",
        }).json()
        assert "OPENAI_API_KEY" in resp.get("error", "")


class TestCompare:
    def test_compare_returns_metrics_and_verdict(self, client):
        resp = client.post("/api/compare", json={
            "prompt_a": "Short: {{input}}",
            "prompt_b": "This is a deliberately much longer prompt wording: {{input}}",
            "input": {"input": "hello world"},
            "provider": "mock",
            "runs": 2,
        }).json()
        assert "error" not in resp
        assert resp["runs"] == 2
        assert resp["a"]["avg_tokens"] < resp["b"]["avg_tokens"]
        assert "Prompt A" in resp["verdict"]  # A is cheaper

    def test_compare_with_expected_output_scores_accuracy(self, client):
        resp = client.post("/api/compare", json={
            "prompt_a": "{{input}}",
            "prompt_b": "{{input}} extra",
            "input": {"input": "x"},
            "provider": "mock",
            "expected_output": "will not match mock output",
            "runs": 1,
        }).json()
        assert resp["a"]["exact_match_rate"] == 0.0
        assert resp["b"]["exact_match_rate"] == 0.0


class TestSavedChains:
    def test_save_load_delete_roundtrip(self, client):
        saved = client.post("/api/chains", json={
            "name": "my-chain",
            "config": {"links": [{"name": "s1", "prompt": "{{input}}"}]},
        }).json()
        assert saved["status"] == "saved"
        cid = saved["id"]

        listed = client.get("/api/chains").json()
        assert any(c["id"] == cid for c in listed["chains"])

        loaded = client.get(f"/api/chains/{cid}").json()
        assert loaded["name"] == "my-chain"

        assert client.delete(f"/api/chains/{cid}").json()["status"] == "deleted"
        assert client.get(f"/api/chains/{cid}").json().get("error") == "not found"


class TestStreaming:
    def test_ws_run_emits_per_link_events(self, client):
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({
                "chain_config": {"links": [
                    {"name": "s1", "prompt": "A: {{input}}"},
                    {"name": "s2", "prompt": "B: {{s1.output}}"},
                ]},
                "input": {"input": "stream me"},
                "provider": "mock",
            })
            events = [ws.receive_json() for _ in range(5)]
        kinds = [e["event"] for e in events]
        assert kinds == ["link_start", "link_complete", "link_start", "link_complete", "done"]
        assert events[-1]["totals"]["tokens"] > 0
