"""FastAPI Playground for LLM Chain Forge.

A full-stack playground over the chainforge library:

- POST /api/run       — execute a multi-step chain with real input variables
                        and a selectable provider; returns per-link traces.
- POST /api/compare   — A/B test two prompts on the same input: side-by-side
                        outputs, token/cost/latency metrics, optional
                        exact-match scoring and Mann-Whitney U significance.
- GET  /api/providers — provider availability (which API keys are configured).
- CRUD /api/chains    — save/load/delete chain definitions (persisted to disk).
- WS   /ws/run        — streaming execution: one event per link as it runs.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chainforge.core.chain import Chain
from chainforge.core.context import ChainContext
from chainforge.core.link import Link

app = FastAPI(title="LLM Chain Forge Playground", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMPLATE = Path(__file__).parent / "templates" / "index.html"

# Saved chains persist across restarts.
CHAINS_FILE = Path(os.environ.get(
    "FORGE_CHAINS_FILE", Path.home() / ".chainforge" / "playground_chains.json"
))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[str]] = {
    "mock": ["mock-model"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"],
}


def _provider_status() -> dict[str, dict[str, Any]]:
    return {
        "mock": {
            "available": True,
            "models": PROVIDER_MODELS["mock"],
            "note": "Free offline simulator — echoes the rendered prompt so you "
                    "can verify template substitution. No API key needed.",
        },
        "openai": {
            "available": bool(os.environ.get("OPENAI_API_KEY")),
            "models": PROVIDER_MODELS["openai"],
            "note": "Set OPENAI_API_KEY to enable.",
        },
        "anthropic": {
            "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "models": PROVIDER_MODELS["anthropic"],
            "note": "Set ANTHROPIC_API_KEY to enable.",
        },
    }


def _make_provider(provider_name: str, model: str):
    """Instantiate the requested provider, raising a clear error if unusable."""
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI selected but OPENAI_API_KEY is not set. "
                "Export the key and restart, or use the mock provider."
            )
        from chainforge.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)

    if provider_name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "Anthropic selected but ANTHROPIC_API_KEY is not set. "
                "Export the key and restart, or use the mock provider."
            )
        from chainforge.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)

    from chainforge.providers.mock_provider import MockProvider

    def _echo(prompt: str) -> str:
        return (
            "[mock provider — a real LLM would answer here]\n"
            "Rendered prompt it received:\n"
            f"{'-' * 40}\n{prompt}"
        )

    return MockProvider(response_fn=_echo, model=model or "mock-model")


def _build_chain(links_cfg: list[dict], provider) -> Chain:
    chain = Chain(name="playground-chain")
    for i, link_cfg in enumerate(links_cfg):
        chain.add_link(Link(
            name=link_cfg.get("name") or f"step{i + 1}",
            prompt_template=link_cfg.get("prompt") or "{{input}}",
            provider=provider,
            temperature=float(link_cfg.get("temperature", 0.7)),
        ))
    return chain


def _link_results_payload(result) -> list[dict[str, Any]]:
    return [
        {
            "link": lr.link_name,
            "output": lr.output,
            "token_usage": lr.token_usage,
            "latency_ms": lr.latency_ms,
            "cost_usd": lr.cost_usd,
            "model": lr.model,
        }
        for lr in result.link_results
    ]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChainRunRequest(BaseModel):
    chain_config: dict
    input: dict = Field(default_factory=dict)
    provider: str = "mock"
    model: str = "mock-model"


class CompareRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    input: dict = Field(default_factory=dict)
    provider: str = "mock"
    model: str = "mock-model"
    expected_output: Optional[str] = None
    runs: int = Field(default=1, ge=1, le=25)


class SaveChainRequest(BaseModel):
    name: str
    config: dict


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_chains() -> dict[str, dict]:
    if CHAINS_FILE.exists():
        try:
            return json.loads(CHAINS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_chains(chains: dict[str, dict]) -> None:
    CHAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAINS_FILE.write_text(json.dumps(chains, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def playground():
    return HTMLResponse(TEMPLATE.read_text(encoding="utf-8") if TEMPLATE.exists() else "<h1>Playground</h1>")


@app.get("/api/providers")
async def providers():
    return _provider_status()


@app.post("/api/run")
async def run_chain(req: ChainRunRequest):
    """Execute a chain config and return the result with per-link traces."""
    try:
        provider = _make_provider(req.provider, req.model)
        chain = _build_chain(req.chain_config.get("links", []), provider)
        result = await chain.run_async(req.input)
        if not result.success:
            return {"error": result.error}
        return {
            "output": result.output,
            "token_usage": result.token_usage.to_dict(),
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "link_results": _link_results_payload(result),
            "provider": req.provider,
            "model": req.model,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/compare")
async def compare_prompts(req: CompareRequest):
    """A/B test two prompts on the same input.

    Runs each prompt `runs` times, aggregates tokens/cost/latency, optionally
    scores exact-match against `expected_output`, and — when there are enough
    scored samples — reports Mann-Whitney U significance.
    """
    try:
        provider = _make_provider(req.provider, req.model)

        async def run_variant(prompt: str) -> dict[str, Any]:
            outputs, latencies, tokens, costs, scores = [], [], [], [], []
            for _ in range(req.runs):
                chain = _build_chain([{"name": "candidate", "prompt": prompt}], provider)
                result = await chain.run_async(req.input)
                if not result.success:
                    raise RuntimeError(result.error or "chain failed")
                outputs.append(result.output)
                latencies.append(result.latency_ms)
                tokens.append(result.token_usage.total_tokens)
                costs.append(result.cost_usd)
                if req.expected_output is not None:
                    scores.append(
                        1.0 if result.output.strip().lower()
                        == req.expected_output.strip().lower() else 0.0
                    )
            return {
                "output": outputs[0],
                "avg_latency_ms": statistics.mean(latencies),
                "avg_tokens": statistics.mean(tokens),
                "total_cost_usd": sum(costs),
                "exact_match_rate": statistics.mean(scores) if scores else None,
                "scores": scores,
            }

        a = await run_variant(req.prompt_a)
        b = await run_variant(req.prompt_b)

        # Significance test — only meaningful with scored, repeated runs.
        p_value = None
        if req.expected_output is not None and req.runs >= 5:
            try:
                from scipy.stats import mannwhitneyu
                if a["scores"] != b["scores"]:
                    _, p_value = mannwhitneyu(a["scores"], b["scores"])
                    p_value = float(p_value)
            except Exception:
                p_value = None

        # Verdict: exact-match rate first, then cost, then latency.
        verdict: str
        if a["exact_match_rate"] is not None and a["exact_match_rate"] != b["exact_match_rate"]:
            winner = "A" if a["exact_match_rate"] > b["exact_match_rate"] else "B"
            verdict = f"Prompt {winner} wins on accuracy"
            if p_value is not None:
                verdict += (
                    f" (p={p_value:.3f}, statistically "
                    f"{'significant' if p_value < 0.05 else 'NOT significant'})"
                )
        elif abs(a["avg_tokens"] - b["avg_tokens"]) > 1:
            winner = "A" if a["avg_tokens"] < b["avg_tokens"] else "B"
            verdict = (
                f"Same accuracy — Prompt {winner} is cheaper "
                f"({min(a['avg_tokens'], b['avg_tokens']):.0f} vs "
                f"{max(a['avg_tokens'], b['avg_tokens']):.0f} avg tokens)"
            )
        else:
            verdict = "No measurable difference between the prompts on this input"

        a.pop("scores"), b.pop("scores")
        return {"a": a, "b": b, "p_value": p_value, "verdict": verdict, "runs": req.runs}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/chains")
async def list_chains():
    chains = _load_chains()
    return {"chains": [
        {"id": cid, "name": c.get("name", cid), "links": len(c.get("config", {}).get("links", []))}
        for cid, c in chains.items()
    ]}


@app.post("/api/chains")
async def save_chain(req: SaveChainRequest):
    chains = _load_chains()
    chain_id = uuid.uuid4().hex[:8]
    chains[chain_id] = {"name": req.name, "config": req.config, "saved_at": time.time()}
    _save_chains(chains)
    return {"status": "saved", "id": chain_id, "name": req.name}


@app.get("/api/chains/{chain_id}")
async def get_chain(chain_id: str):
    chains = _load_chains()
    if chain_id not in chains:
        return {"error": "not found"}
    return chains[chain_id]


@app.delete("/api/chains/{chain_id}")
async def delete_chain(chain_id: str):
    chains = _load_chains()
    if chains.pop(chain_id, None) is None:
        return {"error": "not found"}
    _save_chains(chains)
    return {"status": "deleted"}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    """Streaming execution: emits one event per link as the chain runs."""
    await ws.accept()
    try:
        while True:
            payload = json.loads(await ws.receive_text())
            try:
                provider = _make_provider(
                    payload.get("provider", "mock"), payload.get("model", "mock-model")
                )
                links_cfg = payload.get("chain_config", {}).get("links", [])
                context = ChainContext()
                for key, value in payload.get("input", {}).items():
                    context.set(key, value)

                totals = {"tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0}
                last_output = ""
                for i, link_cfg in enumerate(links_cfg):
                    link = Link(
                        name=link_cfg.get("name") or f"step{i + 1}",
                        prompt_template=link_cfg.get("prompt") or "{{input}}",
                        provider=provider,
                        temperature=float(link_cfg.get("temperature", 0.7)),
                    )
                    await ws.send_text(json.dumps({"event": "link_start", "link": link.name}))
                    result = await link.run_async(context)
                    context.set(f"{link.name}.output", result.output)
                    last_output = result.output
                    totals["tokens"] += result.token_usage.get("total_tokens", 0)
                    totals["cost_usd"] += result.cost_usd
                    totals["latency_ms"] += result.latency_ms
                    await ws.send_text(json.dumps({
                        "event": "link_complete",
                        "link": link.name,
                        "output": result.output,
                        "token_usage": result.token_usage,
                        "latency_ms": result.latency_ms,
                        "cost_usd": result.cost_usd,
                    }))
                await ws.send_text(json.dumps({
                    "event": "done", "output": last_output, "totals": totals,
                }))
            except Exception as e:
                await ws.send_text(json.dumps({"event": "error", "detail": str(e)}))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    # Allows launching the playground directly (e.g. VS Code Run button):
    #   python chainforge/playground/app.py
    import uvicorn

    host = os.environ.get("FORGE_HOST", "127.0.0.1")
    port = int(os.environ.get("FORGE_PORT", "8003"))
    print(f"LLM Chain Forge playground: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
