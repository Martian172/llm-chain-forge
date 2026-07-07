"""FastAPI Playground for LLM Chain Forge."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="LLM Chain Forge Playground", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMPLATE = Path(__file__).parent / "templates" / "index.html"
_chains: dict[str, Any] = {}
_clients: list[WebSocket] = []


class ChainRunRequest(BaseModel):
    chain_config: dict
    input: dict
    provider: str = "mock"
    model: str = "mock-model"


class SaveChainRequest(BaseModel):
    name: str
    config: dict


@app.get("/", response_class=HTMLResponse)
async def playground():
    return HTMLResponse(TEMPLATE.read_text(encoding="utf-8") if TEMPLATE.exists() else "<h1>Playground</h1>")


@app.post("/api/run")
async def run_chain(req: ChainRunRequest):
    """Execute a chain config and return result."""
    from chainforge.providers.mock_provider import MockProvider
    from chainforge.core.chain import Chain
    from chainforge.core.link import Link

    try:
        provider = MockProvider()
        chain = Chain(name="playground-chain")
        for link_cfg in req.chain_config.get("links", []):
            chain.add_link(Link(
                name=link_cfg.get("name", "step"),
                prompt_template=link_cfg.get("prompt", "{{input}}"),
                provider=provider,
            ))
        result = await chain.run_async(req.input)
        return {
            "output": result.output,
            "token_usage": result.token_usage.to_dict(),
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/chains")
async def list_chains():
    return {"chains": list(_chains.keys())}


@app.post("/api/chains")
async def save_chain(req: SaveChainRequest):
    _chains[req.name] = req.config
    return {"status": "saved", "name": req.name}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)
            await ws.send_text(json.dumps({"status": "processing", "input": payload}))
            await ws.send_text(json.dumps({"status": "done", "output": "[Mock] " + str(payload)}))
    except WebSocketDisconnect:
        if ws in _clients:
            _clients.remove(ws)
