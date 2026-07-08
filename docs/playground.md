# Web Playground

A full-stack browser UI for building, running, and A/B testing prompt chains.

## Launch

```bash
python chainforge/playground/app.py
# or: forge playground --port 8003
# or: python -m uvicorn chainforge.playground.app:app --port 8003
```

Open http://127.0.0.1:8003.

## Features

- **Chain builder** — add, remove, rename, and select steps in the sidebar;
  each step has its own prompt. Steps run in order, and any step can reference
  a previous step's result with `{{step_name.output}}`.
- **Input Variables** — a JSON object whose keys become `{{variables}}`
  available to every step. What you type here is what the chain runs on.
- **Live streaming execution** — runs go over a WebSocket; each step's card
  lights up as it executes and shows its own output, latency, tokens, and cost
  (with an automatic HTTP fallback if WebSockets are unavailable).
- **⚖️ Compare Prompts** — write two prompt variants, run both against the
  same input N times, and get side-by-side outputs plus an accuracy / tokens /
  latency / cost table and a winner verdict. Provide an optional expected
  output to enable accuracy scoring, and with 5+ runs a Mann-Whitney U
  p-value tells you whether the difference is statistically significant.
- **Provider selection that actually works** — the dropdown reflects which
  API keys are configured (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); unavailable
  providers are shown disabled with a hint. The mock provider is free and
  offline: it echoes back the exact rendered prompt each step received, so you
  can verify template substitution and chaining without spending a cent.
- **Saved chains** — save/load/delete chain definitions; they persist to disk
  (`~/.chainforge/playground_chains.json`) across restarts.

## API endpoints

The UI is a thin client over a JSON API you can call directly:

| Endpoint | Method | Description |
|---|---|---|
| `/api/run` | POST | Execute a chain; returns final output + per-link traces |
| `/api/compare` | POST | A/B test two prompts; metrics, verdict, optional p-value |
| `/api/providers` | GET | Provider availability and model lists |
| `/api/chains` | GET / POST | List / save chain definitions |
| `/api/chains/{id}` | GET / DELETE | Load / delete a saved chain |
| `/ws/run` | WebSocket | Streaming execution: one event per link |

### Run example

```bash
curl -X POST http://127.0.0.1:8003/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "chain_config": {"links": [
      {"name": "extract",   "prompt": "Extract key facts: {{input}}"},
      {"name": "summarize", "prompt": "Summarize: {{extract.output}}"}
    ]},
    "input": {"input": "AI is transforming healthcare"},
    "provider": "mock", "model": "mock-model"
  }'
```

### Compare example

```bash
curl -X POST http://127.0.0.1:8003/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_a": "Summarize briefly: {{input}}",
    "prompt_b": "You are an expert editor. Summarize in one sentence: {{input}}",
    "input": {"input": "AI is transforming healthcare"},
    "provider": "mock", "runs": 5
  }'
```

Response includes per-variant `avg_tokens`, `avg_latency_ms`, `total_cost_usd`,
optional `exact_match_rate`, a `p_value` when computable, and a human-readable
`verdict` such as *"Same accuracy — Prompt A is cheaper (73 vs 107 avg tokens)"*.

## Using real LLMs

```bash
export OPENAI_API_KEY=sk-...        # and/or ANTHROPIC_API_KEY
python chainforge/playground/app.py
```

Reload the page — the OpenAI/Anthropic options become selectable and every
run/compare then reports real token usage and real dollar cost.
