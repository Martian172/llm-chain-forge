# Web Playground

A browser UI for building and running chains interactively.

## Launch

```bash
forge playground --port 8003
# or directly:
python -m uvicorn chainforge.playground.app:app --port 8003
```

Open http://127.0.0.1:8003.

## Features

- **Chain builder** — add/remove/reorder links, set model and temperature per link
- **Prompt editor** — edit `{{variable}}` templates with live preview
- **Output panel** — response text plus token usage, cost, and latency
- **Runs on the mock provider** — experiment freely with zero API spend

## API endpoints

The playground UI is a thin client over a JSON API you can call directly:

| Endpoint | Method | Description |
|---|---|---|
| `/api/run` | POST | Execute a chain config, return output + usage |
| `/api/chains` | GET / POST | List / save chain definitions |
| `/api/chains/{id}` | DELETE | Delete a saved chain |
| `/ws/run` | WebSocket | Streaming chain execution |

Example:

```bash
curl -X POST http://127.0.0.1:8003/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "chain_config": {"links": [
      {"name": "expand", "prompt": "Expand: {{input}}"},
      {"name": "summarize", "prompt": "Summarize: {{expand.output}}"}
    ]},
    "input": {"input": "AI in healthcare"}
  }'
```
