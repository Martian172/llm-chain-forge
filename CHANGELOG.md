# Changelog — LLM Chain Forge

## [0.1.0] - 2024-01-01

### Added
- `Chain` — composable prompt chain with sequential and parallel execution
- `Link` — individual chain step with prompt templating
- `ChainContext` — shared state manager across chain steps
- Multi-provider support: OpenAI, Anthropic, Mock
- `Evaluator` — evaluation framework with exact match, token efficiency metrics
- `ABTest` — statistical A/B testing between chains (Mann-Whitney U test)
- `CacheManager` — in-memory and disk-backed caching with TTL
- FastAPI playground with CodeMirror editor and real-time WebSocket
- Click CLI: `forge run`, `forge eval`, `forge compare`, `forge playground`, `forge new`
- Chain serialization to/from YAML
- GitHub Actions CI for Python 3.10 and 3.11
