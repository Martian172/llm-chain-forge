# LLM Chain Forge ⚒️ - Build, Test & Optimize LLM Prompt Chains

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/PyPI-v0.1.0-orange.svg?style=for-the-badge&logo=pypi)](https://pypi.org/project/llm-chain-forge/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=for-the-badge&logo=discord)](https://discord.gg/llm-chain-forge)
[![CI](https://img.shields.io/github/actions/workflow/status/yourusername/llm-chain-forge/ci.yml?style=for-the-badge&label=CI)](https://github.com/yourusername/llm-chain-forge/actions)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)

**A lightweight yet powerful Python framework for building, testing, and optimizing LLM prompt chains with built-in evaluation, intelligent caching, and a stunning web playground.**

[🚀 Quick Start](#-quick-start) • [📖 Docs](#-documentation) • [🎮 Playground](#-web-playground) • [🔌 Providers](#-provider-support) • [📊 Evaluation](#-evaluation-framework)

---

```
  ╔══════════════════════════════════════════════════════════╗
  ║   Input ──► [Link 1] ──► [Link 2] ──► [Link 3] ──► Output   ║
  ║                              │                              ║
  ║                         [Branch A]                          ║
  ║                              │                              ║
  ║                         [Branch B]                          ║
  ╚══════════════════════════════════════════════════════════╝
```

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔗 Chain Building
- **Visual chain builder** in the web playground
- **Drag-and-drop** link reordering
- **Branching logic** with conditional routing
- **Parallel execution** of independent links
- **YAML/JSON config** for version control

### 🤖 Multi-Provider Support
- **OpenAI** (GPT-3.5, GPT-4, GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3 Haiku/Sonnet/Opus)
- **Google** (Gemini Pro, Gemini Ultra)
- **Local models** via Ollama
- **Mock provider** for testing

</td>
<td width="50%">

### 📊 Evaluation & Testing
- **Built-in metrics**: ROUGE, exact match, semantic similarity
- **A/B testing** with statistical significance
- **Cost analysis** across providers
- **Latency benchmarking**
- **Test case management**

### ⚡ Performance
- **Intelligent caching** (in-memory + disk)
- **Async/await** support throughout
- **Streaming** responses
- **Token counting** before execution
- **Cost estimation** before running

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```bash
pip install llm-chain-forge

# With all providers
pip install llm-chain-forge[all]

# With specific provider
pip install llm-chain-forge[openai]
pip install llm-chain-forge[anthropic]
```

### Your First Chain (30 seconds)

```python
from chainforge import Chain, Link, Prompt
from chainforge.providers import OpenAIProvider

# Set up provider
provider = OpenAIProvider(api_key="sk-...")

# Build a RAG summarization chain
chain = Chain(name="rag-summarizer")

chain.add_link(Link(
    name="extract_keywords",
    prompt_template=Prompt("""
        Extract the top 5 keywords from this text. Return as comma-separated list.
        
        Text: {input_text}
        
        Keywords:
    """),
    provider=provider,
    model="gpt-4o-mini"
))

chain.add_link(Link(
    name="summarize",
    prompt_template=Prompt("""
        Summarize the following text in 3 sentences. 
        Focus on these key themes: {extract_keywords.output}
        
        Text: {input_text}
        
        Summary:
    """),
    provider=provider,
    model="gpt-4o-mini"
))

# Run the chain
result = chain.run({"input_text": "Your long document here..."})

print(result.output)           # Final summary
print(result.token_usage)      # {'prompt': 245, 'completion': 87}
print(f"Cost: ${result.cost:.4f}")    # Cost: $0.0003
print(f"Latency: {result.latency_ms}ms")  # Latency: 1243ms
```

---

## 📦 Building a Full RAG Chain

```python
from chainforge import Chain, Link, Prompt
from chainforge.providers import OpenAIProvider, AnthropicProvider
from chainforge.cache import CacheManager
from chainforge.evaluation import Evaluator, TestCase

# Initialize with caching
cache = CacheManager(backend="disk", ttl=3600)
provider = OpenAIProvider(api_key="sk-...", cache=cache)

chain = Chain(name="rag-pipeline")

# Step 1: Query reformulation
chain.add_link(Link(
    name="reformulate_query",
    prompt_template=Prompt("""
        Reformulate this user query to be more specific for document retrieval.
        
        Original query: {user_query}
        Chat history: {chat_history}
        
        Reformulated query:
    """),
    provider=provider,
    model="gpt-4o-mini",
    temperature=0.3
))

# Step 2: Generate answer (uses output from step 1)
chain.add_link(Link(
    name="generate_answer",
    prompt_template=Prompt("""
        Answer the user's question based on the retrieved context.
        
        Query: {reformulate_query.output}
        Context: {retrieved_docs}
        
        Provide a comprehensive answer with citations [1], [2], etc.
        
        Answer:
    """),
    provider=provider,
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
))

# Step 3: Fact-check with a different provider
chain.add_link(Link(
    name="fact_check",
    prompt_template=Prompt("""
        Review this answer for factual accuracy.
        Rate confidence: HIGH/MEDIUM/LOW
        
        Answer: {generate_answer.output}
        Source context: {retrieved_docs}
        
        Fact-check report:
    """),
    provider=AnthropicProvider(api_key="sk-ant-..."),
    model="claude-3-haiku-20240307",
    temperature=0.1
))

# Run with full context
result = chain.run({
    "user_query": "What are the main causes of inflation?",
    "chat_history": "[]",
    "retrieved_docs": "Federal Reserve reports show..."
})

# Visualize the chain
print(chain.visualize())
```

Output:
```
┌─────────────────────────────────────────────┐
│           RAG Pipeline Chain                │
├─────────────────────────────────────────────┤
│                                             │
│   [INPUT] ──► reformulate_query             │
│                    │                        │
│                    ▼                        │
│              generate_answer                │
│                    │                        │
│                    ▼                        │
│              fact_check ──► [OUTPUT]        │
│                                             │
│   Provider: OpenAI + Anthropic              │
│   Est. Cost: $0.0045/run                    │
└─────────────────────────────────────────────┘
```

---

## 🌿 Branching Chains

```python
from chainforge import Chain, Link, Prompt

chain = Chain(name="adaptive-responder")

# Route based on query complexity
chain.add_link(Link(name="classify_complexity", ...))

# Branch: simple queries → fast model, complex → powerful model
chain.branch(
    condition=lambda ctx: ctx.get("classify_complexity.output") == "SIMPLE",
    if_true=Link(name="fast_response", model="gpt-4o-mini", ...),
    if_false=Link(name="deep_response", model="gpt-4o", ...)
)

result = chain.run({"user_input": "What is 2+2?"})
```

---

## 📊 Evaluation Framework

```python
from chainforge.evaluation import Evaluator, TestCase

# Define test cases
test_cases = [
    TestCase(
        input={"user_query": "What is machine learning?"},
        expected_output="Machine learning is a subset of AI...",
        metadata={"category": "definition"}
    ),
    TestCase(
        input={"user_query": "Explain neural networks"},
        expected_output="Neural networks are computing systems...",
        metadata={"category": "technical"}
    ),
]

# Run evaluation
evaluator = Evaluator(metrics=["exact_match", "rouge_l", "semantic_similarity"])
report = evaluator.evaluate(chain, test_cases)

print(report.summary())
```

```
╔══════════════════════════════════════════════════════╗
║              Evaluation Report                        ║
╠══════════════════════════════════════════════════════╣
║  Test Cases:     10/10 passed                         ║
║  ROUGE-L Score:  0.847                                ║
║  Semantic Sim:   0.923                                ║
║  Avg Latency:    1,247ms                              ║
║  Total Cost:     $0.0234                              ║
║  Cost/Case:      $0.0023                              ║
╚══════════════════════════════════════════════════════╝
```

---

## 🔬 A/B Testing

```python
from chainforge.evaluation import ABTest

# Compare GPT-4o-mini vs Claude Haiku on same task
chain_a = Chain(name="openai-chain")
chain_b = Chain(name="anthropic-chain")

ab_test = ABTest(chain_a=chain_a, chain_b=chain_b)
report = ab_test.run(test_cases, n_samples=50)

print(report.winner)          # "chain_b"
print(report.p_value)         # 0.023 (statistically significant)
print(report.effect_size)     # 0.34 (Cohen's d)
print(report.cost_savings)    # "Chain B saves 34% on cost"
```

---

## 🎮 Web Playground

Launch the beautiful interactive playground:

```bash
forge playground
# Opens http://localhost:8000
```

Or programmatically:

```python
from chainforge.playground import launch

launch(chain=my_chain, port=8000)
```

**Playground features:**
- 🎨 Dark purple/blue theme with neon accents
- ✏️ Monaco code editor with syntax highlighting
- 🔗 Visual chain builder with drag-and-drop
- 📊 Real-time token usage and cost tracking
- 🔄 Side-by-side A/B comparison view
- 📈 Evaluation dashboard with charts
- 💾 Chain save/load from browser

---

## 🔌 Provider Support

| Provider | Models | Streaming | Function Calling | Cost Tracking |
|----------|--------|-----------|-----------------|---------------|
| **OpenAI** | gpt-3.5-turbo, gpt-4, gpt-4o, gpt-4o-mini | ✅ | ✅ | ✅ |
| **Anthropic** | claude-3-haiku, claude-3-sonnet, claude-3-opus | ✅ | ✅ | ✅ |
| **Google** | gemini-pro, gemini-ultra | ✅ | ✅ | ✅ |
| **Ollama** | llama3, mistral, codellama, phi3 | ✅ | ❌ | ✅ |
| **Mock** | configurable | ✅ | ✅ | ✅ |

### Pricing Reference (per 1K tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o-mini | $0.00015 | $0.00060 |
| gpt-4o | $0.00500 | $0.01500 |
| gpt-4 | $0.03000 | $0.06000 |
| claude-3-haiku | $0.00025 | $0.00125 |
| claude-3-sonnet | $0.00300 | $0.01500 |
| claude-3-opus | $0.01500 | $0.07500 |

---

## ⚔️ LLM Chain Forge vs LangChain

| Feature | **LLM Chain Forge** | LangChain |
|---------|---------------------|-----------|
| Learning curve | 🟢 **5 min** | 🔴 2-3 hours |
| Bundle size | 🟢 **~50KB** | 🔴 ~10MB |
| Built-in A/B testing | 🟢 **Yes** | 🔴 No |
| Web playground | 🟢 **Built-in** | 🔴 Third-party |
| Cost tracking | 🟢 **Per-link** | 🟡 Aggregate |
| Caching | 🟢 **Auto + disk** | 🟡 Manual |
| YAML chains | 🟢 **Full support** | 🟡 Partial |
| Type safety | 🟢 **Pydantic v2** | 🟡 Mixed |
| Async | 🟢 **Native** | 🟡 Wrapper |
| Testing utilities | 🟢 **Built-in** | 🔴 Separate |

---

## 🏗️ Architecture

```
llm-chain-forge/
├── chainforge/
│   ├── core/
│   │   ├── chain.py        # Chain orchestration engine
│   │   ├── link.py         # Individual prompt steps
│   │   └── context.py      # Execution state management
│   ├── providers/
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── mock_provider.py
│   ├── evaluation/
│   │   ├── evaluator.py    # Metrics & evaluation
│   │   └── ab_test.py      # Statistical A/B testing
│   ├── cache/
│   │   └── cache_manager.py
│   └── playground/
│       ├── app.py          # FastAPI server
│       └── templates/
│           └── index.html  # Beautiful UI
├── examples/
├── tests/
└── docs/
```

---

## 🛠️ CLI Usage

```bash
# Run a chain from YAML
forge run examples/chains/summarization.yaml \
    --input '{"text": "Your document here..."}'

# Evaluate a chain
forge eval examples/chains/summarization.yaml tests/test_cases.json

# A/B test two chains
forge compare chain_a.yaml chain_b.yaml --samples 100

# Launch playground
forge playground --port 8000

# Scaffold new project
forge new my-awesome-chain
```

---

## 📖 Documentation

- [📚 Quick Start Guide](docs/quickstart.md)
- [🔌 Provider Configuration](docs/providers.md)
- [📊 Evaluation Guide](docs/evaluation.md)
- [🌐 Playground Guide](docs/playground.md)
- [🔗 Chain YAML Format](docs/yaml-format.md)

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/yourusername/llm-chain-forge
cd llm-chain-forge
pip install -e ".[dev]"
make test
```

---

## 📄 License

MIT © 2024 LLM Chain Forge Contributors

---

<div align="center">

**If you find this useful, give it a ⭐ on GitHub!**

Made with ❤️ by the open-source community

</div>
