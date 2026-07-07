"""CLI for LLM Chain Forge."""
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box

# Windows consoles default to a legacy codepage (e.g. cp1252) that cannot
# encode the unicode symbols in CLI output; force UTF-8 with replacement so
# output never crashes regardless of terminal settings.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()


@click.group()
def cli():
    """⚒️  LLM Chain Forge — Build, Test & Optimize LLM Prompt Chains."""
    pass


def _make_provider(provider_name: str, model: str):
    """Instantiate a provider by name (mock / openai / anthropic)."""
    if provider_name == "openai":
        from chainforge.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if provider_name == "anthropic":
        from chainforge.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    from chainforge.providers.mock_provider import MockProvider
    return MockProvider(model=model)


def _load_chain(chain_file: str, provider_name: str, model: str):
    """Load a YAML chain and attach the default provider to links without one."""
    from chainforge.core.chain import Chain
    from chainforge.core.link import Link

    default_provider = _make_provider(provider_name, model)
    providers_map = {provider_name: default_provider, "default": default_provider}
    chain = Chain.from_yaml(chain_file, providers_map)
    for step in chain.links:
        if isinstance(step, Link) and step.provider is None:
            step.provider = default_provider
    return chain


@cli.command("run")
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--input", "-i", "input_str", default=None, help="Input JSON string")
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default="mock-model", show_default=True)
def run_chain(chain_file, input_str, provider, model):
    """Run a chain from a YAML config file."""
    import json

    input_data = json.loads(input_str) if input_str else {"input": "Hello, world!"}
    console.print(f"[bold purple]⚒️  Running chain:[/bold purple] {chain_file}")

    chain = _load_chain(chain_file, provider, model)
    result = chain.run(input_data)

    if not result.success:
        console.print(f"\n[bold red]Chain failed:[/bold red] {result.error}")
        raise SystemExit(1)

    console.print(f"\n[bold green]Output:[/bold green]\n{result.output}")
    console.print(
        f"\n[dim]Latency: {result.latency_ms:.0f}ms | "
        f"Tokens: {result.token_usage.total_tokens} | "
        f"Cost: ${result.cost_usd:.5f}[/dim]"
    )


@cli.command("eval")
@click.argument("chain_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default="mock-model", show_default=True)
def eval_chain(chain_file, test_file, provider, model):
    """Evaluate a chain against test cases."""
    from chainforge.evaluation.evaluator import Evaluator

    chain = _load_chain(chain_file, provider, model)
    test_cases = Evaluator.from_jsonl(test_file)

    console.print(f"[bold purple]Evaluating {len(test_cases)} test cases...[/bold purple]")
    evaluator = Evaluator()
    report = evaluator.evaluate(chain, test_cases)
    console.print(report.summary())


@cli.command("compare")
@click.argument("chain_a", type=click.Path(exists=True))
@click.argument("chain_b", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--provider", default="mock", show_default=True,
              type=click.Choice(["mock", "openai", "anthropic"]))
@click.option("--model", default="mock-model", show_default=True)
def compare_chains(chain_a, chain_b, test_file, provider, model):
    """A/B test two chains on the same test cases."""
    from chainforge.evaluation.evaluator import Evaluator
    from chainforge.evaluation.ab_test import ABTest

    ca = _load_chain(chain_a, provider, model)
    cb = _load_chain(chain_b, provider, model)
    cases = Evaluator.from_jsonl(test_file)

    ab = ABTest()
    report = ab.compare(ca, cb, cases)
    console.print(report.summary())


@cli.command("playground")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def playground(host, port):
    """Launch the visual chain playground in your browser."""
    import webbrowser, uvicorn
    from chainforge.playground.app import app
    console.print(f"[bold purple]⚒️  Chain Forge Playground[/bold purple]")
    console.print(f"[cyan]  → http://{host}:{port}[/cyan]")
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command("new")
@click.argument("name")
def new_chain(name):
    """Scaffold a new chain project."""
    import os
    os.makedirs(name, exist_ok=True)
    chain_yaml = f"""\
name: {name}
links:
  - name: step1
    prompt: "You are a helpful assistant.\\n\\n{{{{input}}}}"
    model: gpt-4o-mini
    temperature: 0.7
"""
    with open(f"{name}/chain.yaml", "w") as f:
        f.write(chain_yaml)
    console.print(f"[green]✓ Created chain project:[/green] {name}/")
    console.print(f"  [cyan]forge run {name}/chain.yaml[/cyan]")


def main():
    cli()


if __name__ == "__main__":
    main()
