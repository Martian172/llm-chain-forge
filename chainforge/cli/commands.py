"""CLI for LLM Chain Forge."""
import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()


@click.group()
def cli():
    """⚒️  LLM Chain Forge — Build, Test & Optimize LLM Prompt Chains."""
    pass


@cli.command("run")
@click.argument("chain_file", type=click.Path(exists=True))
@click.option("--input", "-i", "input_str", default=None, help="Input JSON string")
@click.option("--provider", default="mock", show_default=True)
@click.option("--model", default="mock-model", show_default=True)
def run_chain(chain_file, input_str, provider, model):
    """Run a chain from a YAML config file."""
    import json, yaml
    from chainforge.core.chain import Chain

    with open(chain_file) as f:
        config = yaml.safe_load(f)

    input_data = json.loads(input_str) if input_str else {"input": "Hello, world!"}
    console.print(f"[bold purple]⚒️  Running chain:[/bold purple] {chain_file}")

    chain = Chain.from_dict(config)
    result = chain.run(input_data)

    console.print(f"\n[bold green]Output:[/bold green]\n{result.output}")
    console.print(f"\n[dim]Latency: {result.latency_ms:.0f}ms | Cost: ${result.cost:.5f}[/dim]")


@cli.command("eval")
@click.argument("chain_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
def eval_chain(chain_file, test_file):
    """Evaluate a chain against test cases."""
    import yaml
    from chainforge.core.chain import Chain
    from chainforge.evaluation.evaluator import Evaluator

    with open(chain_file) as f:
        config = yaml.safe_load(f)
    chain = Chain.from_dict(config)
    test_cases = Evaluator.from_jsonl(test_file)

    console.print(f"[bold purple]Evaluating {len(test_cases)} test cases...[/bold purple]")
    evaluator = Evaluator()
    report = evaluator.evaluate(chain, test_cases)
    console.print(report.summary())


@cli.command("compare")
@click.argument("chain_a", type=click.Path(exists=True))
@click.argument("chain_b", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
def compare_chains(chain_a, chain_b, test_file):
    """A/B test two chains on the same test cases."""
    import yaml
    from chainforge.core.chain import Chain
    from chainforge.evaluation.evaluator import Evaluator
    from chainforge.evaluation.ab_test import ABTest

    with open(chain_a) as f: ca = Chain.from_dict(yaml.safe_load(f))
    with open(chain_b) as f: cb = Chain.from_dict(yaml.safe_load(f))
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
