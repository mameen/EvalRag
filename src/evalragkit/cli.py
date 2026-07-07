"""EvalRag CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import typer
except ImportError:
    print("CLI requires typer: pip install evalragkit[cli]", file=sys.stderr)
    raise SystemExit(1)

from evalragkit.core.experiment import Experiment
from evalragkit.exploration.reporter import Reporter

app = typer.Typer(help="EvalRag - composable RAG evaluation")


@app.command()
def run(
    config: Path = typer.Argument(..., help="Experiment config JSON"),
    output: Path = typer.Option("results.json", help="Output path for results"),
):
    """Run an experiment from a config file."""
    from evalragkit.registry import build_experiment

    cfg = json.loads(config.read_text())
    experiment, dataset = build_experiment(cfg)

    if "ingest" in cfg:
        for path in cfg["ingest"]:
            experiment.ingest(path)

    result = experiment.run(dataset)
    Experiment.save_result(result, str(output))
    typer.echo(f"Results saved to {output}")


@app.command()
def compare(
    files: list[Path] = typer.Argument(..., help="Result JSON files to compare"),
    fmt: str = typer.Option("table", help="Output format: table or json"),
):
    """Compare experiment results side by side."""
    results = [Experiment.load_result(str(f)) for f in files]
    if fmt == "json":
        typer.echo(Reporter.to_json(results))
    else:
        typer.echo(Reporter.to_table(results))


@app.command()
def datasets():
    """List available datasets."""
    from evalragkit.datasets.registry import REGISTRY

    for name, info in REGISTRY.items():
        typer.echo(f"  {name}: {info['description']}")


@app.command()
def download(name: str = typer.Argument(..., help="Dataset name")):
    """Download a dataset to local cache."""
    from evalragkit.datasets.registry import download as dl

    path = dl(name)
    typer.echo(f"Downloaded to {path}")


if __name__ == "__main__":
    app()
