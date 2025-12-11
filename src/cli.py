"""
Command-line interface for the Oncology Data Pipeline.

This module provides CLI commands for data generation, validation,
profiling, and pipeline execution.

Usage:
    python -m src.cli generate --patients 1000 --output data/
    python -m src.cli validate --suite patients
    python -m src.cli profile --input data/patients.csv
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="oncology-dq",
    help="Oncology Data Pipeline - Data Quality CLI",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    patients: int = typer.Option(1000, "--patients", "-p", help="Number of patients to generate"),
    output: Path = typer.Option(Path("data/synthetic"), "--output", "-o", help="Output directory"),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Random seed for reproducibility"
    ),
    format: str = typer.Option("csv", "--format", "-f", help="Output format (csv, parquet)"),
):
    """Generate synthetic oncology data."""
    from src.synthetic_data import OncologyDataFactory

    console.print(f"[bold blue]Generating synthetic data for {patients} patients...[/bold blue]")

    factory = OncologyDataFactory(num_patients=patients, seed=seed)
    dataset = factory.generate()

    if format == "parquet":
        factory.export_to_parquet(dataset, output)
    else:
        factory.export_to_csv(dataset, output)

    console.print(f"[green]Generated data saved to {output}[/green]")

    table = Table(title="Generation Summary")
    table.add_column("Entity", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Patients", str(dataset.patient_count))
    table.add_row("Treatments", str(dataset.treatment_count))
    table.add_row("Lab Results", str(dataset.lab_result_count))
    console.print(table)


@app.command()
def validate(
    suite: str = typer.Option(
        "all", "--suite", "-s", help="Suite to run (patients, treatments, lab_results, all)"
    ),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input CSV file to validate"
    ),
):
    """Run data quality validations."""
    from src.data_quality import ValidationRunner

    console.print(f"[bold blue]Running validation suite: {suite}[/bold blue]")

    runner = ValidationRunner()

    if input_path and input_path.exists():
        import pandas as pd

        df = pd.read_csv(input_path)

        suite_name = f"oncology_{suite}_suite" if suite != "all" else "oncology_patients_suite"
        result = runner.validate_dataframe(df, suite_name)

        status = "[green]PASSED[/green]" if result.success else "[red]FAILED[/red]"
        console.print(f"\nValidation {status}")
        console.print(f"Success Rate: {result.success_percent:.1f}%")

        if result.failed_expectations:
            console.print("\n[yellow]Failed Expectations:[/yellow]")
            for exp in result.failed_expectations[:5]:
                console.print(f"  - {exp.get('expectation_type', 'unknown')}")
    else:
        console.print("[yellow]No input file provided. Run with --input <file.csv>[/yellow]")


@app.command()
def profile(
    input_path: Path = typer.Argument(..., help="CSV file to profile"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML report path"),
):
    """Generate data profile report."""
    import pandas as pd

    from src.profiling import DataProfiler

    console.print(f"[bold blue]Profiling {input_path}...[/bold blue]")

    df = pd.read_csv(input_path)
    profiler = DataProfiler()
    profile = profiler.profile(df, input_path.stem)

    table = Table(title=f"Profile: {input_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Rows", f"{profile.row_count:,}")
    table.add_row("Columns", str(profile.column_count))
    table.add_row("Memory (MB)", f"{profile.memory_usage / 1024 / 1024:.2f}")
    console.print(table)

    if output:
        profiler.generate_report(profile, output)
        console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def scorecard(
    input_path: Path = typer.Argument(..., help="CSV file to score"),
):
    """Calculate data quality scorecard."""
    import pandas as pd

    from src.metrics import QualityScorecardCalculator

    console.print(f"[bold blue]Calculating scorecard for {input_path}...[/bold blue]")

    df = pd.read_csv(input_path)
    calculator = QualityScorecardCalculator()
    scorecard = calculator.calculate(df, input_path.stem)

    console.print(scorecard.summary())


@app.command()
def test_connection(
    backend: str = typer.Option("databricks", "--backend", "-b", help="Backend to test"),
):
    """Test database connection."""
    from src.connectors import ConnectionFactory

    console.print(f"[bold blue]Testing {backend} connection...[/bold blue]")

    try:
        success = ConnectionFactory.test_connection(backend)
        if success:
            console.print(f"[green]Connection to {backend} successful![/green]")
        else:
            console.print(f"[red]Connection to {backend} failed[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent.parent / "dashboards" / "quality_dashboard.py"

    console.print("[bold blue]Launching Quality Dashboard...[/bold blue]")
    console.print("Open http://localhost:8501 in your browser")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


if __name__ == "__main__":
    app()
