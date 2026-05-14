"""
run.py — CLI runner for Architext

Usage:
    python run.py "Build a real-time chat app for 5000 users"
    python run.py --input examples/chat_app.txt
    python run.py --model llama3.1 "Migrate on-premise CRM to AWS"

Useful for:
- Testing the agent without spinning up the Streamlit UI
- Batch processing multiple requirement documents
- Debugging individual nodes
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

load_dotenv()
console = Console()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library loggers
    for lib in ["httpx", "chromadb", "urllib3", "langchain"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def print_report(state: dict):
    """Pretty-print the architecture report to the terminal."""
    report = state.get("report")

    if not report:
        if state.get("needs_clarification"):
            console.print("\n[yellow]⚠️  Need more information:[/yellow]")
            for q in state.get("clarifications_needed", []):
                console.print(f"  • {q}")
        elif state.get("error"):
            console.print(f"\n[red]❌ Error: {state['error']}[/red]")
        return

    recommended = report.get("recommended", {})

    # Header
    console.rule("[bold blue]🏗️  ARCHITEXT ANALYSIS REPORT[/bold blue]")

    # Summary
    console.print(Panel(
        report.get("summary", ""),
        title="Executive Summary",
        border_style="blue",
    ))

    # Recommended approach
    console.print(f"\n[bold green]⭐ Recommended: {recommended.get('name')}[/bold green]")
    console.print(f"[dim]{recommended.get('summary', '')}[/dim]\n")

    # Components table
    comp_table = Table(title="Components", show_header=True, header_style="bold cyan")
    comp_table.add_column("Technology", style="green")
    comp_table.add_column("Type")
    comp_table.add_column("Purpose")
    comp_table.add_column("Est. Cost/mo", justify="right")

    for comp in recommended.get("components", []):
        comp_table.add_row(
            comp.get("technology", ""),
            comp.get("type", ""),
            comp.get("purpose", "")[:50],
            f"${comp.get('estimated_monthly_cost_usd', 0):.0f}",
        )
    console.print(comp_table)

    # Score table
    score_table = Table(title="\nScores", show_header=True, header_style="bold cyan")
    score_table.add_column("Metric")
    score_table.add_column("Score", justify="center")
    score_table.add_column("Visual")

    for metric, score in recommended.get("scores", {}).items():
        bar = "█" * score + "░" * (10 - score)
        color = "green" if score >= 7 else ("yellow" if score >= 5 else "red")
        score_table.add_row(
            metric.replace("_", " ").title(),
            f"{score}/10",
            f"[{color}]{bar}[/{color}]",
        )
    console.print(score_table)

    console.print(f"\n[bold]Total estimated cost: [green]${recommended.get('total_monthly_cost_usd', 0):.0f}/month[/green][/bold]")

    # Mermaid diagram
    diagram = recommended.get("mermaid_diagram", "")
    if diagram:
        console.print("\n[bold cyan]Architecture Diagram (Mermaid):[/bold cyan]")
        console.print("[dim](Paste at https://mermaid.live to render)[/dim]")
        console.print(Panel(diagram, border_style="dim"))

    # Comparison table
    comparison = report.get("comparison_table", "")
    if comparison:
        console.print("\n[bold]Comparison Matrix:[/bold]")
        console.print(Markdown(comparison))

    # Risks
    risks = report.get("risk_assessment", [])
    if risks:
        console.print("\n[bold yellow]⚠️  Risks:[/bold yellow]")
        for risk in risks:
            severity_color = {"high": "red", "medium": "yellow", "low": "green"}.get(risk.get("severity", ""), "white")
            console.print(f"  [{severity_color}][{risk.get('severity', '').upper()}][/{severity_color}] "
                         f"{risk.get('area', '').title()}: {risk.get('description', '')}")
            console.print(f"    → {risk.get('mitigation', '')}")

    # Next steps
    next_steps = report.get("next_steps", [])
    if next_steps:
        console.print("\n[bold]🚀 Next Steps:[/bold]")
        for i, step in enumerate(next_steps, 1):
            console.print(f"  {i}. {step}")

    console.rule()


def main():
    parser = argparse.ArgumentParser(
        description="Architext — AI Solutions Architecture Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py "Build a chat app for 5000 users, AWS, $500/mo budget"
  python run.py --input requirements.txt
  python run.py --model llama3.1 --json-output result.json "Healthcare patient portal"
        """,
    )
    parser.add_argument("prompt", nargs="?", help="Project description")
    parser.add_argument("--input", "-i", help="Path to requirements text file")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--json-output", help="Save full state as JSON to this path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--ingest", action="store_true", help="(Re-)ingest knowledge base and exit")
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip clarification questions and proceed with available info (recommended for most inputs)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Knowledge base ingestion
    if args.ingest:
        console.print("[cyan]Ingesting knowledge base...[/cyan]")
        from rag.ingest import ingest_knowledge_base
        count = ingest_knowledge_base(force_reload=True)
        console.print(f"[green]✅ Ingested {count} document chunks[/green]")
        return

    # Get input
    user_input = ""
    if args.input:
        user_input = Path(args.input).read_text(encoding="utf-8")
    elif args.prompt:
        user_input = args.prompt
    else:
        console.print("[yellow]Enter your project description (Ctrl+D when done):[/yellow]")
        try:
            user_input = sys.stdin.read()
        except KeyboardInterrupt:
            return

    if not user_input.strip():
        console.print("[red]No input provided.[/red]")
        sys.exit(1)

    console.print(f"\n[cyan]🚀 Running Architext with {args.model}...[/cyan]")
    console.print("[dim]This takes 60-90 seconds on a local Llama model.[/dim]\n")

    from agent.graph import run_agent

    state = run_agent(
        user_input=user_input,
        ollama_base_url=args.ollama_url,
        model_name=args.model,
        force_proceed=args.no_clarify,
    )

    # If clarifications needed and not force-proceeding, prompt interactively
    if state.get("needs_clarification") and not state.get("report"):
        questions = state.get("clarifications_needed", [])
        console.print("\n[yellow]⚠️  A few quick questions (press Enter to skip any):[/yellow]")
        clarifications = {}
        for q in questions:
            answer = console.input(f"  [cyan]{q}[/cyan]\n  > ").strip()
            if answer:
                clarifications[q] = answer

        if clarifications:
            console.print("\n[cyan]▶ Continuing with your answers...[/cyan]\n")
            state = run_agent(
                user_input=user_input,
                clarifications=clarifications,
                force_proceed=True,   # don't ask again
                ollama_base_url=args.ollama_url,
                model_name=args.model,
            )
        else:
            # User skipped all questions — just force proceed
            console.print("\n[dim]Proceeding with available information...[/dim]\n")
            state = run_agent(
                user_input=user_input,
                force_proceed=True,
                ollama_base_url=args.ollama_url,
                model_name=args.model,
            )

    print_report(state)

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(state, indent=2, default=str))
        console.print(f"\n[dim]Full state saved to {args.json_output}[/dim]")


if __name__ == "__main__":
    main()