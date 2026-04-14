# utils/pretty_print.py

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

console = Console()


def _verdict_color(verdict: str) -> str:
    return "bold red" if verdict == "HARMFUL" else "bold green"


def _decision_color(decision: str) -> str:
    return "yellow" if decision == "CONTINUE" else "cyan"


def print_case_header(state: dict):
    console.print()
    console.print(Rule("[bold white]COURTROOM EVALUATION[/bold white]", style="white"))
    console.print(Panel(
        state["case"],
        title="[bold]Response Under Review[/bold]",
        border_style="white",
        padding=(1, 2)
    ))


def print_round(state: dict, round_idx: int):
    console.print()
    console.print(Rule(f"[bold white]Round {round_idx + 1}[/bold white]", style="dim"))

    # Prosecution argument
    prosecution_arg = state["prosecution_args"][round_idx] if round_idx < len(state["prosecution_args"]) else None
    if prosecution_arg:
        console.print(Panel(
            prosecution_arg,
            title="[bold red]⚖ Prosecution[/bold red]",
            border_style="red",
            padding=(1, 2)
        ))

    # Defense argument
    defense_arg = state["defense_args"][round_idx] if round_idx < len(state["defense_args"]) else None
    if defense_arg:
        console.print(Panel(
            defense_arg,
            title="[bold blue]🛡 Defense[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))

    # Judge decision for this round
    judge_decision = state["judge_decisions"][round_idx] if round_idx < len(state["judge_decisions"]) else None
    if judge_decision:
        color = _decision_color(judge_decision)
        console.print(f"  [dim]Judge:[/dim] [{color}]{judge_decision}[/{color}]")


def print_jury_verdict(state: dict):
    console.print()
    console.print(Rule("[bold white]JURY DELIBERATION[/bold white]", style="white"))

    # Individual juror votes
    table = Table(
        box=box.ROUNDED,
        border_style="dim",
        show_header=True,
        header_style="bold white"
    )
    table.add_column("Juror", style="dim", width=8)
    table.add_column("Verdict", width=10)
    table.add_column("Confidence", width=12)
    table.add_column("Decisive Argument")

    for i, vote in enumerate(state["jury_votes"]):
        verdict = vote["verdict"]
        color = _verdict_color(verdict)
        table.add_row(
            f"#{i + 1}",
            Text(verdict, style=color),
            f"{vote['confidence']:.0%}",
            vote["reason"]
        )

    console.print(table)

    # Final verdict banner
    console.print()
    final = state["final_verdict"]
    confidence = state["verdict_confidence"]
    color = _verdict_color(final)

    console.print(Panel(
        f"[{color}]{final}[/{color}]  [{color}]({confidence:.0%} avg confidence)[/{color}]",
        title="[bold white]Final Verdict[/bold white]",
        border_style=color.replace("bold ", ""),
        padding=(1, 4)
    ))


def print_summary(state: dict):
    console.print()
    console.print(Rule("[bold white]Summary[/bold white]", style="dim"))

    harmful_votes = sum(1 for v in state["jury_votes"] if v["verdict"] == "HARMFUL")

    console.print(f"  [dim]Rounds taken:[/dim]        {state['round']}")
    console.print(f"  [dim]Grounding failures:[/dim]  {state['grounding_failures']}")
    console.print(f"  [dim]Jury split:[/dim]          {harmful_votes}-{3 - harmful_votes} "
                  f"({'HARMFUL' if harmful_votes >= 2 else 'SAFE'})")
    console.print(f"  [dim]Judge decisions:[/dim]     {' → '.join(state['judge_decisions'])}")
    console.print()


def print_full_result(state: dict, ground_truth: str | None = None):
    """Print the complete courtroom transcript for one case."""
    print_case_header(state)

    rounds = max(len(state["prosecution_args"]), len(state["defense_args"]))
    for i in range(rounds):
        print_round(state, i)

    print_jury_verdict(state)
    print_summary(state)

    # Ground truth comparison if available
    if ground_truth is not None:
        predicted = state["final_verdict"]
        match = predicted == ground_truth
        status = "[bold green]✓ CORRECT[/bold green]" if match else "[bold red]✗ WRONG[/bold red]"
        console.print(f"  Ground truth: [bold]{ground_truth}[/bold]  →  {status}")
        console.print()

    console.print(Rule(style="dim"))