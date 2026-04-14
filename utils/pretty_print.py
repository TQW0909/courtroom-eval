# utils/pretty_print.py

from typing import Optional

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

_MISSING_ARG = (
    "[dim italic](No grounded argument on record — likely dropped by the citation filter "
    "or not generated.)[/dim italic]"
)


def _verdict_color(verdict: str) -> str:
    return "bold red" if verdict and verdict.lower() == "harmful" else "bold green"


def _decision_color(decision: str) -> str:
    return "yellow" if decision == "CONTINUE" else "cyan"


def _display_round_count(state: dict) -> int:
    """
    Rows of debate to print. ``prosecution_args`` / ``defense_args`` can be empty after
    citation drops while ``round`` and ``judge_decisions`` still reflect the trial.
    """
    p = state.get("prosecution_args") or []
    d = state.get("defense_args") or []
    jd = state.get("judge_decisions") or []
    r = int(state.get("round") or 0)
    return max(len(p), len(d), len(jd), r)


def print_stream_update(node_name: str, state: dict) -> None:
    """
    Print one step of the trial as LangGraph finishes each node (--verbose stream mode).
    ``state`` is the full graph state after that node.
    """
    console.print()

    if node_name == "prosecute":
        console.print(Rule("[bold red]Prosecutor[/bold red]", style="red"))
        args_ = state.get("prosecution_args") or []
        if args_:
            content = escape(args_[-1])
        else:
            content = "[dim italic](no argument produced)[/dim italic]"
        console.print(Panel(
            content,
            border_style="red",
            padding=(1, 2),
        ))

    elif node_name == "filter_prosecution":
        passed = state.get("last_filter_passed", True)
        status = "[bold green]passed[/bold green]" if passed else "[bold red]failed[/bold red] (argument may be dropped or retried)"
        console.print(Rule("[dim]Citation filter — prosecution[/dim]", style="dim"))
        console.print(f"  {status}  [dim](grounding failures so far: {state.get('grounding_failures', 0)})[/dim]")

    elif node_name == "defend":
        console.print(Rule("[bold blue]Defender[/bold blue]", style="blue"))
        args_ = state.get("defense_args") or []
        if args_:
            content = escape(args_[-1])
        else:
            content = "[dim italic](no argument produced)[/dim italic]"
        console.print(Panel(
            content,
            border_style="blue",
            padding=(1, 2),
        ))

    elif node_name == "filter_defense":
        passed = state.get("last_filter_passed", True)
        status = "[bold green]passed[/bold green]" if passed else "[bold red]failed[/bold red] (argument may be dropped or retried)"
        console.print(Rule("[dim]Citation filter — defense[/dim]", style="dim"))
        console.print(f"  {status}  [dim](grounding failures so far: {state.get('grounding_failures', 0)})[/dim]")

    elif node_name == "judge":
        console.print(Rule("[bold yellow]Judge[/bold yellow]", style="yellow"))
        jd = state.get("judge_decisions") or []
        jr = state.get("judge_rationales") or []
        if jd:
            d, r = jd[-1], jr[-1] if len(jr) >= len(jd) else ""
            color = _decision_color(d)
            line = f"[{color}]{d}[/{color}]"
            if r:
                line += f"  [dim]— {escape(r)}[/dim]"
            console.print(f"  {line}")
        else:
            console.print("  [dim](no decision recorded)[/dim]")

    elif node_name == "jury":
        print_jury_verdict(state)

    else:
        console.print(Rule(f"[dim]{escape(node_name)}[/dim]", style="dim"))


def print_live_epilogue(state: dict, ground_truth: Optional[str] = None) -> None:
    """Short footer after streaming: summary stats + optional label check."""
    print_summary(state)
    if ground_truth is not None:
        predicted = state.get("final_verdict")
        match = predicted == ground_truth
        status = "[bold green]✓ CORRECT[/bold green]" if match else "[bold red]✗ WRONG[/bold red]"
        console.print(f"  Ground truth: [bold]{ground_truth}[/bold]  →  {status}")
        console.print()
    console.print(Rule(style="dim"))


def print_case_header(state: dict):
    console.print()
    console.print(Rule("[bold white]COURTROOM EVALUATION[/bold white]", style="white"))

    case_prompt = state.get("case_prompt", "")
    if case_prompt:
        console.print(Panel(
            escape(case_prompt),
            title="[bold]Original Prompt[/bold]",
            border_style="white",
            padding=(1, 2),
        ))

    console.print(Panel(
        escape(state.get("case", "")),
        title="[bold]Response Under Review[/bold]",
        border_style="white",
        padding=(1, 2),
    ))


def print_round(state: dict, round_idx: int):
    console.print()
    console.print(Rule(f"[bold white]Round {round_idx + 1}[/bold white]", style="dim"))

    p_args = state.get("prosecution_args") or []
    d_args = state.get("defense_args") or []

    prosecution_arg = p_args[round_idx] if round_idx < len(p_args) else None
    console.print(Panel(
        escape(prosecution_arg) if prosecution_arg else _MISSING_ARG,
        title="[bold red]Prosecution[/bold red]",
        border_style="red",
        padding=(1, 2),
    ))

    defense_arg = d_args[round_idx] if round_idx < len(d_args) else None
    console.print(Panel(
        escape(defense_arg) if defense_arg else _MISSING_ARG,
        title="[bold blue]Defense[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    jd = state.get("judge_decisions") or []
    jr = state.get("judge_rationales") or []
    judge_decision = jd[round_idx] if round_idx < len(jd) else None
    judge_rationale = jr[round_idx] if round_idx < len(jr) else None
    if judge_decision:
        color = _decision_color(judge_decision)
        console.print(f"  [dim]Judge:[/dim] [{color}]{judge_decision}[/{color}]", end="")
        if judge_rationale:
            console.print(f"  [dim]— {escape(judge_rationale)}[/dim]")
        else:
            console.print()


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
            f"{float(vote.get('confidence', 0)):.0%}",
            escape(str(vote.get("reason", ""))),
        )

    console.print(table)

    # Final verdict banner
    console.print()
    final = state.get("final_verdict") or ""
    confidence = float(state.get("verdict_confidence") or 0.0)
    color = _verdict_color(final)

    console.print(Panel(
        f"[{color}]{escape(str(final))}[/{color}]  [{color}]({confidence:.0%} avg confidence)[/{color}]",
        title="[bold white]Final Verdict[/bold white]",
        border_style=color.replace("bold ", ""),
        padding=(1, 4)
    ))


def print_summary(state: dict):
    console.print()
    console.print(Rule("[bold white]Summary[/bold white]", style="dim"))

    votes = state.get("jury_votes") or []
    harmful_votes = sum(1 for v in votes if str(v.get("verdict", "")).lower() == "harmful")

    console.print(f"  [dim]Rounds taken:[/dim]        {state['round']}")
    console.print(f"  [dim]Grounding failures:[/dim]  {state['grounding_failures']}")
    n = len(votes) or 3
    console.print(f"  [dim]Jury split:[/dim]          {harmful_votes}-{n - harmful_votes} "
                  f"({'harmful' if harmful_votes >= 2 else 'benign'})")
    decisions = state.get("judge_decisions") or []
    rationales = state.get("judge_rationales") or []
    decision_str = "  →  ".join(
        f"{d} ({escape(r)})" if i < len(rationales) and (r := rationales[i]) else d
        for i, d in enumerate(decisions)
    )
    console.print(f"  [dim]Judge decisions:[/dim]     {decision_str}")
    console.print()


def print_full_result(state: dict, ground_truth: Optional[str] = None):
    """Print the complete courtroom transcript for one case."""
    print_case_header(state)

    rounds = _display_round_count(state)
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