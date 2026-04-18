# main.py
#
# Entry point for the Courtroom Eval pipeline.
#
# Basic usage:
#   python main.py
#   python main.py --model gpt-4o --cases 20 --max-rounds 4
#   python main.py --model llama3:8b-instruct-q4_K_M --split test --cases 5
#
# Ablation examples:
#   python main.py --model llama3:8b-instruct-q4_K_M --no-filter --cases 10
#   python main.py --model gpt-4o-mini --no-defense --cases 10
#   python main.py --model gpt-4o --jurors 5 --cases 10
#
# Logging:
#   python main.py --model gpt-4o-mini --log results/runs.jsonl --cases 20

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from config import get_model
from tasks import TASKS, DEFAULT_TASK, TaskConfig
from agents.prosecutor import Prosecutor
from agents.defender import Defender
from agents.judge import Judge
from agents.jury import Jury
from filters.citation_filter import CitationFilter, NoOpFilter
from graph.courtroom_graph import build_courtroom_graph, initial_state
from baselines.mirror import MirrorBaseline
from data.jbb_loader import load_jbb
from utils.token_tracker import TokenTracker
from utils.run_logger import RunLogger
from utils.pretty_print import (
    console,
    print_case_header,
    print_full_result,
    print_live_epilogue,
    print_stream_update,
)
from rich.rule import Rule
from rich.table import Table
from rich import box


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Courtroom Eval pipeline on JBB.")

    # Model selection
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model name to use for all agents (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--jury-model", default=None,
        help="Separate model for jurors. Falls back to --model if not set."
    )

    # Task selection
    parser.add_argument(
        "--task", default=DEFAULT_TASK, choices=list(TASKS.keys()),
        help=f"Evaluation task (default: {DEFAULT_TASK}). Available: {', '.join(TASKS.keys())}"
    )

    # Pipeline parameters
    parser.add_argument(
        "--max-rounds", type=int, default=4,
        help="Maximum debate rounds before judge forces a close (default: 4)"
    )
    parser.add_argument(
        "--cases", type=int, default=10,
        help="Number of cases to evaluate (default: 10)"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="JBB dataset split to use (default: test)"
    )
    parser.add_argument(
        "--jurors", type=int, default=3,
        help="Number of jurors in the jury panel (default: 3)"
    )

    # Ablation flags
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Ablation: disable citation filter (all arguments pass through)"
    )
    parser.add_argument(
        "--no-defense", action="store_true",
        help="Ablation: disable defense agent (prosecution-only evaluation)"
    )

    # Output
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each agent step as it runs (streaming transcript)"
    )
    parser.add_argument(
        "--log", default=None, metavar="PATH",
        help="Path to JSONL file for structured run logging (e.g. results/runs.jsonl)"
    )
    parser.add_argument(
        "--baseline", default="courtroom", choices=["courtroom", "mirror"],
        help="Evaluation method: 'courtroom' (default) or 'mirror' (MIRROR baseline)"
    )
    parser.add_argument(
        "--mirror-iterations", type=int, default=5,
        help="Max iterations for the MIRROR baseline (default: 5)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build components
# ---------------------------------------------------------------------------

def build_components(args, tracker: TokenTracker, task: TaskConfig):
    """
    Build the LangGraph pipeline from CLI args.

    The TokenTracker callback is attached to every model so that token usage
    and latency are captured automatically on every LLM call.
    The TaskConfig is passed to every agent so prompts adapt to the task.
    """
    callbacks = [tracker]

    model      = get_model(args.model, callbacks=callbacks)
    jury_model = get_model(args.jury_model or args.model, callbacks=callbacks)

    prosecutor = Prosecutor(model=model, task=task)
    judge      = Judge(model=model, max_rounds=args.max_rounds)
    jury       = Jury(models=[jury_model] * args.jurors, task=task)

    # Ablation: defense
    if args.no_defense:
        defender = _StubDefender()
    else:
        defender = Defender(model=model, task=task)

    # Ablation: citation filter
    if args.no_filter:
        citation_filter = NoOpFilter()
    else:
        citation_filter = CitationFilter()

    return build_courtroom_graph(prosecutor, defender, judge, jury, citation_filter,
                                 token_tracker=tracker)


class _StubDefender:
    """Ablation stub — always forfeits. Produces a placeholder so the graph continues."""

    def argue(self, state: dict) -> dict:
        return {
            **state,
            "defense_args": state["defense_args"] + ["[no defense — ablation]"],
        }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_cases(split: str, n: int) -> list[dict]:
    """
    Returns a list of dicts with keys:
      - 'response' (str): the AI-generated response (Target field from JBB behaviors)
      - 'prompt'   (str): the original user goal / jailbreak prompt (Goal field)
      - 'label'    (str): ground-truth label — 'harmful' or 'benign'

    Uses the JBB behaviors split which has harmful and benign cases.
    """
    behaviors, _ = load_jbb()
    harmful_cases = behaviors["harmful"]
    benign_cases  = behaviors["benign"]

    half = n // 2
    cases = []

    for row in harmful_cases.select(range(min(half, len(harmful_cases)))):
        cases.append({
            "response": row["Target"],
            "prompt":   row.get("Goal", ""),
            "label":    "harmful",
        })

    for row in benign_cases.select(range(min(half, len(benign_cases)))):
        cases.append({
            "response": row["Target"],
            "prompt":   row.get("Goal", ""),
            "label":    "benign",
        })

    return cases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["predicted"] == r["ground_truth"])
    tp = sum(1 for r in results if r["predicted"] == "harmful" and r["ground_truth"] == "harmful")
    fp = sum(1 for r in results if r["predicted"] == "harmful" and r["ground_truth"] == "benign")
    fn = sum(1 for r in results if r["predicted"] == "benign"    and r["ground_truth"] == "harmful")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total":     total,
        "correct":   correct,
        "accuracy":  correct / total,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def print_metrics(metrics: dict, model_name: str):
    console.print()
    console.print(Rule("[bold white]Evaluation Results[/bold white]", style="white"))

    table = Table(box=box.ROUNDED, border_style="dim", show_header=True, header_style="bold white")
    table.add_column("Metric",    style="dim",  width=14)
    table.add_column("Value",     width=10)

    table.add_row("Model",     model_name)
    table.add_row("Cases",     str(metrics["total"]))
    table.add_row("Correct",   str(metrics["correct"]))
    table.add_row("Accuracy",  f'{metrics["accuracy"]:.1%}')
    table.add_row("Precision", f'{metrics["precision"]:.1%}')
    table.add_row("Recall",    f'{metrics["recall"]:.1%}')
    table.add_row("F1",        f'{metrics["f1"]:.3f}')

    console.print(table)
    console.print()


def print_token_summary(totals: dict):
    """Print a token-usage summary table after the run."""
    console.print(Rule("[bold white]Resource Usage[/bold white]", style="white"))

    table = Table(box=box.ROUNDED, border_style="dim", show_header=True, header_style="bold white")
    table.add_column("Role",           style="dim", width=14)
    table.add_column("Calls",          width=7,  justify="right")
    table.add_column("Input Tok",      width=11, justify="right")
    table.add_column("Output Tok",     width=11, justify="right")
    table.add_column("Total Tok",      width=11, justify="right")
    table.add_column("Latency (s)",    width=12, justify="right")

    for role, stats in totals.get("by_role", {}).items():
        table.add_row(
            role,
            str(stats["calls"]),
            f'{stats["input_tokens"]:,}',
            f'{stats["output_tokens"]:,}',
            f'{stats["total_tokens"]:,}',
            f'{stats["latency_ms"] / 1000:.1f}',
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        str(totals.get("total_calls", 0)),
        f'{totals.get("total_input_tokens", 0):,}',
        f'{totals.get("total_output_tokens", 0):,}',
        f'{totals.get("total_tokens", 0):,}',
        f'{totals.get("total_latency_ms", 0) / 1000:.1f}',
    )

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_courtroom(args, cases: list[dict]) -> list[dict]:
    graph = build_components(args.model, args.jury_model, args.max_rounds)

    args = parse_args()

    # --- Resolve task ---
    task = TASKS[args.task]

    # --- Header ---
    console.print()
    console.print(Rule("[bold white]Courtroom Eval[/bold white]", style="white"))
    console.print(f"  Task:       [cyan]{task.name}[/cyan] — {task.description}")
    console.print(f"  Labels:     [cyan]{task.labels[0]}[/cyan] / [cyan]{task.labels[1]}[/cyan]")
    console.print(f"  Model:      [cyan]{args.model}[/cyan]")
    console.print(f"  Max rounds: [cyan]{args.max_rounds}[/cyan]")
    console.print(f"  Cases:      [cyan]{args.cases}[/cyan]  (split: {args.split})")
    console.print(f"  Jurors:     [cyan]{args.jurors}[/cyan]")

    ablation_flags = []
    if args.no_filter:
        ablation_flags.append("no-filter")
    if args.no_defense:
        ablation_flags.append("no-defense")
    if ablation_flags:
        console.print(f"  Ablation:   [yellow]{', '.join(ablation_flags)}[/yellow]")
    if args.log:
        console.print(f"  Logging to: [dim]{args.log}[/dim]")
    console.print()

    # --- Token tracker ---
    tracker = TokenTracker()

    # --- Build graph ---
    graph = build_components(args, tracker, task)

    # --- Load data ---
    cases = get_cases(args.split, args.cases)

    # --- Run logger (optional) ---
    run_config = {
        "task": task.name,
        "labels": list(task.labels),
        "model": args.model,
        "jury_model": args.jury_model or args.model,
        "max_rounds": args.max_rounds,
        "jurors": args.jurors,
        "cases_requested": args.cases,
        "split": args.split,
        "no_filter": args.no_filter,
        "no_defense": args.no_defense,
    }
    logger = RunLogger(args.log, config=run_config) if args.log else None

    # --- Evaluate ---
    results = []

    for i, case in enumerate(cases):
        console.print(f"[dim]Case {i + 1}/{len(cases)}[/dim]")

        tracker.reset()  # fresh token counts per case

        state = initial_state(
            case["response"],
            max_rounds=args.max_rounds,
            case_prompt=case.get("prompt", ""),
        )

        try:
            if args.verbose:
                print_case_header(state)
                final_state = None
                for chunk in graph.stream(state, stream_mode="updates"):
                    for node_name, node_state in chunk.items():
                        final_state = node_state
                        print_stream_update(node_name, node_state)
                if final_state is None:
                    raise RuntimeError("Graph stream produced no updates")
                predicted = final_state["final_verdict"]
                print_live_epilogue(final_state, ground_truth=case["label"])
            else:
                final_state = graph.invoke(state)
                predicted = final_state["final_verdict"]

                verdict_color = "bold red" if predicted and predicted.lower() == "harmful" else "bold green"
                correct = "✓" if predicted and predicted.lower() == case["label"] else "✗"
                console.print(
                    f"  [{verdict_color}]{predicted or 'NO VERDICT':>8}[/{verdict_color}]  "
                    f"gt={case['label']:<8} {correct}  "
                    f"rounds={final_state.get('round', 0)}  "
                    f"filter_fails={final_state.get('grounding_failures', 0)}"
                )
        except Exception as e:
            console.print(f"  [bold red]ERROR:[/bold red] {e}")
            results.append({
                "predicted":    "benign",
                "ground_truth": case["label"],
                "error":        str(e),
                "token_usage":  tracker.summary(),
            }
            results.append(result)
            if logger:
                logger.add_case(result)
            continue

        token_usage = tracker.summary()

        result = {
            "predicted":          predicted,
            "ground_truth":       case["label"],
            "confidence":         final_state.get("verdict_confidence"),
            "rounds":             final_state.get("round"),
            "grounding_failures": final_state.get("grounding_failures", 0),
            "token_usage":        token_usage,
        }
        results.append(result)
        if logger:
            logger.add_case(result)

    return results


def run_mirror(args, cases: list[dict]) -> list[dict]:
    model1 = get_model(args.model)
    model2 = get_model(args.jury_model or args.model)
    mirror = MirrorBaseline(
        model1=model1,
        model2=model2,
        max_iterations=args.mirror_iterations,
        verbose=args.verbose,
    )
    results = []

    for i, case in enumerate(cases):
        console.print(f"[dim]Case {i + 1}/{len(cases)}[/dim]")

        try:
            outcome = mirror.classify(
                prompt=case.get("prompt", ""),
                response=case["response"],
            )
            predicted = outcome["verdict"]

            if args.verbose:
                converged_str = "yes" if outcome["converged"] else "no (tie-break)"
                console.print(
                    f"  Verdict: [bold]{predicted}[/bold]  "
                    f"conf={outcome['confidence']:.2f}  "
                    f"iters={outcome['iterations']}  converged={converged_str}"
                )
        except Exception as e:
            console.print(f"  [bold red]ERROR:[/bold red] {e}")
            results.append({
                "predicted":    "benign",
                "ground_truth": case["label"],
                "error":        str(e),
            })
            continue

        results.append({
            "predicted":    predicted,
            "ground_truth": case["label"],
            "confidence":   outcome["confidence"],
            "iterations":   outcome["iterations"],
            "converged":    outcome["converged"],
        })

    return results


def main():
    args = parse_args()

    baseline_label = f"MIRROR (max {args.mirror_iterations} iters)" \
        if args.baseline == "mirror" else "Courtroom"

    console.print()
    console.print(Rule(f"[bold white]Courtroom Eval — {baseline_label}[/bold white]", style="white"))
    console.print(f"  Model:      [cyan]{args.model}[/cyan]")
    if args.baseline == "courtroom":
        console.print(f"  Max rounds: [cyan]{args.max_rounds}[/cyan]")
    else:
        console.print(f"  Max iters:  [cyan]{args.mirror_iterations}[/cyan]")
    console.print(f"  Cases:      [cyan]{args.cases}[/cyan]  (split: {args.split})")
    console.print()

    cases = get_cases(args.split, args.cases)

    if args.baseline == "mirror":
        results = run_mirror(args, cases)
    else:
        results = run_courtroom(args, cases)

    metrics = compute_metrics(results)
    print_metrics(metrics, args.model)

    # --- Token summary ---
    # Aggregate token usage across all cases
    all_totals = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "total_calls": 0,
        "by_role": {},
    }
    for r in results:
        usage = r.get("token_usage", {})
        all_totals["total_input_tokens"] += usage.get("total_input_tokens", 0)
        all_totals["total_output_tokens"] += usage.get("total_output_tokens", 0)
        all_totals["total_tokens"] += usage.get("total_tokens", 0)
        all_totals["total_latency_ms"] += usage.get("total_latency_ms", 0)
        all_totals["total_calls"] += usage.get("calls", 0)
        for role, stats in usage.get("by_role", {}).items():
            if role not in all_totals["by_role"]:
                all_totals["by_role"][role] = {
                    "input_tokens": 0, "output_tokens": 0,
                    "total_tokens": 0, "latency_ms": 0.0, "calls": 0,
                }
            for k in ("input_tokens", "output_tokens", "total_tokens", "calls"):
                all_totals["by_role"][role][k] += stats.get(k, 0)
            all_totals["by_role"][role]["latency_ms"] += stats.get("latency_ms", 0)

    print_token_summary(all_totals)

    # --- Write log ---
    if logger:
        record = logger.finalize(metrics)
        console.print(f"  [dim]Run logged to {args.log} (run_id: {record['run_id']})[/dim]")
        console.print()


if __name__ == "__main__":
    main()
