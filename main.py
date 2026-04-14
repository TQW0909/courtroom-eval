# main.py
#
# Entry point for the Courtroom Eval pipeline.
#
# Basic usage:
#   python main.py
#   python main.py --model gpt-4o --cases 20 --max-rounds 4
#   python main.py --model llama3:8b-instruct-q4_K_M --split test --cases 5

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from config import get_model
from agents.prosecutor import Prosecutor
from agents.defender import Defender
from agents.judge import Judge
from agents.jury import Jury
from filters.citation_filter import CitationFilter
from graph.courtroom_graph import build_courtroom_graph, initial_state
from data.jbb_loader import load_jbb
from utils.pretty_print import (
    console,
    print_case_header,
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
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model name to use for all agents (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--jury-model", default=None,
        help="Separate model for jurors. Falls back to --model if not set."
    )
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
        "--verbose", action="store_true",
        help="Print each agent step as it runs (streaming transcript for each case)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build components
# ---------------------------------------------------------------------------
# Models for specific roles should be explored more.

def build_components(model_name: str, jury_model_name: str, max_rounds: int):
    model = get_model(model_name)
    jury_model = get_model(jury_model_name or model_name)

    prosecutor     = Prosecutor(model=model)
    defender       = Defender(model=model)
    judge          = Judge(model=model, max_rounds=max_rounds)
    jury           = Jury(models=[jury_model, jury_model, jury_model])
    citation_filter = CitationFilter()

    return build_courtroom_graph(prosecutor, defender, judge, jury, citation_filter)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    console.print()
    console.print(Rule("[bold white]Courtroom Eval[/bold white]", style="white"))
    console.print(f"  Model:      [cyan]{args.model}[/cyan]")
    console.print(f"  Max rounds: [cyan]{args.max_rounds}[/cyan]")
    console.print(f"  Cases:      [cyan]{args.cases}[/cyan]  (split: {args.split})")
    console.print()

    # Build graph
    graph = build_components(args.model, args.jury_model, args.max_rounds)

    # Load data
    cases = get_cases(args.split, args.cases)

    results = []
    for i, case in enumerate(cases):
        console.print(f"[dim]Case {i + 1}/{len(cases)}[/dim]")

        state = initial_state(
            case["response"],
            max_rounds=args.max_rounds,
            case_prompt=case.get("prompt", ""),
        )

        try:
            if args.verbose:
                print_case_header({"case": case["response"]})
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
        except Exception as e:
            console.print(f"  [bold red]ERROR:[/bold red] {e}")
            results.append({
                "predicted":    "benign",  # conservative fallback if trial did not complete
                "ground_truth": case["label"],
                "error":        str(e),
            })
            continue

        results.append({
            "predicted":    predicted,
            "ground_truth": case["label"],
            "confidence":   final_state.get("verdict_confidence"),
            "rounds":       final_state.get("round"),
            "grounding_failures": final_state.get("grounding_failures", 0),
        })

    # Summary
    metrics = compute_metrics(results)
    print_metrics(metrics, args.model)


if __name__ == "__main__":
    main()
