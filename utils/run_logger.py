# utils/run_logger.py
#
# Structured JSONL logger for courtroom-eval runs.
# Each invocation of main.py produces one run record containing run-level
# config, per-case results, and aggregate metrics. Records are appended to
# a single JSONL file so multiple runs can be compared.

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RunLogger:
    """
    Collects per-case results during a run and writes a single JSONL record
    at the end.

    Usage:
        logger = RunLogger("results/runs.jsonl", config={...})
        for case in cases:
            logger.add_case({...})
        logger.finalize(metrics={...})
    """

    def __init__(self, path: str, config: dict):
        """
        Parameters
        ----------
        path : str
            Path to the .jsonl output file. Created if missing; appended otherwise.
        config : dict
            Run-level configuration (model, max_rounds, ablation flags, etc.)
        """
        self.path = path
        self.config = config
        self.cases: List[dict] = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add_case(self, record: dict):
        """Append a per-case result dict."""
        self.cases.append(record)

    def finalize(self, metrics: dict):
        """
        Write the complete run record to the JSONL file.

        The record structure:
        {
            "run_id": "20260415T183000Z",
            "timestamp": "2026-04-15T18:30:00+00:00",
            "config": { model, max_rounds, cases, ablation flags, ... },
            "cases": [ per-case records ],
            "metrics": { accuracy, precision, recall, f1, ... },
            "totals": { aggregated token usage + latency }
        }
        """
        # Aggregate token/latency totals across cases
        totals = self._aggregate_totals()

        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "cases": self.cases,
            "metrics": metrics,
            "totals": totals,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        return record

    def _aggregate_totals(self) -> dict:
        """Sum token usage and latency across all cases."""
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_latency = 0.0
        total_calls = 0
        by_role: Dict[str, dict] = {}

        for case in self.cases:
            usage = case.get("token_usage", {})
            total_input += usage.get("total_input_tokens", 0)
            total_output += usage.get("total_output_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            total_latency += usage.get("total_latency_ms", 0)
            total_calls += usage.get("calls", 0)

            for role, stats in usage.get("by_role", {}).items():
                if role not in by_role:
                    by_role[role] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency_ms": 0.0,
                        "calls": 0,
                    }
                for k in ("input_tokens", "output_tokens", "total_tokens", "calls"):
                    by_role[role][k] += stats.get(k, 0)
                by_role[role]["latency_ms"] += stats.get("latency_ms", 0)

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_latency, 1),
            "total_calls": total_calls,
            "by_role": by_role,
        }
