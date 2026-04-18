# utils/token_tracker.py
#
# LangChain callback handler that captures per-call token usage and latency
# for both OpenAI and Ollama models. Accumulates stats by agent role so the
# run logger can write them out per-case.

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class CallRecord:
    """One LLM invocation."""
    role: str               # e.g. "prosecutor", "defender", "judge", "juror"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float       # wall-clock milliseconds
    model: str


class TokenTracker(BaseCallbackHandler):
    """
    Attach as a callback to any LangChain model to capture token usage.

    Usage:
        tracker = TokenTracker()
        model = ChatOllama(model="llama3:8b", callbacks=[tracker])
        # ... run pipeline ...
        summary = tracker.summary()       # aggregate stats
        tracker.reset()                    # clear for next case
    """

    def __init__(self):
        super().__init__()
        self.records: List[CallRecord] = []
        self._pending_start: Dict[str, float] = {}   # run_id → start time
        self._current_role: str = "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_role(self, role: str):
        """Call this before each agent invocation to tag the records."""
        self._current_role = role

    def reset(self):
        """Clear all records (call between cases)."""
        self.records.clear()
        self._pending_start.clear()

    def summary(self) -> dict:
        """
        Return aggregate stats suitable for JSON logging.

        {
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_tokens": int,
            "total_latency_ms": float,
            "calls": int,
            "by_role": {
                "prosecutor": {"input_tokens": ..., "output_tokens": ..., ...},
                ...
            }
        }
        """
        by_role: Dict[str, dict] = {}
        total_in = total_out = total_tok = 0
        total_lat = 0.0

        for r in self.records:
            total_in += r.input_tokens
            total_out += r.output_tokens
            total_tok += r.total_tokens
            total_lat += r.latency_ms

            if r.role not in by_role:
                by_role[r.role] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0.0,
                    "calls": 0,
                }
            entry = by_role[r.role]
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["total_tokens"] += r.total_tokens
            entry["latency_ms"] += r.latency_ms
            entry["calls"] += 1

        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tok,
            "total_latency_ms": round(total_lat, 1),
            "calls": len(self.records),
            "by_role": by_role,
        }

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     *, run_id, **kwargs):
        self._pending_start[str(run_id)] = time.perf_counter()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages,
                            *, run_id, **kwargs):
        self._pending_start[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        rid = str(run_id)
        start = self._pending_start.pop(rid, None)
        latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

        # Extract token usage — works for both OpenAI and Ollama
        input_tokens = 0
        output_tokens = 0
        model_name = ""

        # Try response.llm_output first (OpenAI style)
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        model_name = llm_output.get("model_name", llm_output.get("model", ""))

        # Try per-generation metadata (Ollama often puts it here)
        if input_tokens == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", {}) or {}
                    # Ollama returns prompt_eval_count / eval_count
                    input_tokens += meta.get("prompt_eval_count", 0)
                    output_tokens += meta.get("eval_count", 0)

                    # Also check response_metadata on AIMessage
                    msg = getattr(gen, "message", None)
                    if msg:
                        rmeta = getattr(msg, "response_metadata", {}) or {}
                        if not input_tokens:
                            input_tokens = rmeta.get("prompt_eval_count", 0)
                        if not output_tokens:
                            output_tokens = rmeta.get("eval_count", 0)
                        if not model_name:
                            model_name = rmeta.get("model", "")

                        # LangChain >=0.2 usage_metadata
                        umeta = getattr(msg, "usage_metadata", {}) or {}
                        if not input_tokens:
                            input_tokens = umeta.get("input_tokens", 0)
                        if not output_tokens:
                            output_tokens = umeta.get("output_tokens", 0)

        total_tokens = input_tokens + output_tokens

        self.records.append(CallRecord(
            role=self._current_role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 1),
            model=model_name,
        ))
