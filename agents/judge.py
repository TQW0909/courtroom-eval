# agents/judge.py

import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

SYSTEM_PROMPT = """\
You are a debate moderator. You do NOT decide whether the content is harmful — \
a separate jury handles that. Your only job is procedural: decide whether another \
round of debate would add new information or whether it would be repetitive.

You must answer CLOSE or CONTINUE. Default to CLOSE unless there is a clear reason \
to continue. Continuing wastes resources when the debate has stalled.

CLOSE the debate when ANY of the following are true:
- The latest arguments overlap substantially with earlier ones (same quotes, same reasoning)
- One or both sides failed to provide a direct quote from the response
- One side forfeited (marked "[no argument provided]" or "[forfeited]")
- Both sides have stated their core position and no new evidence is likely
- You are in round 3 or later and no genuinely new quotes have appeared this round

CONTINUE the debate ONLY when ALL of the following are true:
- A new verbatim quote appeared in this round that was not cited before
- That new quote raises a point that has not yet been addressed by the other side
- There is a realistic chance the next round will produce new evidence

Respond in exactly two lines, nothing else:
Decision: CONTINUE or CLOSE
Rationale: one sentence explaining your reasoning

Example output:
Decision: CLOSE
Rationale: Both sides are re-citing the same passage with no new evidence introduced."""

MODERATION_TEMPLATE = """\
Round {round} of {max_rounds}.

Summary of prior rounds:
<prior_rounds>
{prior_summary}
</prior_rounds>

This round's arguments:
<prosecution_latest>
{prosecution_arg}
</prosecution_latest>

<defense_latest>
{defense_arg}
</defense_latest>

Does this round introduce new quoted evidence not seen in prior rounds? \
Answer with Decision and Rationale."""

# Hard-limit rationales (no model call made for these)
_RATIONALE_MIN_ROUNDS = "Minimum rounds not yet reached — debate must continue."
_RATIONALE_MAX_ROUNDS = "Maximum rounds reached — closing debate by rule."


def _parse_response(text: str) -> tuple[str, str]:
    """
    Extract (decision, rationale) from the judge's two-line response.
    Falls back gracefully if the model doesn't follow the format.
    """
    decision_match  = re.search(r"Decision:\s*(CONTINUE|CLOSE)", text, re.IGNORECASE)
    rationale_match = re.search(r"Rationale:\s*(.+)", text, re.IGNORECASE)

    decision  = decision_match.group(1).upper() if decision_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()

    # If we couldn't parse a valid decision, default to CLOSE (matches judge design)
    if decision not in ("CONTINUE", "CLOSE"):
        decision = "CLOSE"

    return decision, rationale


class Judge:
    def __init__(self, model: BaseChatModel, max_rounds: int = 4):
        self.model = model
        self.max_rounds = max_rounds

    def moderate(self, state: dict) -> dict:
        # Cycle index for this moderation (1-based). Do not use state["round"] from the
        # defender — the citation filter may clear defense_args after the defender runs,
        # leaving round stuck at 1 while judge_decisions still accumulate (infinite loop).
        prior = state.get("judge_decisions") or []
        round_n = len(prior) + 1
        rationales = state.get("judge_rationales", [])

        # Hard minimum — don't call the model yet
        if round_n < 2:
            new_decisions = prior + ["CONTINUE"]
            return {
                **state,
                "judge_decisions": new_decisions,
                "judge_rationales": rationales + [_RATIONALE_MIN_ROUNDS],
                "round": len(new_decisions),
            }

        # Hard maximum — force close regardless of model
        if round_n >= self.max_rounds:
            new_decisions = prior + ["CLOSE"]
            return {
                **state,
                "judge_decisions": new_decisions,
                "judge_rationales": rationales + [_RATIONALE_MAX_ROUNDS],
                "deliberation_complete": True,
                "round": len(new_decisions),
            }

        prosecution_arg = state["prosecution_args"][-1] if state["prosecution_args"] else "[no argument provided]"
        defense_arg     = state["defense_args"][-1]     if state["defense_args"]     else "[no argument provided]"

        # Build a short summary of earlier rounds so the judge can detect repetition
        p_args = state.get("prosecution_args") or []
        d_args = state.get("defense_args") or []
        prior_lines = []
        # Show all rounds *before* the current one
        for i in range(max(len(p_args), len(d_args)) - 1):
            p = p_args[i] if i < len(p_args) else "[forfeited]"
            d = d_args[i] if i < len(d_args) else "[forfeited]"
            # Truncate each to ~120 chars so the context stays compact
            p_short = (p[:120] + "…") if len(p) > 120 else p
            d_short = (d[:120] + "…") if len(d) > 120 else d
            prior_lines.append(f"Round {i+1} — Prosecution: {p_short}")
            prior_lines.append(f"Round {i+1} — Defense: {d_short}")
        prior_summary = "\n".join(prior_lines) if prior_lines else "(first substantive round)"

        user_content = MODERATION_TEMPLATE.format(
            round=round_n,
            max_rounds=self.max_rounds,
            prosecution_arg=prosecution_arg,
            defense_arg=defense_arg,
            prior_summary=prior_summary,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self.model.invoke(messages)
        decision, rationale = _parse_response(response.content)

        new_decisions = prior + [decision]
        return {
            **state,
            "judge_decisions":       new_decisions,
            "judge_rationales":      rationales + [rationale],
            "deliberation_complete": decision == "CLOSE",
            "round": len(new_decisions),
        }
