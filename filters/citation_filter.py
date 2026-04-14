# filters/citation_filter.py

import re


def validate_argument(argument: str, source_text: str) -> tuple[bool, str]:
    """
    Returns (passes: bool, cleaned_argument: str).
    Strips or flags claims not supported by a direct quote from source_text.
    """
    quotes = re.findall(r'"([^"]+)"', argument)

    if not quotes:
        return False, argument  # no quotes at all — reject

    grounded = [q for q in quotes if q.lower() in source_text.lower()]
    if len(grounded) == 0:
        return False, argument  # all quotes are hallucinated

    return True, argument


class CitationFilter:
    """
    LangGraph-compatible wrapper around validate_argument.
    Runs symmetrically after both the Prosecutor and Defender.

    Retry logic (per turn, not per trial):
    - First failure  → increment consecutive_failures to 1, route back to retry.
    - Second failure → forfeit the turn: drop the bad argument from the list so
                       the transcript stays clean, reset consecutive_failures to 0,
                       and proceed to the next stage. The judge's CLOSE criteria
                       ("one side failed to provide quotes") will pick this up.
    - Any pass       → reset consecutive_failures to 0 and proceed.

    grounding_failures is a cumulative trial-level counter kept for reporting only.
    """

    def _check(self, state: dict, args_key: str) -> dict:
        args = state[args_key]
        if not args:
            return state

        latest_arg = args[-1]
        passes, _ = validate_argument(latest_arg, state["case"])

        if passes:
            return {
                **state,
                "last_filter_passed": True,
                "consecutive_failures": 0,
            }

        # Argument failed — is this a retry or a forfeit?
        consecutive = state.get("consecutive_failures", 0)

        if consecutive == 0:
            # First failure: give the agent one retry
            return {
                **state,
                "last_filter_passed": False,
                "consecutive_failures": 1,
                "grounding_failures": state.get("grounding_failures", 0) + 1,
            }
        else:
            # Second failure: forfeit — drop the bad argument, reset, proceed
            forfeited_args = args[:-1]  # remove the ungrounded argument entirely
            return {
                **state,
                args_key: forfeited_args,
                "last_filter_passed": False,
                "consecutive_failures": 0,
                "grounding_failures": state.get("grounding_failures", 0) + 1,
            }

    def validate_prosecution(self, state: dict) -> dict:
        """Node: validates the most recent prosecution argument."""
        return self._check(state, "prosecution_args")

    def validate_defense(self, state: dict) -> dict:
        """Node: validates the most recent defense argument."""
        return self._check(state, "defense_args")
