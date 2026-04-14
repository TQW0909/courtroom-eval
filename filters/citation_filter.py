# filters/citation_filter.py

import re
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip curly quotes → straight quotes."""
    text = text.lower()
    # Normalise unicode quotation marks to ASCII
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')   # « »
    # Collapse runs of whitespace (newlines, tabs, multiple spaces)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_quotes(argument: str) -> list[str]:
    """
    Pull every quoted span from the argument.
    Handles straight double quotes, curly double quotes, and «guillemets».
    """
    # Straight doubles
    straight = re.findall(r'"([^"]+)"', argument)
    # Curly doubles
    curly = re.findall(r'\u201c([^\u201d]+)\u201d', argument)
    # Guillemets
    guillemets = re.findall(r'\u00ab([^\u00bb]+)\u00bb', argument)
    return straight + curly + guillemets


def _is_grounded(quote: str, source: str, threshold: int = 12) -> bool:
    """
    Check whether *quote* is grounded in *source*.

    1. Exact substring match (after normalisation) → pass.
    2. If the quote is long enough (≥ threshold words), check whether any
       contiguous window of 6+ words from the quote appears in the source.
       This handles small model imprecisions (dropped punctuation, minor
       word substitutions) while still requiring substantial overlap.
    """
    nq = _normalize(quote)
    ns = _normalize(source)

    # Fast path: exact substring
    if nq in ns:
        return True

    # Fuzzy path: sliding-window word overlap for longer quotes
    words = nq.split()
    if len(words) < threshold:
        return False  # short quotes must match exactly

    window = min(6, len(words))
    for i in range(len(words) - window + 1):
        fragment = " ".join(words[i : i + window])
        if fragment in ns:
            return True

    return False


def validate_argument(argument: str, source_text: str) -> tuple[bool, str]:
    """
    Returns (passes: bool, cleaned_argument: str).
    An argument passes if it contains at least one quote that is grounded
    in the source text.
    """
    quotes = _extract_quotes(argument)

    if not quotes:
        return False, argument  # no quotes at all — reject

    grounded = [q for q in quotes if _is_grounded(q, source_text)]
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

        # Argument failed — drop it immediately in both cases so prosecution_args
        # stays clean. On first failure we route back so the agent tries again
        # with a fresh attempt. On second failure we forfeit and proceed.
        consecutive = state.get("consecutive_failures", 0)
        dropped_args = args[:-1]  # remove the bad argument either way

        if consecutive == 0:
            # First failure: drop the arg, give the agent one retry
            return {
                **state,
                args_key: dropped_args,
                "last_filter_passed": False,
                "consecutive_failures": 1,
                "grounding_failures": state.get("grounding_failures", 0) + 1,
            }
        else:
            # Second failure: forfeit — drop the arg, reset, proceed
            return {
                **state,
                args_key: dropped_args,
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
