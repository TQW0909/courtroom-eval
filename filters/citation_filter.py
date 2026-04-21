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


def _is_grounded(quote: str, source: str, threshold: int = 8) -> bool:
    """
    Check whether *quote* is grounded in *source*.

    1. Exact substring match (after normalisation) → pass.
    2. If the quote is long enough (≥ threshold words), check whether a
       *majority* of contiguous 4-word windows from the quote appear in the
       source.  This catches minor imprecisions while rejecting fabrications
       that share only a few words with the source.
    """
    nq = _normalize(quote)
    ns = _normalize(source)

    # Reject very short "quotes" (1-2 words) — too easy to fabricate
    if len(nq.split()) < 3:
        return False

    # Fast path: exact substring
    if nq in ns:
        return True

    # Fuzzy path: sliding-window word overlap for longer quotes
    words = nq.split()
    if len(words) < threshold:
        return False  # short quotes must match exactly

    window = min(4, len(words))
    hits = 0
    total = len(words) - window + 1
    for i in range(total):
        fragment = " ".join(words[i : i + window])
        if fragment in ns:
            hits += 1

    # Require >50% of windows to match — a fabricated quote with a few
    # real words sprinkled in won't pass
    return hits > total / 2


def _redact_quote_line(argument: str, quote: str, tag: str = "UNVERIFIED") -> str:
    """
    Find the line in *argument* that contains *quote* and strike it through
    with a [tag] marker.  Returns the modified argument text.

    Handles both straight and curly quote wrappers around the phrase.
    """
    for line in argument.splitlines():
        # Check if this line contains the quote (normalised comparison)
        if _normalize(quote) in _normalize(line):
            redacted_line = f"[{tag} — REMOVED: {quote[:60]}]"
            return argument.replace(line, redacted_line, 1)
    return argument


def validate_argument(argument: str, response_text: str, prompt_text: str = "") -> tuple[bool, str]:
    """
    Returns (passes: bool, cleaned_argument: str).

    Every quoted phrase is checked against the RESPONSE text.  Quotes that
    are fabricated (not in the response) or that only match the PROMPT are
    redacted in-place — the cleaned argument is returned with those lines
    replaced by [UNVERIFIED — REMOVED: …] markers.

    The argument passes if at least one grounded quote survives after
    redaction.  The jury only ever sees the cleaned version.

    If the argument contains a REASONING: section (refusal fallback) and
    no quotes, it passes automatically — the jury weighs it at lower weight.
    """
    quotes = _extract_quotes(argument)

    # Allow REASONING-only responses through (refusal fallback)
    if not quotes:
        if "REASONING:" in argument:
            return True, argument
        return False, argument  # no quotes and no reasoning — reject

    cleaned = argument
    grounded_count = 0

    for q in quotes:
        in_response = _is_grounded(q, response_text)

        if in_response:
            grounded_count += 1
            continue  # keep this quote as-is

        # Quote is NOT in the response — redact it
        if prompt_text and _is_grounded(q, prompt_text):
            cleaned = _redact_quote_line(cleaned, q, tag="PROMPT-ONLY")
        else:
            cleaned = _redact_quote_line(cleaned, q, tag="FABRICATED")

    if grounded_count == 0:
        return False, cleaned  # all quotes were fabricated or prompt-only

    return True, cleaned


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
        passes, cleaned_arg = validate_argument(
            latest_arg,
            response_text=state["case"],
            prompt_text=state.get("case_prompt", ""),
        )

        if passes:
            # Replace the latest argument with the cleaned version (fabricated
            # quotes redacted).  The jury only sees verified evidence.
            cleaned_args = args[:-1] + [cleaned_arg]
            return {
                **state,
                args_key: cleaned_args,
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


class NoOpFilter:
    """
    Pass-through filter for ablation studies (--no-filter).
    Always passes — arguments are never dropped or retried.
    """

    def validate_prosecution(self, state: dict) -> dict:
        return {**state, "last_filter_passed": True, "consecutive_failures": 0}

    def validate_defense(self, state: dict) -> dict:
        return {**state, "last_filter_passed": True, "consecutive_failures": 0}
