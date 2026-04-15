# graph/courtroom_graph.py

from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, END



# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CourtroomState(TypedDict):
    case: str               # the AI-generated response under review
    case_prompt: str        # the original user prompt that produced the response
    round: int  # completed judge moderation steps (set in Judge; equals len(judge_decisions))
    max_rounds: int
    prosecution_args: List[str]
    defense_args: List[str]
    judge_decisions: List[str]      # CONTINUE or CLOSE per round
    judge_rationales: List[str]     # one-sentence procedural reasoning per round
    jury_votes: List[dict]          # each juror's verdict + confidence + reason
    final_verdict: Optional[str]    # majority vote result
    verdict_confidence: Optional[float]
    deliberation_complete: bool
    grounding_failures: int         # total filter failures across the whole trial (for reporting)
    consecutive_failures: int       # failures in the current turn only — resets to 0 on any pass
    last_filter_passed: bool        # set by citation filter each turn


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_courtroom_graph(prosecutor, defender, judge, jury, citation_filter,
                          token_tracker=None):
    """
    Build and compile the courtroom LangGraph.

    Flow (symmetric filter on both sides):
        prosecute → filter_prosecution → defend → filter_defense → judge
             ↑            ↓ (fail)                     ↓ (fail)
             └────────────┘               defend ←─────┘
                                              ↓ (pass)
                                           judge
                                          ↙       ↘
                                     CONTINUE    CLOSE
                                        ↓           ↓
                                    prosecute    jury → END

    Parameters
    ----------
    prosecutor      : Prosecutor instance  (.argue)
    defender        : Defender instance    (.argue)
    judge           : Judge instance       (.moderate)
    jury            : Jury instance        (.deliberate)
    citation_filter : CitationFilter       (.validate_prosecution, .validate_defense)
    token_tracker   : TokenTracker | None  (optional — tags each LLM call by role)
    """
    graph = StateGraph(CourtroomState)

    def _tag(role: str):
        """Set the current role on the token tracker (no-op if tracker is None)."""
        if token_tracker:
            token_tracker.set_role(role)

    # ------------------------------------------------------------------
    # Node definitions — pure state transformers, no side-effects
    # ------------------------------------------------------------------

    def prosecute_node(state: dict) -> dict:
        _tag("prosecutor")
        return prosecutor.argue(state)

    def filter_prosecution_node(state: dict) -> dict:
        return citation_filter.validate_prosecution(state)

    def defend_node(state: dict) -> dict:
        _tag("defender")
        return defender.argue(state)

    def filter_defense_node(state: dict) -> dict:
        return citation_filter.validate_defense(state)

    def judge_node(state: dict) -> dict:
        _tag("judge")
        return judge.moderate(state)

    def jury_node(state: dict) -> dict:
        _tag("juror")
        return jury.deliberate(state)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------

    graph.add_node("prosecute", prosecute_node)
    graph.add_node("filter_prosecution", filter_prosecution_node)
    graph.add_node("defend", defend_node)
    graph.add_node("filter_defense", filter_defense_node)
    graph.add_node("judge", judge_node)
    graph.add_node("jury", jury_node)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    graph.set_entry_point("prosecute")

    # Prosecution always goes through its filter first
    graph.add_edge("prosecute", "filter_prosecution")

    def route_after_prosecution_filter(state: dict) -> str:
        """
        consecutive_failures == 1 means this is the first failure — give one retry.
        consecutive_failures == 0 after a failure means the turn was forfeited — proceed.
        """
        if not state.get("last_filter_passed", True) and state.get("consecutive_failures", 0) == 1:
            return "prosecute"
        return "defend"

    graph.add_conditional_edges(
        "filter_prosecution",
        route_after_prosecution_filter,
        {"prosecute": "prosecute", "defend": "defend"},
    )

    # Defense always goes through its filter next
    graph.add_edge("defend", "filter_defense")

    def route_after_defense_filter(state: dict) -> str:
        """
        consecutive_failures == 1 means this is the first failure — give one retry.
        consecutive_failures == 0 after a failure means the turn was forfeited — proceed.
        """
        if not state.get("last_filter_passed", True) and state.get("consecutive_failures", 0) == 1:
            return "defend"
        return "judge"

    graph.add_conditional_edges(
        "filter_defense",
        route_after_defense_filter,
        {"defend": "defend", "judge": "judge"},
    )

    # Judge routes: another round or call the jury
    def route_after_judge(state: dict) -> str:
        if state.get("deliberation_complete", False):
            return "jury"
        return "prosecute"

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"prosecute": "prosecute", "jury": "jury"},
    )

    # Jury is the terminal node
    graph.add_edge("jury", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: default initial state
# ---------------------------------------------------------------------------

def initial_state(case: str, max_rounds: int = 4, case_prompt: str = "") -> CourtroomState:
    """Return a fresh CourtroomState for a given AI response under review."""
    return CourtroomState(
        case=case,
        case_prompt=case_prompt,
        round=0,
        max_rounds=max_rounds,
        prosecution_args=[],
        defense_args=[],
        judge_decisions=[],
        jury_votes=[],
        final_verdict=None,
        verdict_confidence=None,
        deliberation_complete=False,
        grounding_failures=0,
        consecutive_failures=0,
        last_filter_passed=True,
        judge_rationales=[],
    )
