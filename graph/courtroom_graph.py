# graph/courtroom_graph.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from utils.pretty_print import (
    print_case_header,
    print_round,
    print_jury_verdict,
    print_summary,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CourtroomState(TypedDict):
    case: str
    round: int
    max_rounds: int
    prosecution_args: List[str]
    defense_args: List[str]
    judge_decisions: List[str]      # CONTINUE or CLOSE per round
    jury_votes: List[dict]          # each juror's verdict + confidence + reason
    final_verdict: str | None       # majority vote result
    verdict_confidence: float | None
    deliberation_complete: bool
    grounding_failures: int         # total filter failures across the whole trial (for reporting)
    consecutive_failures: int       # failures in the current turn only — resets to 0 on any pass
    last_filter_passed: bool        # set by citation filter each turn


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_courtroom_graph(prosecutor, defender, judge, jury, citation_filter):
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
    """
    graph = StateGraph(CourtroomState)

    # ------------------------------------------------------------------
    # Node definitions (thin wrappers that add pretty-print side-effects)
    # ------------------------------------------------------------------

    def prosecute_node(state: dict) -> dict:
        if state["round"] == 0:
            print_case_header(state)
        return prosecutor.argue(state)

    def filter_prosecution_node(state: dict) -> dict:
        return citation_filter.validate_prosecution(state)

    def defend_node(state: dict) -> dict:
        return defender.argue(state)

    def filter_defense_node(state: dict) -> dict:
        return citation_filter.validate_defense(state)

    def judge_node(state: dict) -> dict:
        new_state = judge.moderate(state)
        round_idx = len(new_state["prosecution_args"]) - 1
        print_round(new_state, round_idx)
        return new_state

    def jury_node(state: dict) -> dict:
        new_state = jury.deliberate(state)
        print_jury_verdict(new_state)
        print_summary(new_state)
        return new_state

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

def initial_state(case: str, max_rounds: int = 4) -> CourtroomState:
    """Return a fresh CourtroomState for a given AI response under review."""
    return CourtroomState(
        case=case,
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
    )
