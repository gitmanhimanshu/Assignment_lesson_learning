"""The pipeline as a LangGraph state machine.

This is the same loop that ran as a plain for-loop in earlier versions, and it behaves
identically. What the graph buys is that the retry is an edge rather than a control-flow
statement: route_after_eval is the only place that decides whether to go round again, and
the attempt cap lives inside it, so there is exactly one answer to "how can this loop fail
to terminate". In a for-loop that decision was spread across a break, a counter and an
else clause.

    load_memory -> generate -> evaluate -> (generate | save_memory) -> END
"""

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import config, evolution, memory
from src.evaluator import evaluate_lesson
from src.generator import generate_lesson
from src.rubric import CHECKS
from src.schemas import CheckVerdict

FAIL_HINTS = {c.name: c.fail_hint for c in CHECKS}

# Every draft lands here the moment it exists. A lesson costs a generation call, and an
# evaluator that dies mid-run should not take it down with it -- that happened twice while
# this was being built.
DRAFT_PATH = Path("outputs/last-draft.md")


class RunState(TypedDict):
    topic: str
    attempt: int
    lesson: str
    rules: list[str]
    failures: list[CheckVerdict]
    history: list[list[CheckVerdict]]
    status: str
    run_id: int
    learned: list[tuple[str, str]]


def load_memory(state: RunState) -> dict:
    with memory.connect() as conn:
        rules = memory.active_rules(conn)

    if rules:
        print(f"loaded {len(rules)} learned rule(s) from previous runs:")
        for rule in rules:
            print(f"  - {rule}")
    else:
        print("no learned rules yet -- this is an early run")

    return {"rules": rules, "attempt": 0, "failures": [], "history": []}


def generate(state: RunState) -> dict:
    attempt = state["attempt"] + 1
    print(f"\nattempt {attempt}/{config.MAX_ATTEMPTS}")
    print("  generating...")

    lesson = generate_lesson(state["topic"], state["failures"], FAIL_HINTS, state["rules"])
    DRAFT_PATH.parent.mkdir(exist_ok=True)
    DRAFT_PATH.write_text(lesson, encoding="utf-8")

    print(f"  generated {len(lesson.split())} words -> {DRAFT_PATH}")
    return {"lesson": lesson, "attempt": attempt}


def evaluate(state: RunState) -> dict:
    print("  evaluating...")
    evaluation = evaluate_lesson(state["lesson"], state["topic"])

    for verdict in evaluation.verdicts:
        print(f"  {'PASS' if verdict.passed else 'FAIL'}  {verdict.check_name}")
        if not verdict.passed:
            print(f"        why: {verdict.reason}")
            print(f"        ev : {verdict.evidence!r}")

    status = "passed" if evaluation.all_passed else "failed"
    return {
        "failures": evaluation.failures,
        "history": state["history"] + [evaluation.verdicts],
        "status": status,
    }


def route_after_eval(state: RunState) -> str:
    """The only place the loop can decide to go round again, and the only place the cap is
    enforced. Everything else in the graph is a straight line."""
    if state["status"] == "passed":
        print(f"\n  all {len(CHECKS)} checks passed on attempt {state['attempt']}")
        return "save_memory"

    names = ", ".join(f.check_name for f in state["failures"])
    print(f"\n  {len(state['failures'])} check(s) failed: {names}")

    if state["attempt"] >= config.MAX_ATTEMPTS:
        print(f"  out of attempts after {config.MAX_ATTEMPTS} tries")
        return "save_memory"

    print("  feeding the reasons back and regenerating")
    return "generate"


def save_memory(state: RunState) -> dict:
    status = "passed" if state["status"] == "passed" else "flagged_for_review"
    with memory.connect() as conn:
        run_id = memory.save_run(conn, state["topic"], status, state["history"])
        learned = evolution.evolve(conn)

    if learned:
        print(f"\nlearned {len(learned)} new rule(s) from repeated failures:")
        for check_name, rule in learned:
            print(f"  [{check_name}] {rule}")

    return {"status": status, "run_id": run_id, "learned": learned}


def build_graph():
    graph = StateGraph(RunState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("save_memory", save_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_eval, {"generate": "generate", "save_memory": "save_memory"}
    )
    graph.add_edge("save_memory", END)
    return graph.compile()
