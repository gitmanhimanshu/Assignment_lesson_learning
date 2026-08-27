"""Turning repeated failures into standing instructions for the generator.

The split here is the point of the whole module, so it is worth stating plainly: SQL decides
WHEN to learn, the LLM decides WHAT the rule says. A check that has failed three times across
the history is a fact about the database, not a judgement call, and asking a model "have you
been failing this a lot?" would make the trigger as unreliable as the thing it is meant to
correct. The model is only used for the part it is actually good at -- turning a pile of
failure reasons into one instruction a writer can follow.
"""

import sqlite3
from datetime import datetime

from src import llm
from src.rubric import check_by_name

# Three failures of the same check, anywhere in the history. Two is noise -- a model has a bad
# day, or one topic is genuinely harder. Three is a pattern worth writing down.
FAIL_THRESHOLD = 3

RULE_PROMPT = """A writer producing beginner lessons keeps making the same mistake. A reviewer
has rejected their work for it {count} times.

The check they keep failing: {check_name}
What the check asks: {question}

The reasons the reviewer gave, and the lines they quoted:
{examples}

Write ONE instruction for this writer that would stop this from happening again. Rules:
- One or two sentences, no more.
- Concrete and testable. "Be clearer" is useless. "Never open with a definition longer than
  fifteen words" is useful.
- Write it as an instruction to the writer, starting with a verb.
- Output only the instruction. No preamble, no quotes around it, no explanation."""


def repeated_failures(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT check_name, COUNT(*) AS fails FROM check_results"
        " WHERE passed = 0 GROUP BY check_name HAVING fails >= ?",
        (FAIL_THRESHOLD,),
    ).fetchall()


def evolve(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Add a rule for each check that has crossed the threshold and has none yet.

    Returns the (check_name, rule_text) pairs written this time, so the caller can show the
    user what the system just learned.
    """
    already_covered = {
        r["trigger_check"]
        for r in conn.execute("SELECT trigger_check FROM learned_rules WHERE active = 1")
    }

    learned = []
    for row in repeated_failures(conn):
        check_name = row["check_name"]
        if check_name in already_covered:
            continue

        rule = _write_rule(conn, check_name, row["fails"])
        conn.execute(
            "INSERT INTO learned_rules (trigger_check, rule_text, added_at, active)"
            " VALUES (?, ?, ?, 1)",
            (check_name, rule, datetime.now().isoformat(timespec="seconds")),
        )
        learned.append((check_name, rule))

    conn.commit()
    return learned


def _write_rule(conn: sqlite3.Connection, check_name: str, count: int) -> str:
    rows = conn.execute(
        "SELECT reason, evidence FROM check_results"
        " WHERE passed = 0 AND check_name = ? ORDER BY id DESC LIMIT 5",
        (check_name,),
    ).fetchall()

    examples = "\n\n".join(
        f"- reason: {r['reason']}\n  quoted: {r['evidence']!r}" for r in rows
    )
    prompt = RULE_PROMPT.format(
        count=count,
        check_name=check_name,
        question=check_by_name(check_name).question,
        examples=examples,
    )

    # Written by the generator model on purpose: this is a writing instruction, and the model
    # that has to follow it phrases it in terms it will actually act on.
    rule = llm.text_of(llm.invoke_with_retry(llm.generator_llm(), prompt))
    if not rule:
        raise RuntimeError(f"empty rule generated for {check_name}")
    return rule
