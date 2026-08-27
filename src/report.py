"""The rejection log.

Built by reading check_results back out of the database, not from whatever happened to be in
memory when the run finished. That distinction matters more than it looks: a log assembled
from local variables is a story the program tells about itself, and it stays convincing even
when the run went sideways. This one can only say what was actually recorded.
"""

import sqlite3

from src import memory
from src.rubric import check_by_name


def rejection_log(conn: sqlite3.Connection, run_id: int, topic: str, status: str) -> str:
    rows = memory.rejection_rows(conn, run_id)
    attempts = memory.attempt_count(conn, run_id)

    lines = [
        f"# Rejection log: {topic}",
        "",
        f"Run {run_id} | {attempts} attempt(s) | final status: **{status}**",
        "",
    ]

    if not rows:
        lines += ["The first draft cleared every check. Nothing was rejected."]
        return "\n".join(lines) + "\n"

    by_attempt: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_attempt.setdefault(row["attempt_no"], []).append(row)

    for attempt_no in range(1, attempts + 1):
        failures = by_attempt.get(attempt_no, [])
        if not failures:
            lines += [f"## Attempt {attempt_no} -- passed", "", "Every check passed.", ""]
            continue

        lines += [f"## Attempt {attempt_no} -- {len(failures)} check(s) failed", ""]
        for row in failures:
            lines += [
                f"### {row['check_name']}",
                "",
                f"**What was wrong:** {row['reason']}",
                "",
                "**Evidence the reviewer quoted:**",
                "",
                f"> {row['evidence']}",
                "",
            ]
            if attempt_no < attempts:
                lines += [
                    f"**Fed back on retry:** {check_by_name(row['check_name']).fail_hint}",
                    "",
                ]

    if status != "passed":
        lines += [
            "## Outcome",
            "",
            "The checks above were still failing when the attempt limit was reached. This "
            "lesson was not approved and should not ship without a human reading it first.",
            "",
        ]

    return "\n".join(lines) + "\n"
