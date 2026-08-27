"""What the system remembers between runs.

Only outcomes are stored: which run happened, which checks failed on which attempt, and the
rules learned from that. No lesson text and no topic knowledge lives in here. That is a
deliberate boundary -- the database is the system's record of its own behaviour, not a cache
of content. It also means the file stays small enough to commit, so a reviewer can open the
history the runs actually produced.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.schemas import CheckVerdict

DB_PATH = Path("memory/memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    final_status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS check_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    attempt_no INTEGER NOT NULL,
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    reason TEXT,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS learned_rules (
    id INTEGER PRIMARY KEY,
    trigger_check TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    added_at TIMESTAMP NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def active_rules(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT rule_text FROM learned_rules WHERE active = 1 ORDER BY added_at"
    ).fetchall()
    return [r["rule_text"] for r in rows]


def save_run(
    conn: sqlite3.Connection,
    topic: str,
    status: str,
    attempts: list[list[CheckVerdict]],
) -> int:
    """Write one run and every verdict from every attempt. Returns the new run id."""
    cur = conn.execute(
        "INSERT INTO runs (topic, attempts, final_status, created_at) VALUES (?, ?, ?, ?)",
        (topic, len(attempts), status, datetime.now().isoformat(timespec="seconds")),
    )
    run_id = cur.lastrowid

    conn.executemany(
        "INSERT INTO check_results (run_id, attempt_no, check_name, passed, reason, evidence)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (run_id, attempt_no, v.check_name, int(v.passed), v.reason, v.evidence)
            for attempt_no, verdicts in enumerate(attempts, start=1)
            for v in verdicts
        ],
    )
    conn.commit()
    return run_id


def rejection_rows(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """The failures of one run, in attempt order. This is what the rejection log is built
    from -- reading it back out of the database rather than from memory is the whole point,
    because a log assembled from local variables can say anything."""
    return conn.execute(
        "SELECT attempt_no, check_name, reason, evidence FROM check_results"
        " WHERE run_id = ? AND passed = 0 ORDER BY attempt_no, id",
        (run_id,),
    ).fetchall()


def attempt_count(conn: sqlite3.Connection, run_id: int) -> int:
    row = conn.execute(
        "SELECT attempts FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    return row["attempts"]
