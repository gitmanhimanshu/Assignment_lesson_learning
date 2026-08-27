"""Import runs recorded in outputs/*.json into memory.db.

The loop existed before the database did, and those early runs are real: real generations,
real verdicts, written to disk as they happened. This replays them into the schema rather
than throwing the history away and paying for the same runs again. It is not a fixture
generator -- it refuses to invent anything, and a run already imported is skipped.
"""

import json
import sys
from pathlib import Path

from src import memory
from src.schemas import CheckVerdict


def main() -> int:
    files = sorted(Path("outputs").glob("run-*.json"))
    if not files:
        print("no outputs/run-*.json to import")
        return 1

    with memory.connect() as conn:
        existing = {
            (r["topic"], r["attempts"], r["final_status"])
            for r in conn.execute("SELECT topic, attempts, final_status FROM runs")
        }

        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            attempts = [
                [CheckVerdict(**v) for v in entry["verdicts"]]
                for entry in data["log"]
                if entry["verdicts"]
            ]
            if not attempts:
                print(f"skip {path.name}: no verdicts recorded (run crashed before evaluation)")
                continue

            key = (data["topic"], len(attempts), data["status"])
            if key in existing:
                print(f"skip {path.name}: already imported")
                continue

            run_id = memory.save_run(conn, data["topic"], data["status"], attempts)
            existing.add(key)
            print(f"imported {path.name} as run {run_id} ({len(attempts)} attempts)")

        total = conn.execute("SELECT COUNT(*) AS n FROM check_results").fetchone()["n"]
        print(f"\ncheck_results now holds {total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
