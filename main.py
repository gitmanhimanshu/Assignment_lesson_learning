import argparse
import sys
from datetime import datetime
from pathlib import Path

from src import config, memory
from src.evaluator import evaluate_lesson
from src.graph import build_graph
from src.report import rejection_log
from src.rubric import CHECKS

OUTPUT_DIR = Path("outputs")
BROKEN_LESSON = Path("demo/broken_lesson.md")


def run(topic: str) -> int:
    app = build_graph()
    final = app.invoke({"topic": topic})

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    lesson_path = OUTPUT_DIR / f"lesson-{stamp}.md"
    log_path = OUTPUT_DIR / f"rejection-log-{stamp}.md"

    lesson_path.write_text(final["lesson"], encoding="utf-8")
    with memory.connect() as conn:
        log = rejection_log(conn, final["run_id"], topic, final["status"])
    log_path.write_text(log, encoding="utf-8")

    print(f"\nstatus: {final['status']}")
    print(f"wrote {lesson_path}")
    print(f"wrote {log_path}")

    if final["status"] != "passed":
        print("\nthis lesson did not clear the rubric and is not approved to ship")
    return 0 if final["status"] == "passed" else 1


def demo_fail() -> int:
    """Feed a deliberately bad lesson straight to the evaluator, skipping generation.

    Written for the walkthrough recording. Skipping the generator is not just a shortcut: it
    means the evaluator is judging text nobody in this system wrote, so what it catches is
    not an artefact of the two models agreeing with each other.
    """
    lesson = BROKEN_LESSON.read_text(encoding="utf-8")
    print(f"evaluating {BROKEN_LESSON} ({len(lesson.split())} words), no generation step\n")

    evaluation = evaluate_lesson(lesson, "Introduction to RAG")
    for verdict in evaluation.verdicts:
        print(f"{'PASS' if verdict.passed else 'FAIL'}  {verdict.check_name}")
        if not verdict.passed:
            print(f"      why: {verdict.reason}")
            print(f"      ev : {verdict.evidence!r}")

    failed = len(evaluation.failures)
    print(f"\n{failed} of {len(CHECKS)} checks failed")
    return 0


def main() -> int:
    # Models write real typography -- non-breaking hyphens, curly quotes, en dashes. A default
    # Windows console is cp1252 and raises UnicodeEncodeError on the first one, killing a run
    # that has already paid for its API calls.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate a beginner lesson, judge it against a rubric, retry until it passes."
    )
    parser.add_argument("--topic", default="Introduction to RAG")
    parser.add_argument(
        "--demo-fail",
        action="store_true",
        help="evaluate demo/broken_lesson.md instead of generating anything",
    )
    args = parser.parse_args()

    if args.demo_fail:
        return demo_fail()

    print(f"topic: {args.topic}")
    print(f"generator: {config.GENERATOR_MODEL}")
    print(f"evaluator: {config.EVALUATOR_MODEL}")
    print(f"rubric: {len(CHECKS)} binary checks, max {config.MAX_ATTEMPTS} attempts\n")
    return run(args.topic)


if __name__ == "__main__":
    sys.exit(main())
