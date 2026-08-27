"""Lesson evaluation.

The prompt never says the lesson was written by a model. Told that, a reviewer model softens
its verdicts and starts giving credit for effort. Here it is reviewing a colleague's draft
before it reaches learners, which is the frame that actually produces rejections.

The judged checks go in one structured call rather than one call per check. Nine calls per
attempt times three attempts would be twenty-seven requests per run, which does not survive a
free tier, and per-check calls also lose the reviewer's view of the lesson as a whole -- flow
and forward references are not visible one question at a time.

Two checks never reach the model at all. Counting long sentences and spotting words from a
fixed list have one correct answer each, so they are computed in Python and only the checks
that genuinely need judgement are asked. The model was answering both correctly, which is
worth saying plainly -- this is not a patch over a bad evaluator. It is that a countable
question should not be settled by sampling, and moving them out makes those two verdicts
exact, reproducible and free.
"""

from src import llm
from src.rubric import AUDIENCE, CHECK_NAMES, CHECKS
from src.schemas import CheckVerdict, Evaluation

JUDGED_CHECKS = [c for c in CHECKS if c.verify is None]
COMPUTED_CHECKS = [c for c in CHECKS if c.verify is not None]
JUDGED_NAMES = [c.name for c in JUDGED_CHECKS]

REVIEW_PROMPT = """You are a curriculum reviewer at an education company in India. You read
draft lessons before they reach learners and your job is to find what is wrong with them. A
reviewer who waves a weak draft through costs the company students who give up on the
subject. You are not here to encourage the writer.

THE READER this lesson has to serve:
{audience}

THE DRAFT LESSON, on the topic "{topic}":
<<<LESSON
{lesson}
LESSON

Answer every check below about that draft. For each check:

- Write your reason first, then the evidence, then the verdict. Decide last, not first.
- Evidence must be copied word for word out of the draft. Never paraphrase it. Never write a
  description of the draft in the evidence field -- quote the actual sentence or phrase that
  made you decide, whether the check passed or failed.
- Mark passed only if the draft satisfies the check exactly as written. A draft that mostly
  satisfies a check does not satisfy it.

CHECKS:
{checks}

Return exactly one verdict for each of the {count} checks, using the check names exactly as
written above."""


def build_prompt(lesson: str, topic: str) -> str:
    checks = "\n\n".join(
        f"{i}. {c.name}\n   {c.question}" for i, c in enumerate(JUDGED_CHECKS, start=1)
    )
    return REVIEW_PROMPT.format(
        audience=AUDIENCE, topic=topic, lesson=lesson, checks=checks, count=len(JUDGED_CHECKS)
    )


def evaluate_lesson(lesson: str, topic: str) -> Evaluation:
    prompt = build_prompt(lesson, topic)
    judged = _normalise(llm.invoke_with_retry(llm.evaluator_llm(Evaluation), prompt))

    verdicts = {v.check_name: v for v in judged.verdicts}
    for check in COMPUTED_CHECKS:
        passed, reason, evidence = check.verify(lesson)
        verdicts[check.name] = CheckVerdict(
            check_name=check.name, reason=reason, evidence=evidence, passed=passed
        )

    # Rubric order, so the printed report and the rejection log always read the same way.
    return Evaluation(verdicts=[verdicts[name] for name in CHECK_NAMES])


def _normalise(evaluation: Evaluation) -> Evaluation:
    """Reject an evaluation that does not answer the rubric.

    Constrained decoding guarantees the shape of the response, not its content: the model can
    still return ten verdicts, or invent a twelfth check name. A missing check is fatal,
    because treating "not answered" as "passed" is exactly the silent pass this system exists
    to prevent.
    """
    seen: dict[str, CheckVerdict] = {}
    unknown = []
    for verdict in evaluation.verdicts:
        if verdict.check_name not in JUDGED_NAMES:
            unknown.append(verdict.check_name)
        elif verdict.check_name not in seen:
            seen[verdict.check_name] = verdict

    missing = [name for name in JUDGED_NAMES if name not in seen]
    if missing:
        raise RuntimeError(
            f"evaluator answered {len(seen)}/{len(JUDGED_NAMES)} checks, "
            f"skipped {missing}. Usually means the response hit max_tokens."
        )
    if unknown:
        print(f"      evaluator invented check name(s), ignored: {unknown}")

    return Evaluation(verdicts=[seen[name] for name in JUDGED_NAMES])
