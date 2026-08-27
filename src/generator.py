"""Lesson generation.

The generator is never shown the rubric. It gets the audience and the writing task, and on a
retry it gets only the specific points the reviewer raised. Handing it the full checklist up
front would make attempt one pass every time, which would make the evaluator decorative and
prove nothing about the loop. The rubric is the bar, not the brief.
"""

from src import llm
from src.rubric import AUDIENCE
from src.schemas import CheckVerdict

BRIEF = """You are writing a lesson for an introductory AI course in India.

THE READER:
{audience}

THE TASK:
Write one standalone lesson that teaches "{topic}" to this reader, starting from zero
knowledge. Cover what it is, why it matters, and how it works.

FORMAT:
- Markdown, roughly 700 to 900 words.
- Open with a single "# " title, then use "## " section headings.
- No emoji anywhere.

Output only the lesson itself. No preamble, no closing note, no commentary on your process."""

# Kept separate from the brief so it is obvious in the logs which instructions the system
# taught itself and which ones were written by hand.
RULES = """

HOUSE RULES, learned from lessons that were rejected before:
{rules}"""

REVISION = """
A reviewer read your previous draft and rejected it. Their findings:

{findings}

Rewrite the lesson completely, fixing every point above. Keep whatever was already working.
Do not simply patch the quoted sentences -- the same problem is usually present elsewhere in
the draft as well."""


def build_prompt(
    topic: str,
    failures: list[CheckVerdict],
    fail_hints: dict[str, str],
    rules: list[str] | None = None,
) -> str:
    prompt = BRIEF.format(audience=AUDIENCE, topic=topic)

    if rules:
        prompt += RULES.format(rules="\n".join(f"- {r}" for r in rules))

    if not failures:
        return prompt

    findings = []
    for i, failure in enumerate(failures, start=1):
        findings.append(
            f"[{i}] {failure.check_name}\n"
            f"    problem: {failure.reason}\n"
            f"    they quoted: {failure.evidence!r}\n"
            f"    fix: {fail_hints[failure.check_name]}"
        )

    return prompt + REVISION.format(findings="\n\n".join(findings))


def generate_lesson(
    topic: str,
    failures: list[CheckVerdict],
    fail_hints: dict[str, str],
    rules: list[str] | None = None,
) -> str:
    prompt = build_prompt(topic, failures, fail_hints, rules)
    lesson = llm.text_of(llm.invoke_with_retry(llm.generator_llm(), prompt))

    if not lesson:
        raise RuntimeError(f"generator returned an empty lesson for topic {topic!r}")
    return lesson
