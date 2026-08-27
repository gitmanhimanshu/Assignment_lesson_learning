"""The rubric: what "good enough to ship" actually means, as data rather than prose.

Checks live here as structured objects so that three different consumers read the same
definition -- the evaluator prompt renders them as questions, the generator prompt renders
the fail_hints as feedback, and the evolution query groups failures by check name. If these
were strings scattered through prompt templates, those three would drift apart.
"""

import re
from dataclasses import dataclass
from typing import Callable

# Everything below is calibrated to this reader. It is deliberately specific: "beginner" is
# not a useful target, and a rubric written for a generic beginner would happily pass content
# this learner cannot actually read.
AUDIENCE = """A 12th-grade graduate in India who studied in a non-English-medium school.
They have limited English vocabulary and read English slowly. They have no background in
programming, in mathematics beyond school level, or in machine learning. They are motivated
and want to start a career in AI. They can follow a clear explanation, but they will stop
reading at a long sentence full of unfamiliar words, and they cannot fill in a gap the
writer left out."""


@dataclass(frozen=True)
class Check:
    name: str
    question: str  # asked of the evaluator; phrased so that "yes" means pass
    fail_hint: str  # fed back to the generator verbatim when this check fails

    # Some checks are arithmetic, not judgement. Where a check can be settled by counting,
    # this holds the function that counts and the evaluator is never asked about it. The
    # model was in fact getting these right, so this is not a fix for a wrong answer -- it is
    # that "three or more sentences over 25 words" has one correct answer, and a question with
    # one correct answer should not be decided by sampling. It is also exact, reproducible,
    # free, and it shortens the prompt.
    verify: Callable[[str], tuple[bool, str, str]] | None = None


MAX_SENTENCE_WORDS = 25
LONG_SENTENCE_LIMIT = 3

BANNED_WORDS = [
    "leverage",
    "utilize",
    "facilitate",
    "paradigm",
    "robust",
    "comprehensive",
    "seamless",
    "myriad",
    "delve",
    "furthermore",
    "subsequently",
    "crucial",
    "pivotal",
]


def _sentences(lesson: str) -> list[str]:
    """Prose sentences only, never spanning a line break.

    Splitting the whole document on punctuation glues a bold sub-heading to the paragraph
    under it, because a heading has no full stop -- which inflated one measurement to a
    37-word "sentence" that was really a heading plus a line. Markdown from a model puts each
    paragraph on its own line, so splitting per line first is both simpler and more accurate.

    The sentence splitter within a line is deliberately crude. It over-splits on "e.g." and
    similar, which makes it lenient rather than harsh, and lenient is the safe direction for
    a check that hard-fails.
    """
    sentences = []
    for line in lesson.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        line = re.sub(r"^[-*]\s+", "", line)  # list bullet
        line = re.sub(r"^\d+\.\s+", "", line)  # numbered item
        if re.fullmatch(r"\*\*.*\*\*:?", line):  # bold-only line, a heading in practice
            continue

        sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return sentences


def check_sentence_length(lesson: str) -> tuple[bool, str, str]:
    sentences = _sentences(lesson)
    long = [s for s in sentences if len(s.split()) > MAX_SENTENCE_WORDS]
    longest = max(sentences, key=lambda s: len(s.split()), default="")

    if len(long) >= LONG_SENTENCE_LIMIT:
        worst = max(long, key=lambda s: len(s.split()))
        return (
            False,
            f"{len(long)} of {len(sentences)} sentences run over {MAX_SENTENCE_WORDS} words. "
            f"The limit is {LONG_SENTENCE_LIMIT}. The longest is {len(worst.split())} words.",
            worst,
        )
    return (
        True,
        f"{len(long)} of {len(sentences)} sentences run over {MAX_SENTENCE_WORDS} words, "
        f"under the limit of {LONG_SENTENCE_LIMIT}.",
        longest,
    )


def check_vocabulary(lesson: str) -> tuple[bool, str, str]:
    found = []
    for word in BANNED_WORDS:
        match = re.search(rf"\b{word}\w*", lesson, re.I)
        if match:
            found.append(match)

    if found:
        first = min(found, key=lambda m: m.start())
        sentence = next(
            (s for s in _sentences(lesson) if first.group(0).lower() in s.lower()),
            first.group(0),
        )
        words = sorted({m.group(0).lower() for m in found})
        return False, f"Uses advanced words a simple one would cover: {', '.join(words)}.", sentence

    return (
        True,
        f"None of the {len(BANNED_WORDS)} words on the advanced-vocabulary list appear.",
        _sentences(lesson)[0] if _sentences(lesson) else lesson[:80],
    )


# Ordered the way a reader meets the problems: truth first, then readability, then teaching
# quality, then structure.
CHECKS: list[Check] = [
    Check(
        name="factually_accurate",
        question=(
            "Is every factual claim about the topic correct? Fail if any statement is wrong, "
            "is stated as a universal fact when it is only true of some implementations, or "
            "promises an absolute guarantee the technology does not actually give -- for "
            "example claiming it never makes mistakes, or always returns a correct result."
        ),
        fail_hint=(
            "Correct or remove the inaccurate claim. Replace absolute promises with honest "
            "limits: say a problem is reduced, not that it is eliminated. If something varies "
            "between systems, say that it varies instead of stating one case as the rule."
        ),
    ),
    Check(
        name="no_invented_specifics",
        question=(
            "Does the lesson avoid passing invented information off as real-world fact? Fail "
            "if it states numbers, percentages, dates, benchmark results, research citations, "
            "or claims about real companies and products as established fact when they look "
            "fabricated. Made-up names and details inside a clearly hypothetical example are "
            "fine and must not fail this check."
        ),
        fail_hint=(
            "Delete the invented specific, or make it obvious it is an example. Teach the "
            "idea without fake precision: write 'much faster' rather than a percentage you "
            "cannot source."
        ),
    ),
    Check(
        name="defines_before_use",
        question=(
            "Is every technical term explained in plain words at the moment it is first "
            "used? Fail if any term is used first and defined later, or never defined."
        ),
        fail_hint=(
            "For every technical term, give its plain-English meaning in the same sentence "
            "or the sentence immediately after, the first time the term appears. Never "
            "postpone a definition to a later section."
        ),
    ),
    Check(
        name="short_sentences",
        question=(
            "Are the sentences short enough for a reader with limited English? Fail if "
            "three or more sentences in the lesson run longer than about 25 words."
        ),
        fail_hint=(
            "Split the long sentences. One idea per sentence, most sentences under 20 words. "
            "Prefer a full stop over a comma or a semicolon."
        ),
        verify=check_sentence_length,
    ),
    Check(
        name="plain_vocabulary",
        question=(
            "Does the lesson use everyday English? Fail if it reaches for formal or advanced "
            "words where a simple one exists -- for example leverage, utilize, facilitate, "
            "paradigm, robust, comprehensive, seamless, myriad, delve, furthermore, "
            "subsequently, crucial, pivotal."
        ),
        fail_hint=(
            "Replace advanced words with everyday ones: use instead of leverage or utilize, "
            "help instead of facilitate, strong instead of robust, so instead of therefore. "
            "Write the way you would explain it out loud to a friend."
        ),
        verify=check_vocabulary,
    ),
    Check(
        name="everyday_analogy",
        question=(
            "Does the lesson use at least one analogy drawn from ordinary daily life to make "
            "the main idea concrete? Fail if there is no analogy, or if the analogy depends "
            "on knowledge this reader is unlikely to have."
        ),
        fail_hint=(
            "Add one analogy from everyday life that this reader definitely knows -- a "
            "library, a shop, an exam, a phone contact list. Explain how the analogy maps "
            "onto the topic, do not just name it."
        ),
    ),
    Check(
        name="concrete_example_present",
        question=(
            "Is there at least one specific example with real details -- an actual question, "
            "an actual piece of text, actual content? Fail if examples are only gestured at, "
            "such as 'for example, imagine a chatbot' with nothing filled in."
        ),
        fail_hint=(
            "Write out one example in full with invented but specific details: the exact "
            "question a user types, the exact sentence that gets found, the exact answer that "
            "comes back."
        ),
    ),
    Check(
        name="example_walked_through",
        question=(
            "Is at least one example traced all the way through, with the example's own "
            "content shown at every stage and the final result written out in full for the "
            "reader to see? Fail if a stage only describes what happens in general instead of "
            "showing it happening to this example, and fail if the lesson says a result is "
            "produced without ever showing that result."
        ),
        fail_hint=(
            "Carry one example from start to finish. At each stage show what the example "
            "actually looks like at that point -- the exact text that gets found, the exact "
            "material handed to the next stage -- and finish by writing out the final answer "
            "word for word, not by saying that an answer is produced."
        ),
    ),
    Check(
        name="covers_what_and_why",
        question=(
            "Does the lesson clearly say what the topic is, and explain why it matters -- what "
            "problem it solves, or what goes wrong without it? Fail if either is missing or is "
            "only implied."
        ),
        fail_hint=(
            "State plainly what the topic is, early. Then show the problem it solves, ideally "
            "by describing what goes wrong when you do not have it."
        ),
    ),
    Check(
        name="explains_how_in_steps",
        question=(
            "Does the lesson explain how the topic works as a numbered or clearly separated "
            "sequence of steps? Fail if the mechanism is buried inside one dense paragraph."
        ),
        fail_hint=(
            "Break the mechanism into clearly separated numbered steps. One step does one "
            "thing, and the reader should be able to say what changed after each step."
        ),
    ),
    Check(
        name="logical_flow",
        question=(
            "Does the lesson build in order, so that nothing relies on an idea that has not "
            "been explained yet? Fail on any forward reference to an unexplained concept."
        ),
        fail_hint=(
            "Reorder so each idea only uses what came before it. If a later concept is needed "
            "early, explain it early or drop the reference."
        ),
    ),
]

CHECK_NAMES = [c.name for c in CHECKS]


def check_by_name(name: str) -> Check:
    for check in CHECKS:
        if check.name == name:
            return check
    raise KeyError(f"unknown check {name!r}; rubric defines {CHECK_NAMES}")
