# Self-evaluating lesson generator

Give it a topic. It writes a beginner lesson, hands the lesson to a second model that reviews
it against a rubric of pass/fail checks, and rewrites on any failure. Three attempts, then it
stops. If the lesson still has not cleared the rubric it is marked `flagged_for_review` rather
than shipped.

The topic is a CLI argument. Nothing about any particular subject is hardcoded, and no topic
knowledge is stored anywhere in the system.

```
load_memory -> generate -> evaluate -> (generate | save_memory) -> END
     |             |           |              |
   SQLite       Gemini       Groq       retry or stop
```

## Setup

Python 3.10 or newer.

```
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env         # then paste your two keys into .env
python scripts/check_providers.py
```

Both keys are free. Gemini: https://aistudio.google.com/apikey. Groq:
https://console.groq.com/keys.

`check_providers.py` does more than ping the APIs. It also feeds the evaluator a deliberately
jargon-heavy sentence and fails if the evaluator passes it, because an evaluator that approves
everything is worse than no evaluator at all.

## Running it

```
python main.py                                  # defaults to "Introduction to RAG"
python main.py --topic "How neural networks learn"
python main.py --demo-fail                      # judge a deliberately bad lesson, no generation
```

A run writes two files into `outputs/`: the lesson, and a rejection log listing every check
that failed, the sentence the reviewer quoted, and what was fed back before the rewrite.

`--demo-fail` skips generation entirely and points the evaluator at `demo/broken_lesson.md`, a
lesson written by hand to be bad in specific ways. It costs one API call instead of six, which
matters on a free tier, and because nothing in this system wrote that file, what the evaluator
catches in it is not two models agreeing with each other.

It fails ten of the eleven checks, quoting the fabricated "reduce hallucination rates by 87.3%"
line, the 60-word sentence, and the nine advanced words by name. The eleventh,
`covers_what_and_why`, passes -- the broken lesson genuinely does say what RAG is and why it
matters. That one pass is the useful part of the demo: an evaluator that failed all eleven
would be reacting to the document looking bad rather than reading it.

### Where things are stored

SQLite is the only store. `memory/memory.db` holds the runs, every check result, and the
learned rules; the files in `outputs/` are reports rendered from it, not a second copy of the
data. The rejection log in particular is built by querying `check_results` back out of the
database rather than from anything held in memory during the run.

`python -m scripts.backfill_history` is a one-time migration, not a live code path. An earlier
version of the loop wrote each run to `outputs/run-*.json` before the database existed; those
runs were real, so they were replayed into the schema instead of being paid for again. Nothing
writes those JSON files any more, so running it now finds nothing new to import.

## What each piece does

| File | Role |
|---|---|
| `src/rubric.py` | The checks, as data. Audience definition lives here too |
| `src/schemas.py` | Pydantic models the evaluator must fill in |
| `src/generator.py` | Gemini. Writes and rewrites the lesson |
| `src/evaluator.py` | Groq. Returns one pass/fail verdict per check |
| `src/graph.py` | The LangGraph state machine and the retry edge |
| `src/memory.py` | SQLite: runs, check results, learned rules |
| `src/evolution.py` | Turns repeated failures into standing instructions |
| `src/report.py` | Builds the rejection log from database rows |

## The rubric

Eleven checks, in `src/rubric.py`. Each one is a `name`, a `question` put to the evaluator, and
a `fail_hint` fed back to the writer when it fails.

```
factually_accurate        no unsourced certainty, no absolute guarantees
no_invented_specifics     no fabricated statistics, dates or citations
defines_before_use        every term explained where it first appears
short_sentences           fails at three or more sentences over ~25 words
plain_vocabulary          no leverage, utilize, robust, comprehensive, crucial
everyday_analogy          at least one analogy this reader actually knows
concrete_example_present  a real question and real text, not "imagine a chatbot"
example_walked_through    the example carried to a final answer, written out
covers_what_and_why       both, explicitly, not implied
explains_how_in_steps     separated steps, not one dense paragraph
logical_flow              nothing depends on an idea explained later
```

Every check is calibrated to one reader, described in full in `AUDIENCE`: a 12th-grade
graduate in India from a non-English-medium school, limited English vocabulary, no technical
background. That specificity is what makes the checks decidable. "Is this beginner-friendly"
is a matter of taste; "would this reader stop at this sentence" is a question with an answer.

## Design decisions and trade-offs

**Two providers, not one.** The generator is Gemini, the evaluator is an OpenAI model served
by Groq. A model reviewing its own output rates it generously, and it is blind in the same
places twice: the sentence it thought was clear when writing is the sentence it thinks is
clear when reviewing. Two labs' weights do not share the same blind spots. The original plan
named Llama 3.3 70B for the evaluator; Groq retired it, so it became `openai/gpt-oss-120b`,
which keeps the property that mattered.

**The generator never sees the rubric.** It gets the audience and the writing task. On a
retry it gets only the specific findings that failed. Handing it all eleven checks up front
would make attempt one pass almost every time, which makes the evaluator decorative and
proves nothing about the loop. The rubric is the bar, not the brief.

**Binary checks, no scores.** A 1-10 score has to be collapsed to a decision anyway, and the
threshold is where the real judgement hides. Worse, scores let a model hedge: 7/10 means
"something is wrong but I would rather not say what". Forcing pass or fail means every
rejection comes with a reason and a quote, and those are the only things a rewrite can act
on. It also makes the failure history countable, which is what the evolution query needs.

**Evidence is mandatory, and it comes before the verdict.** Every verdict must quote the
exact line from the lesson that triggered it. Field order in `CheckVerdict` is
`reason, evidence, passed`, and it is load-bearing: under constrained JSON decoding the model
emits keys in schema order, so it has to write its argument before it commits to a judgement.
Put `passed` first and it decides first and reasons backwards.

**SQLite, not a vector store.** The question this system asks its memory is "which check has
failed at least three times", which is `GROUP BY` and `COUNT`. There is no similarity search
anywhere in that, and no embeddings to store. A vector database would be a heavier dependency
answering a question nobody asked.

**SQL decides when to learn, the model decides what to write.** `evolution.py` finds checks
that have failed three or more times with one query. Only then is a model asked to write a
rule that would prevent that failure class, and the rule goes into `learned_rules` and gets
injected into every later generation. Asking a model "do you think you have been failing this
a lot" would make the trigger as unreliable as the thing it is meant to correct. The threshold
is three because two is noise: a bad day, or one genuinely hard topic.

**The retry cap is 2.** Three attempts total. In practice the first rewrite fixes most of what
was raised and the second cleans up the rest; past that the model starts trading one failure
for another rather than converging. The cap is enforced in exactly one place,
`route_after_eval`, so there is one answer to how this loop terminates.

**The rejection log is read back out of the database.** It could have been built from
variables held during the run, which would have been simpler. But a log assembled from local
state is a story the program tells about itself, and it keeps sounding convincing when the
run went wrong. Reading `check_results` back means the log can only report what was recorded.

## Known limitations

- **Nine of the eleven checks are one model's opinion.** It is a good one, and requiring
  evidence keeps it honest, but it is not ground truth. The other two -- `short_sentences` and
  `plain_vocabulary` -- are computed in Python instead, because they have exactly one correct
  answer. The rest cannot be settled that way and stay judged.
- **The rubric was tuned against a small number of runs.** The checks were sharpened twice
  after reading real output, both times to close a gap that was genuinely there. Two runs is
  still two runs.
- **Checks can contradict each other.** They did: `concrete_example_present` demands invented
  example details while `no_invented_specifics` punished them, and a run was flagged for
  review purely because a lesson said "Let's imagine you have a manual for a TechPro Smart
  Speaker". Fixed by exempting clearly hypothetical examples, but nothing stops the next pair
  of checks from colliding, and nothing in the system detects it.
- **Self-evolution is slow to show up.** A rule needs three failures of the same check across
  all history before it is written. That is the right threshold for signal, but it means an
  empty database shows no learned rules until the history has built up. `memory/memory.db` is
  committed with the real history behind it -- nine runs, three learned rules -- so those rules
  are readable without paying for nine runs first. Delete the file and the system starts from
  nothing again.
- **A learned rule is never removed.** Rules have an `active` column and nothing ever sets it
  to 0. A rule that stops helping, or that fights another rule, will sit in the prompt
  indefinitely.
- **The evidence field is not always a clean quote.** The schema asks for the exact line and
  mostly gets it, but the evaluator sometimes appends its own aside -- one verdict quoted a
  passage and added "(no concrete chunk or answer shown)". Harmless here, and still grounded
  in the real text, but it means evidence cannot be used for an exact string match against the
  lesson. One recorded reason also contains a garbled token the model invented mid-sentence.
- **No tests.** The system is verified by running it and reading the output, which is how the
  three real bugs so far were found, but that is not the same as a suite.
- **Cost and rate limits shape the design.** All eleven checks go in one call because eleven
  calls per attempt does not survive a free tier. That is a real trade-off: per-check calls
  would probably judge slightly more carefully.

## Things that broke, and what changed

Worth recording because each one is a design decision that came from evidence rather than
from planning.

**The evaluator answered 3 of 11 checks.** Valid JSON, eight checks silently missing. It hit
the token ceiling; `max_tokens` had been left unset and the reasoning tokens ate the budget.
`_normalise` in `src/evaluator.py` raises on a missing check rather than letting it default to
a pass, which is why this surfaced as a crash instead of a suspiciously good score. Fixed with
an explicit `max_tokens` and `reasoning_effort="low"`.

**A generated lesson was thrown away by a crash.** Artifacts were written at the end of the
run, so a failure in the evaluator discarded a lesson that had already been paid for. They are
now written after every step.

**Every evaluator call failed with 413 once the graph was wired up.** Groq counts the
`max_tokens` reservation against the free tier's per-minute token budget, not just the tokens
actually generated. Reserving 8192 for an eleven-verdict response asked for 10.5k tokens
against an 8k ceiling, so the request could never succeed -- and because Groq reports it as
`rate_limit_exceeded`, the retry logic dutifully waited a minute and tried the identical
too-large request again. `max_tokens` came down to 4000, and oversized requests are now
raised immediately with an explanation instead of being retried.

**Runs died on a hyphen.** Models write real typography, and a default Windows console is
cp1252, so the first non-breaking hyphen raised `UnicodeEncodeError` and killed a run after
its API calls were spent. `main.py` reconfigures stdout to UTF-8 before anything else.

**Two checks were moved out of the model's hands, and the reason turned out to be the
opposite of what it looked like.** A hand-written script measuring sentence lengths said a
lesson had four sentences over the limit on a check that fails at three, while the evaluator
had passed it -- which looked like the evaluator rubber-stamping. It was not. The measuring
script was gluing bold sub-headings onto the paragraph below them, because a heading has no
full stop, and inventing 37-word sentences that did not exist. Measured per line, the lesson
had two. The evaluator had been right. `short_sentences` and `plain_vocabulary` moved into
Python anyway, on the narrower argument that a question with one correct answer should not be
decided by sampling, and they now fail `demo/broken_lesson.md` at five of seventeen sentences
with nine banned words.

**The first draft passed all eleven checks.** Two of those passes did not survive reading the
lesson: the worked example was introduced and never carried to an answer, and the lesson
claimed RAG output "will not make up facts", which is the same overclaim that was planted in
`demo/broken_lesson.md` on purpose. Neither was a new check -- both questions were sharpened
so the bar stopped being ambiguous. Two other checks were verified against ground truth
computed in plain Python first, to confirm the evaluator was being accurate rather than
generous, and it was.
