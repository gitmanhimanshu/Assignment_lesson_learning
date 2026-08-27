import os

from dotenv import load_dotenv

load_dotenv()

GENERATOR_MODEL = "gemini-2.5-flash"
# Deliberately a different provider and a different lab's weights from the generator. A model
# grading its own output rates it generously; this is the cheapest defence against that.
EVALUATOR_MODEL = "openai/gpt-oss-120b"

# LangChain defaults to method="function_calling", which gpt-oss-120b fails outright -- it
# answers in prose and never emits the tool call, and Groq returns 400 tool_use_failed.
# json_schema constrains decoding instead, and is also what makes field order in the schema
# control the model's order of reasoning.
STRUCTURED_OUTPUT_METHOD = "json_schema"

# Three attempts total. Two is enough to see whether feedback actually helps, and past the
# third the model tends to thrash between failures rather than converge.
MAX_RETRIES = 2
MAX_ATTEMPTS = MAX_RETRIES + 1

# High enough that a regenerated draft is genuinely different from the rejected one. At 0 the
# model returns near-identical prose and the retry is wasted.
GENERATOR_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.0

# Groq counts the max_tokens reservation against the free tier's tokens-per-minute limit, not
# just the tokens actually produced. At 8192 a ~900 word lesson asked for 10.5k against an 8k
# ceiling and was rejected with 413 every single time, which no amount of retrying fixes.
# Eleven verdicts come to roughly 1.5k tokens, so this leaves a wide margin and still fits.
EVALUATOR_MAX_TOKENS = 4000

# The verdicts come out of the rubric questions, which are already precise. Spending hidden
# reasoning tokens on top of that mostly burns budget and truncates the answer.
EVALUATOR_REASONING_EFFORT = "low"


def require_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
