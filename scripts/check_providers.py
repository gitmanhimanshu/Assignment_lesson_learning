"""Connectivity smoke test. Run this once after filling in .env, before building anything else.

Beyond checking that both keys work, this verifies the evaluator can return a parseable
Pydantic object. That is the riskiest assumption in the whole design -- the loop is built on
structured verdicts, and Groq's reasoning models happily answer in prose and never call the
tool. Better to find that out here than three phases in.
"""

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

GENERATOR_MODEL = "gemini-2.5-flash"
EVALUATOR_MODEL = "openai/gpt-oss-120b"

# LangChain defaults to method="function_calling", which gpt-oss-120b fails outright:
# it writes its answer as prose and never emits the tool call, so Groq returns
# 400 tool_use_failed. json_schema constrains decoding instead and works.
STRUCTURED_OUTPUT_METHOD = "json_schema"

PING = "Reply with exactly one word: pong"

JARGON_SAMPLE = "RAG uses a vector DB to do ANN lookups over embeddings."
JARGON_QUESTION = (
    f'Text: "{JARGON_SAMPLE}"\n'
    "Question: does this text avoid unexplained jargon for a total beginner?"
)


class _Verdict(BaseModel):
    """Throwaway stand-in for the real evaluator schema, just to prove decoding works."""

    reason: str = Field(description="what is wrong, in plain language")
    evidence: str = Field(description="the exact phrase from the text that triggered this")
    passed: bool


def check_gemini(api_key: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, google_api_key=api_key, temperature=0)
    return llm.invoke(PING).content.strip()


def check_groq(api_key: str) -> str:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=EVALUATOR_MODEL, api_key=api_key, temperature=0)
    return llm.invoke(PING).content.strip()


def check_groq_structured(api_key: str) -> str:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=EVALUATOR_MODEL, api_key=api_key, temperature=0).with_structured_output(
        _Verdict, method=STRUCTURED_OUTPUT_METHOD
    )
    verdict = llm.invoke(JARGON_QUESTION)

    # The sample is deliberately jargon-heavy. An evaluator that passes it is not grounding
    # its verdict in the text, and the whole rubric would be theatre.
    if verdict.passed:
        raise RuntimeError(
            f"evaluator passed an obviously jargon-heavy sentence: {verdict.reason!r}"
        )
    return f"passed={verdict.passed} evidence={verdict.evidence[:50]!r}"


def main() -> int:
    load_dotenv()

    providers = [
        ("Gemini generate", GENERATOR_MODEL, "GOOGLE_API_KEY", check_gemini),
        ("Groq generate", EVALUATOR_MODEL, "GROQ_API_KEY", check_groq),
        ("Groq structured", EVALUATOR_MODEL, "GROQ_API_KEY", check_groq_structured),
    ]

    failures = 0
    for label, model, env_var, call in providers:
        key = os.getenv(env_var)
        if not key:
            print(f"FAIL  {label:17} {env_var} is not set (copy .env.example to .env)")
            failures += 1
            continue

        try:
            result = call(key)
        # Each SDK raises its own auth / rate-limit / not-found types. For a smoke test the
        # only thing that matters is that the call did not come back, and the exception
        # message is what tells us which of those it was.
        except Exception as exc:
            print(f"FAIL  {label:17} {model}")
            print(f"      {type(exc).__name__}: {exc}")
            failures += 1
            continue

        print(f"OK    {label:17} {model} -> {result}")

    if failures:
        print(f"\n{failures} check(s) failed. Fix before continuing.")
        return 1

    print("\nBoth providers reachable, evaluator returns structured verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
