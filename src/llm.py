"""Model clients and the one piece of error handling that actually earns its place: retrying
free-tier rate limits.
"""

import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src import config

RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "quota", "resource_exhausted")

# Groq reports an oversized request as a rate limit too, because the max_tokens reservation
# counts against the per-minute token budget. It is not transient -- the same request will be
# too large in a minute and in an hour -- so retrying it just burns a minute per attempt.
TOO_LARGE_MARKERS = ("request too large", "reduce your message size", "413")


def generator_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GENERATOR_MODEL,
        google_api_key=config.require_key("GOOGLE_API_KEY"),
        temperature=config.GENERATOR_TEMPERATURE,
    )


def evaluator_llm(schema: type) -> Any:
    llm = ChatGroq(
        model=config.EVALUATOR_MODEL,
        api_key=config.require_key("GROQ_API_KEY"),
        temperature=config.EVALUATOR_TEMPERATURE,
        # Eleven verdicts, each with a reason and a quoted line, is a long structured
        # response. Left unset, the reasoning tokens ate the budget and the model stopped
        # after three verdicts -- valid JSON, silently missing most of the rubric.
        max_tokens=config.EVALUATOR_MAX_TOKENS,
        reasoning_effort=config.EVALUATOR_REASONING_EFFORT,
    )
    return llm.with_structured_output(schema, method=config.STRUCTURED_OUTPUT_METHOD)


def text_of(response: Any) -> str:
    """The text out of a chat response, whatever shape the provider used.

    LangChain's content is a union: Gemini 2.5 returns a plain string, Gemini 3.x returns a
    list of content blocks, and a block is either a dict with a "text" key or already a
    string. Calling .strip() on the list version raises AttributeError, which is a confusing
    way to find out you changed models.
    """
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            # Anything else is a non-text block -- a thought signature, an image part.
            # Skipping it is correct; the caller asked for text.
        return "".join(parts).strip()

    if content is None:
        return ""

    raise TypeError(f"unexpected content type from model: {type(content).__name__}")


def invoke_with_retry(llm: Any, prompt: str, tries: int = 3) -> Any:
    """Retry only on rate limits. Everything else fails immediately and loudly."""
    for attempt in range(1, tries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            # Google and Groq raise unrelated exception hierarchies, and langchain-google-genai
            # has already renamed its error classes once. Matching the message is uglier than
            # catching a type but it does not silently stop working after an SDK bump.
            message = str(exc).lower()
            if any(marker in message for marker in TOO_LARGE_MARKERS):
                raise RuntimeError(
                    "the request exceeded the provider's per-minute token budget. This is a "
                    "size problem, not a timing one -- lower EVALUATOR_MAX_TOKENS or shorten "
                    f"the lesson. Original error: {exc}"
                ) from exc

            is_rate_limit = any(marker in message for marker in RATE_LIMIT_MARKERS)
            if not is_rate_limit or attempt == tries:
                raise

            wait = 20 * attempt
            print(f"      rate limited, waiting {wait}s (attempt {attempt}/{tries})")
            time.sleep(wait)

    raise AssertionError("unreachable")
