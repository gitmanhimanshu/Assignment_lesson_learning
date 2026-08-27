"""Pydantic models for the evaluator's output.

Field order in CheckVerdict is load-bearing, not cosmetic. The evaluator runs under
constrained JSON decoding, which emits keys in schema order, so putting reason and evidence
ahead of passed forces the model to write its argument before it commits to a verdict. Put
passed first and it decides first and rationalises afterwards, which is how you get a
rubber-stamp evaluator.
"""

from pydantic import BaseModel, Field, field_validator


class CheckVerdict(BaseModel):

    check_name: str = Field(description="the exact name of the rubric check being answered")
    reason: str = Field(
        description="what is wrong (or right) with the lesson on this check, in plain language"
    )
    evidence: str = Field(
        description=(
            "the exact sentence or phrase copied word for word from the lesson that made you "
            "decide this. Never paraphrase and never describe the lesson here."
        )
    )
    passed: bool = Field(description="true if the lesson satisfies this check, false if not")

    @field_validator("evidence", "reason")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # A verdict without evidence is an opinion. Rejecting it here means a lazy evaluator
        # response fails loudly instead of quietly becoming a pass.
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class Evaluation(BaseModel):
    verdicts: list[CheckVerdict]

    @property
    def failures(self) -> list[CheckVerdict]:
        return [v for v in self.verdicts if not v.passed]

    @property
    def all_passed(self) -> bool:
        return not self.failures
