from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    decision: str = Field(..., description="out_of_scope|direct_answer|retrieve")
    reason: str = Field("", description="Reason for the decision")


class GradeResult(BaseModel):
    relevant: bool
    reason: str = Field("", description="Reason for relevance decision")


class RewriteResult(BaseModel):
    rewritten_query: str
