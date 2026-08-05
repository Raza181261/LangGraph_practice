from pydantic import BaseModel, Field
from typing import List, Optional

class Reflection(BaseModel):
    missing: str = Field(..., description="Critique what is missing.")
    superfluous: str = Field(..., description="Critique what is superfluous.")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    
    answer: str = Field(..., description="250 words detailed answer to the question.")
    search_queries: List[str] = Field(..., description="1-3 search queries for researching improvements to address the critique of your current answer.")
    reflection: Reflection = Field(..., description=" your reflection on the initial answer.")

class RevisedAnswer(AnswerQuestion):
    """Revise your answer to your question."""

    references: List[str] = Field(description="Citations motivating your updated answer")