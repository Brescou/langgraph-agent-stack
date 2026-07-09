"""Pydantic schemas for the custom pack example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EchoInput(BaseModel):
    """Tutorial pack input: text to transform."""

    text: str = Field(..., min_length=1, max_length=4000, description="Source text")


class EchoOutput(BaseModel):
    """Tutorial pack output."""

    original: str
    echoed: str
    word_count: int = Field(ge=0)
