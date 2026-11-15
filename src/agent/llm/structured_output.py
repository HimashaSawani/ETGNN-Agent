from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


def extract_json_str(raw_text: str) -> str:
    """Extract raw JSON text from markdown code blocks or outer brackets."""
    text = raw_text.strip()

    # Look for ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        # Look for outer { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    # Clean potential trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def parse_and_validate_json(
    raw_text: str,
    schema_cls: Optional[Type[T]] = None
) -> Dict[str, Any] | T:
    """
    Parses JSON string and optionally validates it against a Pydantic model.
    """
    cleaned = extract_json_str(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"Failed to parse LLM response as JSON: {e}\nRaw input:\n{raw_text[:500]}") from e

    if not isinstance(data, dict):
        raise StructuredOutputError(f"Expected JSON object (dict), got {type(data).__name__}")

    if schema_cls is not None:
        try:
            return schema_cls.model_validate(data)
        except ValidationError as e:
            raise StructuredOutputError(f"JSON does not adhere to {schema_cls.__name__}: {e}") from e

    return data

