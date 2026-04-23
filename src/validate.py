from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import ValidationError
from .schema import Doc

def validate_docs(docs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors = []
    for i, d in enumerate(docs):
        try:
            Doc.model_validate(d)
        except ValidationError as e:
            errors.append(f"Doc {i} TEXT_ID={d.get('TEXT_ID')}: {e}")
    return (len(errors) == 0), errors
