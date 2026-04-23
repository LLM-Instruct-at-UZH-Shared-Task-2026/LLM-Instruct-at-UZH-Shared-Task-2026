from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import csv

# CSV format: Dimensions;Categories;CODE
# We embed enriched text for semantic similarity and use CODE as the tag label.

def load_tag_metadata(csv_path: str | None) -> List[Dict[str, str]]:
    """Load tag metadata from the education_dimensions_updated.csv file.

    The CSV uses semicolon as delimiter with columns: Dimensions;Categories;CODE.
    NA/empty CODE rows are skipped.
    Returns a list of dicts with keys: _tag (CODE), _dim, _cat, _text (enriched).
    """
    if not csv_path:
        return []
    p = Path(csv_path)
    if not p.exists():
        # Try relative fallback
        fallback = Path("dataset/education_dimensions_updated.csv")
        if fallback.exists():
            p = fallback
        else:
            return []
    rows: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            rr = {k.strip(): (v or "").strip() for k, v in r.items()}
            # Primary columns: Dimensions, Categories, CODE
            code = rr.get("CODE", "")
            if not code or code.upper() == "NA":
                continue
            dim = rr.get("Dimensions", "")
            cat = rr.get("Categories", "")
            rr["_tag"] = code
            rr["_dim"] = dim
            rr["_cat"] = cat
            # Enriched semantic text for embedding: human-readable phrase
            parts = [x for x in [dim, cat, code] if x]
            rr["_text"] = " | ".join(parts)
            rows.append(rr)
    return rows


def tag_display(row: Dict[str, str]) -> str:
    """Return a human-readable tag string combining dim + category."""
    return f"{row['_dim']}: {row['_cat']}" if row.get("_dim") and row.get("_cat") else row["_tag"]
