from __future__ import annotations
from typing import Any, Dict


def build_doc_skeleton(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Build a clean output skeleton from a normalised input document.

    Input docs are always normalised by data.py to use body.paras + matched_paras.
    Output matches the submission schema exactly.
    """
    text_id = raw_doc.get("TEXT_ID") or raw_doc.get("text_id") or raw_doc.get("id")
    if not text_id:
        raise ValueError("Doc missing TEXT_ID")
    title = raw_doc.get("TITLE") or raw_doc.get("title")
    rec = raw_doc.get("RECOMMENDATION") or raw_doc.get("recommendation")

    body = raw_doc.get("body") or {}
    # After normalisation, always body.paras
    paras_raw = body.get("paras") or body.get("paragraphs") or []

    meta_src = (raw_doc.get("METADATA") or {}).get("structure") or {}

    out: Dict[str, Any] = {
        "TEXT_ID": text_id,
        "RECOMMENDATION": rec,
        "TITLE": title,
        "METADATA": {
            "structure": {
                "doc_title": meta_src.get("doc_title") or title,
                "nb_paras": len(paras_raw),
                "preambular_para": [],
                "operative_para": [],
                "think": "",
            }
        },
        "body": {"paragraphs": []},
    }

    for p in paras_raw:
        out["body"]["paragraphs"].append({
            "para_number": int(p.get("para_number", len(out["body"]["paragraphs"]) + 1)),
            "para": p.get("para"),
            "para_en": p.get("para_en"),
            "type": None,
            "tags": [],
            "matched_pars": {},
            "think": "",
        })
    return out
