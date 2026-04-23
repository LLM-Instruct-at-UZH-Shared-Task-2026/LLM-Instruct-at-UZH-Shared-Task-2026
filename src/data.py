from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset
from .utils import read_json


def _get_paras(body: Dict) -> List[Dict]:
    """Handle both 'paragraphs' and 'paras' key in body."""
    return body.get("paragraphs") or body.get("paras") or []


def _normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw document to a consistent internal format.

    Handles:
    - Test/official JSON format: TEXT_ID + body.paragraphs + matched_pars
    - HF dataset format (pre-normalised)
    The output always uses body.paras + matched_paras (schema fields).
    """
    body_raw = doc.get("body") or {}
    paras_raw = _get_paras(body_raw)
    paras_out = []
    for p in paras_raw:
        # matched_pars (test data) → matched_paras (schema)
        mp = p.get("matched_paras") or p.get("matched_pars") or {}
        paras_out.append({
            "para_number": p.get("para_number", len(paras_out) + 1),
            "para": p.get("para"),
            "para_en": p.get("para_en"),
            "type": p.get("type"),
            "tags": p.get("tags") or [],
            "matched_paras": mp,
            "think": p.get("think", ""),
        })
    doc_out = dict(doc)
    if paras_out:
        doc_out["body"] = {"paras": paras_out}
    return doc_out


def _normalize_flat_doc(para_list: List[Dict[str, Any]], doc_id: str = "") -> Dict[str, Any]:
    """Convert flat train format (list[{type,level,text_fr,text_en}]) to internal schema."""
    paras_out = []
    for i, p in enumerate(para_list):
        paras_out.append({
            "para_number": i + 1,
            "para": p.get("text_fr") or "",
            "para_en": p.get("text_en") or "",
            "type": p.get("type") or "",
            "tags": [],
            "matched_paras": {},
            "think": "",
        })
    return {
        "TEXT_ID": doc_id,
        "body": {"paras": paras_out},
    }


def load_docs(cfg) -> Dict[str, List[Dict[str, Any]]]:
    hf_name = cfg.get("data", "hf_dataset", default=None)
    cache_dir = cfg.get("data", "cache_dir", default=None)
    local_train = cfg.get("data", "local_train_path", default=None)
    local_test = cfg.get("data", "local_test_path", default=None)
    train_split = cfg.get("data", "train_split", default="train")
    test_split = cfg.get("data", "test_split", default="test")
    local_test_dir = cfg.get("data", "local_test_dir", default=None)
    local_train_dir = cfg.get("data", "local_train_dir", default=None)

    out: Dict[str, List[Dict[str, Any]]] = {}

    # Local single-file paths
    if local_train:
        train_path = Path(local_train)
        if train_path.exists():
            raw = read_json(train_path)
            out["train"] = [_normalize_doc(d) for d in (raw if isinstance(raw, list) else [raw])]
        else:
            print(f"[data] local_train_path not found: {train_path}")
    if local_test:
        test_path = Path(local_test)
        if test_path.exists():
            raw = read_json(test_path)
            out["test"] = [_normalize_doc(d) for d in (raw if isinstance(raw, list) else [raw])]
        else:
            print(f"[data] local_test_path not found: {test_path}")

    # Load train docs from a directory of flat JSON files (train format)
    if local_train_dir and "train" not in out:
        import glob, json
        train_dir = Path(local_train_dir)
        if not train_dir.exists():
            print(f"[data] local_train_dir not found: {train_dir}")
            files = []
        else:
            files = sorted(glob.glob(os.path.join(local_train_dir, "*.json")))
            if not files:
                print(f"[data] no JSON files found in local_train_dir: {train_dir}")
        docs = []
        for fp in files:
            raw = json.loads(Path(fp).read_text(encoding="utf-8"))
            doc_id = Path(fp).stem
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text_fr" in raw[0]:
                # Flat train format: list[{type, level, text_fr, text_en}]
                docs.append(_normalize_flat_doc(raw, doc_id=doc_id))
            elif isinstance(raw, list):
                docs.extend([_normalize_doc(d) for d in raw])
            else:
                docs.append(_normalize_doc(raw))
        if docs:
            out["train"] = docs

    # Load test docs from a directory of JSON files
    if local_test_dir and "test" not in out:
        import glob, json
        test_dir = Path(local_test_dir)
        if not test_dir.exists():
            print(f"[data] local_test_dir not found: {test_dir}")
            files = []
        else:
            files = sorted(glob.glob(os.path.join(local_test_dir, "*.json")))
            if not files:
                print(f"[data] no JSON files found in local_test_dir: {test_dir}")
        docs = []
        for fp in files:
            raw = json.loads(Path(fp).read_text(encoding="utf-8"))
            if isinstance(raw, list):
                docs.extend([_normalize_doc(d) for d in raw])
            else:
                docs.append(_normalize_doc(raw))
        if docs:
            out["test"] = docs

    # HuggingFace dataset — only load splits not already covered by local paths
    need_train = "train" not in out
    need_test = "test" not in out
    if hf_name and (need_train or need_test):
        try:
            ds = load_dataset(hf_name, cache_dir=cache_dir)
            if train_split in ds and need_train:
                out["train"] = [_normalize_doc(dict(x)) for x in ds[train_split]]
            if test_split in ds and need_test:
                out["test"] = [_normalize_doc(dict(x)) for x in ds[test_split]]
            out["_available_splits"] = list(ds.keys())
        except Exception as e:
            print(f"[data] HF dataset load failed: {e}")

    if "train" not in out and "test" not in out:
        checked = []
        for label, value in [
            ("data.local_train_path", local_train),
            ("data.local_test_path", local_test),
            ("data.local_train_dir", local_train_dir),
            ("data.local_test_dir", local_test_dir),
            ("data.hf_dataset", hf_name),
        ]:
            if value:
                checked.append(f"{label}={value}")
        raise FileNotFoundError(
            "No data loaded. Checked: "
            + (", ".join(checked) if checked else "no configured data source")
        )
    return out
