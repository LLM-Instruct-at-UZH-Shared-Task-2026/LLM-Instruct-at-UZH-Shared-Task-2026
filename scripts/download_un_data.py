#!/usr/bin/env python3
"""Download UN / UNESCO education resolutions from the UN Digital Library.

These serve as external training data for the RAG index (Phase 4 of CDA).

The UN Digital Library provides a public SRU/MARCXML API.
This script queries for education-related resolutions, fetches the full text,
parses paragraph items, and saves JSONL in the same format as train-data.

Usage::

    python -m scripts.download_un_data \\
        --out-dir     dataset/external-data \\
        --max-records 500 \\
        --lang-pair   fr,en \\
        --queries     "education UNESCO resolution" "education rights"

Requirements:
    pip install requests lxml

Note: The UN Digital Library is a public dataset.  No API key is needed.
      Rate-limit responsibly: the script includes a 0.5 s delay between requests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

try:
    import requests
    from lxml import etree
except ImportError:
    print("Missing dependencies.  Run:  pip install requests lxml")
    sys.exit(1)

# UN Digital Library public SRU endpoint
_SRU_BASE = "https://digitallibrary.un.org/search"

# UNESDOC (UNESCO Digital Library) OPDS/OAI endpoint
_UNESDOC_OAI = "https://unesdoc.unesco.org/ark:/48223/pf0000386010"

# Namespaces used in the MARC XML responses
_MARC_NS = "http://www.loc.gov/MARC21/slim"


def _sru_search(query: str, page: int = 1, page_size: int = 25) -> bytes:
    """Query the UN Digital Library and return raw XML bytes."""
    params = {
        "p":  query,
        "of": "hx",          # MARC XML output
        "rg": page_size,
        "jrec": (page - 1) * page_size + 1,
        "sf": "",
        "so": "d",
        "rm": "",
        "c": "",
        "f": "",
        "action_search": "Search",
    }
    url = _SRU_BASE + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def _extract_text_from_marc(xml_bytes: bytes) -> List[dict]:
    """Parse MARC XML and extract {title, paragraphs} items.

    MARC field 520 = summary/abstract (paragraph text).
    MARC field 245 = title.
    MARC field 041 = language code.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return []

    ns = {"m": _MARC_NS}
    records = root.findall(".//m:record", ns) or root.findall(".//record")
    if not records and root.tag.endswith("record"):
        records = [root]

    results = []
    for rec in records:
        # Extract title
        title = ""
        for tag in ["245", "240", "242"]:
            subfields = rec.findall(f".//m:datafield[@tag='{tag}']/m:subfield", ns)
            if not subfields:
                subfields = rec.findall(f".//datafield[@tag='{tag}']/subfield")
            t = " ".join(sf.text or "" for sf in subfields).strip()
            if t:
                title = t
                break

        # Extract abstract / summary paragraphs
        abstracts_fr = []
        abstracts_en = []

        for field in rec.findall(".//m:datafield[@tag='520']", ns) or \
                      rec.findall(".//datafield[@tag='520']"):
            ind1 = field.get("ind1", " ")
            lang_sf = field.find("m:subfield[@code='9']", ns) or \
                      field.find("subfield[@code='9']")
            lang = (lang_sf.text or "").lower().strip() if lang_sf is not None else ""

            text_sf = field.find("m:subfield[@code='a']", ns) or \
                      field.find("subfield[@code='a']")
            text = (text_sf.text or "").strip() if text_sf is not None else ""

            if not text:
                continue

            if lang in ("fr", "fre"):
                abstracts_fr.append(text)
            else:
                abstracts_en.append(text)

        if not (abstracts_fr or abstracts_en):
            continue

        paragraphs = []
        max_len = max(len(abstracts_fr), len(abstracts_en))
        for i in range(max_len):
            text_fr = abstracts_fr[i] if i < len(abstracts_fr) else ""
            text_en = abstracts_en[i] if i < len(abstracts_en) else ""
            # If only one language, use it for both (model will handle it)
            if not text_fr:
                text_fr = text_en
            if not text_en:
                text_en = text_fr
            paragraphs.append({
                "type": "paragraph",
                "level": None,
                "text_fr": text_fr,
                "text_en": text_en,
            })

        if paragraphs:
            results.append({"title": title, "paragraphs": paragraphs})

    return results


def _clean_filename(title: str, idx: int) -> str:
    """Convert title to a safe filename stub."""
    clean = re.sub(r"[^A-Za-z0-9_\-]", "_", title)[:60].strip("_")
    return f"UN_ext_{idx:05d}_{clean or 'record'}"


def main():
    ap = argparse.ArgumentParser(description="Download UN education resolutions.")
    ap.add_argument("--out-dir",     default="dataset/external-data",
                    help="Directory to save downloaded files.")
    ap.add_argument("--max-records", type=int, default=500,
                    help="Maximum total records to download.")
    ap.add_argument("--page-size",   type=int, default=25,
                    help="Records per API page.")
    ap.add_argument("--delay",       type=float, default=0.5,
                    help="Seconds to wait between API requests.")
    ap.add_argument("--queries",     nargs="+", default=[
        'education UNESCO resolution',
        '"right to education" resolution',
        '"education for all" resolution',
        'inclusive education resolution',
    ], help="Search queries to run against the UN Digital Library.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[UN-DL] Saving to {out_dir}  (max {args.max_records} records)")

    global_idx = 0
    total_paras = 0

    for query in args.queries:
        print(f"\n[UN-DL] Query: {query!r}")
        page = 1
        fetched = 0

        while fetched < args.max_records:
            remaining = args.max_records - fetched
            pg_size = min(args.page_size, remaining)
            try:
                xml_bytes = _sru_search(query, page=page, page_size=pg_size)
            except Exception as exc:
                print(f"  [warn] Request failed (page {page}): {exc}")
                break

            docs = _extract_text_from_marc(xml_bytes)
            if not docs:
                print(f"  [info] No more results at page {page}.")
                break

            for doc in docs:
                fname = _clean_filename(doc["title"], global_idx)
                out_path = out_dir / f"{fname}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(doc["paragraphs"], f, ensure_ascii=False, indent=2)
                total_paras += len(doc["paragraphs"])
                global_idx += 1
                fetched += 1

            print(f"  page {page}: {len(docs)} docs  "
                  f"(total fetched={fetched}, paras={total_paras})")
            page += 1
            time.sleep(args.delay)

            if len(docs) < pg_size:
                break   # reached end of results

    print(f"\n[UN-DL] Done.  {global_idx} files written, "
          f"{total_paras} paragraphs total → {out_dir}")


if __name__ == "__main__":
    main()
