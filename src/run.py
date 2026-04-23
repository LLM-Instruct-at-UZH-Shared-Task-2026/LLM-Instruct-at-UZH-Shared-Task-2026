from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .config import Cfg
from .data import load_docs
from .pipeline.repair import get_repair_stats, reset_repair_stats
from .utils import ensure_dir, write_json
from .validate import validate_docs


def _load_checkpoint(ckpt_path: Path) -> dict:
    """Load already-processed docs from a JSONL checkpoint file."""
    done: dict[str, dict] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    if d.get("TEXT_ID"):
                        done[d["TEXT_ID"]] = d
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--split", type=str, default="test", help="train|test")
    ap.add_argument("--limit", type=int, default=0, help="limit docs (0=all)")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint.jsonl if present")
    ap.add_argument("--run-id", default=None,
                    help="Run ID for output directory (default: auto timestamp)")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = Cfg.load(args.config)
    base_out_dir = ensure_dir(cfg.get("output", "out_dir", default="outputs"))
    sub_name = cfg.get("output", "submission_name", default="submission.json")
    judge_name = cfg.get("output", "judge_report_name", default="judge_report.md")

    trace_enabled = bool(cfg.get("logging", "enabled", default=False))
    trace_jsonl_name = cfg.get("logging", "trace_jsonl_name", default="trace_events.jsonl")
    trace_summary_name = cfg.get("logging", "trace_summary_name", default="trace_summary.json")
    repair_stats_name = cfg.get("logging", "repair_stats_name", default="repair_stats.json")

    # Output directory for this run
    out_dir = ensure_dir(str(Path(base_out_dir) / "runs" / run_id))
    print(f"[run] Run ID: {run_id}  →  {out_dir}")
    ckpt_path = Path(out_dir) / "checkpoint.jsonl"

    # ── Data ──────────────────────────────────────────────────────────────────
    data = load_docs(cfg)
    if args.split not in data:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(data.keys())}")
    docs = data[args.split]
    if args.limit and args.limit > 0:
        docs = docs[:args.limit]

    # ── Resume: skip already-processed docs ──────────────────────────────────
    done_map: dict = {}
    if args.resume or ckpt_path.exists():
        done_map = _load_checkpoint(ckpt_path)
        if done_map:
            before = len(docs)
            docs = [d for d in docs if d.get("TEXT_ID") not in done_map]
            print(f"[resume] Skipping {before - len(docs)} already-done docs; {len(docs)} remaining.")

    # ── Models ────────────────────────────────────────────────────────────────
    generator = cfg.get("models", "generator_name")
    embed_name = cfg.get("models", "embedding_name")
    device = cfg.get("models", "device", default="auto")
    load_in_4bit = bool(cfg.get("models", "load_in_4bit", default=True))
    enable_thinking = bool(cfg.get("models", "enable_thinking", default=False))
    _tb = cfg.get("models", "thinking_budget", default=None)
    thinking_budget = int(_tb) if _tb is not None else None
    _bc = cfg.get("models", "batch_chunk_size", default=16)
    batch_chunk_size = int(_bc)

    task1_mode = cfg.get("task1", "mode", default="llm")
    task1_type_mode = cfg.get("task1", "type_mode", default=task1_mode)
    task1_tag_mode = cfg.get("task1", "tag_mode", default=task1_mode)
    task2_mode = cfg.get("task2", "mode", default="heuristic")
    need_llm = (
        (task1_type_mode != "heuristic")
        or (task1_tag_mode != "heuristic")
        or (task2_mode != "heuristic")
    )

    llm = None
    if need_llm and generator and docs:
        from .llm import LocalLLM
        llm = LocalLLM(
            generator,
            device=device,
            load_in_4bit=load_in_4bit,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            batch_chunk_size=batch_chunk_size,
        )

    embedder = None
    if embed_name:
        from .embed import Embedder
        dev = "cuda" if device in ("auto", "cuda") else "cpu"
        embedder = Embedder(embed_name, device=dev)

    # ── RAG index (optional) ──────────────────────────────────────────────────
    rag_index = None
    use_rag = bool(cfg.get("rag", "enabled", default=False))
    if use_rag and embedder:
        index_path = cfg.get("rag", "index_path", default="outputs/rag_index.faiss")
        meta_path  = cfg.get("rag", "meta_path",  default="outputs/rag_meta.json")
        from pathlib import Path as _Path
        if _Path(index_path).exists() and _Path(meta_path).exists():
            try:
                from .rag import RagIndex
                print("[RAG] Loading pre-built FAISS index …")
                rag_index = RagIndex.load(index_path, meta_path, embedder)
                print(f"[RAG] Loaded {len(rag_index)} entries from {index_path}")
            except (ImportError, FileNotFoundError, ValueError) as exc:
                print(f"[RAG] Could not load pre-built index: {exc}")
                print("[RAG] Falling back to in-memory retrieval from train data.")
        if rag_index is None and "train" in data:
            from .embed import RAGIndex
            print("[RAG] Building in-memory index from training docs …")
            rag_index = RAGIndex(embedder)
            rag_index.build(data["train"], lang=cfg.get("task1", "language", default="en"))
            print(f"[RAG] In-memory index built: {len(data['train'])} docs.")
        elif rag_index is None:
            print("[RAG] enabled=true but no pre-built index and no train data found; RAG skipped.")

    # ── Checkpoint callback (appends one doc at a time) ───────────────────────
    ckpt_fh = None
    def _on_doc_done(doc_result: dict) -> None:
        nonlocal ckpt_fh
        if ckpt_fh is None:
            ckpt_fh = open(ckpt_path, "a", encoding="utf-8")
        ckpt_fh.write(json.dumps(doc_result, ensure_ascii=False) + "\n")
        ckpt_fh.flush()

    # ── Pipeline ─────────────────────────────────────────────────────────────
    from .pipeline.pipeline import run_pipeline
    reset_repair_stats()
    new_pred = run_pipeline(
        docs, cfg, llm=llm, embedder=embedder, rag_index=rag_index,
        on_doc_done=_on_doc_done,
        trace_enabled=trace_enabled,
        trace_path=str(Path(out_dir) / trace_jsonl_name),
        trace_summary_path=str(Path(out_dir) / trace_summary_name),
    )
    if ckpt_fh:
        ckpt_fh.close()

    # Merge checkpoint (previously done) + new results, preserving original order
    all_ids_order = [d.get("TEXT_ID") for d in data[args.split]]
    new_map = {d["TEXT_ID"]: d for d in new_pred}
    merged_map = {**done_map, **new_map}
    pred = [merged_map[tid] for tid in all_ids_order if tid in merged_map]

    # ── Validate ─────────────────────────────────────────────────────────────
    ok, errs = validate_docs(pred)
    if not ok:
        print(f"[WARNING] Schema validation FAILED ({len(errs)} errors). First error:\n", errs[0])
    else:
        print("[OK] Schema validation passed.")

    # ── Write submission ──────────────────────────────────────────────────────
    sub_path = Path(out_dir) / sub_name
    write_json(sub_path, pred)
    print("Wrote submission:", sub_path)

    if trace_enabled:
        repair_stats_path = Path(out_dir) / repair_stats_name
        write_json(repair_stats_path, get_repair_stats())
        print("Wrote repair stats:", repair_stats_path)

    # Clean up checkpoint once submission is written successfully
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("[checkpoint] Removed checkpoint file.")

    # ── Optional Gemini judge ─────────────────────────────────────────────────
    # use_gemini = bool(cfg.get("evaluation", "use_gemini", default=False))
    # if use_gemini:
    #     load_dotenv()
    #     env_key = cfg.get("evaluation", "gemini_model_env", default="GEMINI_MODEL")
    #     model_name = os.getenv(env_key, "gemini-2.0-flash")
    #     per_doc = int(cfg.get("evaluation", "per_doc_sample", default=8))
    #     from .eval_gemini import judge_docs
    #     reports = judge_docs(pred, model_name=model_name, per_doc_sample=per_doc)
    #     rep_path = Path(out_dir) / judge_name
    #     rep_path.write_text(
    #         "\n\n".join(
    #             f"## {r['TEXT_ID']}\n\n```\n{r['raw']}\n```" for r in reports
    #         ),
    #         encoding="utf-8",
    #     )
    #     print("Wrote judge report:", rep_path)
    print("[run] Skipped judge. Done.")


if __name__ == "__main__":
    main()
