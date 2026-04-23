#!/usr/bin/env python3
"""Validate the local environment for the ArgMining 2026 pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Check local setup for this repository.")
    ap.add_argument("--config", default="config.yaml", help="Path to config file relative to repo root.")
    ap.add_argument(
        "--check-judge",
        action="store_true",
        help="Also validate optional Gemini judge dependencies and environment variables.",
    )
    args = ap.parse_args()

    try:
        import yaml
    except ImportError:
        _print("FAIL", "Missing dependency 'PyYAML'. Run: pip install -r requirements.txt")
        return 1

    if _has_module("dotenv"):
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")

    cfg_path = repo_root / args.config
    issues = 0
    warnings = 0

    if not cfg_path.exists():
        _print("FAIL", f"Config file not found: {cfg_path}")
        return 1
    _print("OK", f"Config file found: {cfg_path.relative_to(repo_root)}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        _print("FAIL", f"Config file is not a YAML mapping: {cfg_path.relative_to(repo_root)}")
        return 1

    for module_name, package_name in [
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
        ("dotenv", "python-dotenv"),
    ]:
        if _has_module(module_name):
            _print("OK", f"Dependency available: {package_name}")
        else:
            _print("FAIL", f"Missing dependency '{package_name}'. Run: pip install -r requirements.txt")
            issues += 1

    data_cfg = cfg.get("data", {})
    hf_dataset = data_cfg.get("hf_dataset")
    train_dir = data_cfg.get("local_train_dir")
    test_dir = data_cfg.get("local_test_dir")
    tags_csv = data_cfg.get("tags_csv_path")

    def _check_path(label: str, rel_path: str | None, required: bool) -> None:
        nonlocal issues, warnings
        if not rel_path:
            if required:
                _print("FAIL", f"{label} is not configured.")
                issues += 1
            return
        path = repo_root / rel_path
        if path.exists():
            _print("OK", f"{label}: {rel_path}")
        else:
            level = "FAIL" if required else "WARN"
            _print(level, f"{label} not found: {rel_path}")
            if required:
                issues += 1
            else:
                warnings += 1

    local_data_required = not hf_dataset
    if hf_dataset:
        _print("OK", f"Hugging Face dataset configured: {hf_dataset}")
    else:
        _print("OK", "No Hugging Face dataset configured; local dataset paths will be used.")

    _check_path("Training data directory", train_dir, required=local_data_required)
    _check_path("Test data directory", test_dir, required=local_data_required)
    _check_path("Tag taxonomy CSV", tags_csv, required=True)

    rag_cfg = cfg.get("rag", {})
    if rag_cfg.get("enabled", False):
        index_path = rag_cfg.get("index_path")
        meta_path = rag_cfg.get("meta_path")
        index_exists = bool(index_path) and (repo_root / index_path).exists()
        meta_exists = bool(meta_path) and (repo_root / meta_path).exists()

        if index_exists and meta_exists:
            _print("OK", f"Prebuilt RAG index found: {index_path}")
            if _has_module("faiss"):
                _print("OK", "FAISS runtime available for prebuilt RAG index.")
            else:
                _print(
                    "WARN",
                    "Prebuilt RAG index exists but FAISS is not installed. "
                    "The pipeline can still fall back to in-memory RAG if training data is present.",
                )
                warnings += 1
        elif train_dir and (repo_root / train_dir).exists():
            _print("OK", "RAG can fall back to an in-memory index built from local training data.")
        else:
            _print(
                "WARN",
                "RAG is enabled but neither a prebuilt FAISS index nor local training data is available.",
            )
            warnings += 1

    if args.check_judge:
        if _has_module("google.genai"):
            _print("OK", "Optional judge dependency available: google-genai")
        else:
            _print("FAIL", "Missing optional dependency 'google-genai'. Run: pip install -r requirements.txt")
            issues += 1

        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL")
        if api_key:
            _print("OK", "GOOGLE_API_KEY is set.")
        else:
            _print("WARN", "GOOGLE_API_KEY is not set. The Gemini judge will not run yet.")
            warnings += 1
        if model_name:
            _print("OK", f"GEMINI_MODEL is set: {model_name}")
        else:
            _print("WARN", "GEMINI_MODEL is not set. .env.example uses gemini-2.0-flash.")
            warnings += 1

    print()
    if issues:
        _print("SUMMARY", f"{issues} required issue(s) and {warnings} warning(s) detected.")
        return 1

    _print("SUMMARY", f"Setup looks runnable. {warnings} warning(s) detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
