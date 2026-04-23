# LLM-Instruct at UZH Shared Task - ArgMining 2026

This repository contains the code used to run the LLM-Instruct system for the ArgMining Workshop 2026 shared task.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
```

## Local Data Layout

This repo intentionally keeps workshop data and run artifacts out of git. By default, `config.yaml` expects:

- `dataset/train-data/`
- `dataset/test-data/`
- `dataset/education_dimensions_updated.csv`

If your files live elsewhere, update the paths in `config.yaml`.

## Run The Pipeline

```bash
python -m src.run --config config.yaml --split test --run-id phase3_tagboost_v1
```

Main output:

- `outputs/runs/<run-id>/submission.json`

Helpful flags:

- `--limit N` to smoke-test on a small subset
- `--resume` to continue from a checkpointed run directory

## Run The Optional Judge

```bash
python eval/judge.py \
  --submission outputs/runs/<run-id>/submission.json \
  --run-id <run-id>_eval
```

## RAG Notes

- With `rag.enabled: true`, the pipeline first tries to load `outputs/rag_index.faiss` and `outputs/rag_meta.json`.
- If those files are absent or FAISS is unavailable, the code now falls back to building an in-memory RAG index from the local training data.
- To build a reusable FAISS index yourself, install `faiss-cpu` and then run `scripts/build_rag_index.py` after pseudo-label generation.

## Analysis Utilities

Generate the paper-facing non-debate analysis pack:

```bash
python scripts/analyze_plan_nondebate.py --root .
```

Main outputs are written under:

- `results/ablations/`
- `results/language_split/`
- `results/relation_eval/`
- `results/repair_stats/`

## Repo Notes

- The default config keeps `debate.enabled: false`.
- `commands.txt` is a convenient run log for reproducibility, but local data and outputs are intentionally ignored by git.
- `scripts/download_un_data.py` can be used to collect additional external UN data when needed.
