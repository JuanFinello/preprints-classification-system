#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_evaluate_claude.py
Runs evaluate_preprint_claude over every PDF already scored by GPT
(results_pipeline/*.json) and writes the scores straight into
claude_scores_pipeline.csv — replacing the old manual add_claude_score.py step.

Rows that already have all 4 scores filled in (manually via add_claude_score.py,
or from a previous run of this script) are left untouched unless --overwrite
is passed, so a manual correction never gets silently clobbered.

Usage:
    python batch_evaluate_claude.py
    python batch_evaluate_claude.py --overwrite

Environment:
    ANTHROPIC_API_KEY   required
"""

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import date
from pathlib import Path

import anthropic

from evaluate_preprint_claude import DEFAULT_MODEL, evaluate_preprint_claude

PDFS_DIR = Path("../PIPELINE/sig_preprints_pdf")
GPT_RESULTS_DIR = Path("results_pipeline")
CLAUDE_RESULTS_DIR = Path("results_pipeline_claude")
CLAUDE_CSV = Path("claude_scores_pipeline.csv")
CRITERIA_PATH = Path("criteria.yaml")

CRITERIA_SCORES = ["author_credibility", "research_question_and_methods",
                    "results_and_conclusion", "references"]
CLAUDE_FIELDNAMES = ["pdf", *CRITERIA_SCORES, "feedback", "notes", "date_added"]


def _flatten(result: dict) -> dict:
    criteria = result.get("criteria", {})
    row = {"pdf": result["source_pdf"]}
    for c in CRITERIA_SCORES:
        row[c] = criteria.get(c, {}).get("score", "")
    row["feedback"] = ""
    row["notes"] = f"Auto-scored by {result.get('model', DEFAULT_MODEL)} via API"
    row["date_added"] = date.today().isoformat()
    return row


def _load_claude_rows(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["pdf"]: row for row in csv.DictReader(f)}


def _save_claude_rows(csv_path: Path, rows: dict[str, dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLAUDE_FIELDNAMES)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({fn: row.get(fn, "") for fn in CLAUDE_FIELDNAMES})


def _is_fully_scored(row: dict) -> bool:
    return all(str(row.get(c, "")).strip() for c in CRITERIA_SCORES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score GPT-evaluated preprints with Claude via API.")
    ap.add_argument("--pdfs-dir", default=str(PDFS_DIR))
    ap.add_argument("--gpt-results-dir", default=str(GPT_RESULTS_DIR))
    ap.add_argument("--results-dir", default=str(CLAUDE_RESULTS_DIR))
    ap.add_argument("--csv", default=str(CLAUDE_CSV))
    ap.add_argument("--criteria", default=str(CRITERIA_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true",
                     help="Re-score papers that already have all 4 scores filled in")
    args = ap.parse_args()

    csv_path = Path(args.csv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    pdfs_dir = Path(args.pdfs_dir)
    gpt_results_dir = Path(args.gpt_results_dir)
    results_dir = Path(args.results_dir)
    criteria_path = Path(args.criteria)

    gpt_jsons = sorted(gpt_results_dir.glob("*.json"))
    if not gpt_jsons:
        print(f"No GPT results found in {gpt_results_dir} — run batch_evaluate.py first.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    claude_rows = _load_claude_rows(csv_path)
    client = anthropic.Anthropic(api_key=api_key)

    n_scored = 0
    for i, gpt_json_path in enumerate(gpt_jsons, 1):
        gpt_result = json.loads(gpt_json_path.read_text(encoding="utf-8"))
        pdf_name = gpt_result["source_pdf"]
        pdf_path = pdfs_dir / pdf_name

        existing_row = claude_rows.get(pdf_name)
        if existing_row and _is_fully_scored(existing_row) and not args.overwrite:
            print(f"[{i}/{len(gpt_jsons)}] SKIP (already scored): {pdf_name}")
            continue

        if not pdf_path.exists():
            print(f"[{i}/{len(gpt_jsons)}] SKIP (PDF not found): {pdf_path}")
            continue

        print(f"[{i}/{len(gpt_jsons)}] Scoring with Claude: {pdf_name}")
        try:
            result = evaluate_preprint_claude(
                pdf_path, criteria_path, args.model, client,
                author_data=gpt_result["author_enrichment"],
            )
            out_path = results_dir / gpt_json_path.name
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            claude_rows[pdf_name] = _flatten(result)
            n_scored += 1
            scoring = result["scoring"]
            print(f"  -> avg {scoring['average']}  (scores: {scoring['per_criterion']})")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    _save_claude_rows(csv_path, claude_rows)
    print(f"\n{'='*55}")
    print(f"{n_scored} paper(s) scored this run. {csv_path} updated ({len(claude_rows)} total rows).")


if __name__ == "__main__":
    main()
