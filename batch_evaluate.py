#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_evaluate.py
Runs evaluate_preprint over all PDFs in a directory and consolidates results
into a summary CSV.

Usage:
    python batch_evaluate.py
    python batch_evaluate.py --papers-dir papers --results-dir results --model claude-haiku-4-5-20251001

Environment:
    OPENAI_API_KEY   required
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from openai import OpenAI

from evaluate_preprint import DEFAULT_MODEL, evaluate_preprint


def process_all(papers_dir: Path, results_dir: Path, criteria_path: Path,
                model: str, client: OpenAI, skip_existing: bool = True) -> list[dict]:

    results_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(papers_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {papers_dir}")
        return []

    print(f"Found {len(pdfs)} PDFs in {papers_dir}")
    rows = []

    for i, pdf in enumerate(pdfs, 1):
        out_path = results_dir / (pdf.stem + ".json")

        if skip_existing and out_path.exists():
            print(f"[{i}/{len(pdfs)}] SKIP (already evaluated): {pdf.name}")
            try:
                data = json.loads(out_path.read_text())
                rows.append(_flatten(data))
            except Exception:
                pass
            continue

        print(f"[{i}/{len(pdfs)}] Processing: {pdf.name}")

        try:
            result = evaluate_preprint(pdf, criteria_path, model, client)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            scoring = result["scoring"]
            failed = scoring.get("failed_criteria", [])
            if failed:
                print(f"  ✗ EVALUATION FAILED: no usable answer for {', '.join(failed)}")
                print(f"    (diagnostics saved in {out_path.name}, under criteria.<name>.error)")
            else:
                print(f"  → {scoring['recommendation']}  (avg {scoring['average']})")

            warnings = result.get("data_quality_warnings", [])
            if warnings:
                print(f"  ⚠ DATA QUALITY WARNING — author_credibility scored on incomplete data:")
                for w in warnings:
                    print(f"    - {w}")

            rows.append(_flatten(result))

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            rows.append({
                "pdf": pdf.name,
                "error": str(e),
                "recommendation": "error",
            })

    return rows


def _flatten(result: dict) -> dict:
    """Flatten a full result JSON into a single CSV row."""
    scoring = result.get("scoring", {})
    criteria = result.get("criteria", {})
    author_enrich = result.get("author_enrichment", {})
    extracted = author_enrich.get("extracted", {})
    s2_authors = author_enrich.get("semantic_scholar", {}).get("authors", [])

    row = {
        "pdf": result.get("source_pdf", ""),
        "model": result.get("model", ""),
        "n_pages": result.get("n_pages", ""),
        "recommendation": scoring.get("recommendation", ""),
        "average_score": scoring.get("average", ""),
        "total_score": scoring.get("total", ""),
        "max_score": scoring.get("max", ""),
        "failed_criteria": "; ".join(scoring.get("failed_criteria", [])),
    }

    for crit in ["author_credibility", "research_question_and_methods",
                 "results_and_conclusion", "references", "feedback"]:
        c = criteria.get(crit, {})
        row[f"{crit}_score"] = c.get("score", "")
        row[f"{crit}_justification"] = c.get("justification", "")

    sub_scores = criteria.get("author_credibility", {}).get("sub_scores", {})
    for key in ("institution_reputability", "author_expertise", "institutional_collaboration"):
        row[f"author_credibility_{key}_score"] = sub_scores.get(key, {}).get("score", "")

    rqm_sub_scores = criteria.get("research_question_and_methods", {}).get("sub_scores", {})
    for key in ("objective_and_hypothesis", "public_health_relevance", "study_design_rigor"):
        row[f"research_question_and_methods_{key}_score"] = rqm_sub_scores.get(key, {}).get("score", "")

    row["n_authors"] = extracted.get("n_authors", "")
    row["n_institutes"] = extracted.get("n_institutes", "")
    row["corresponding_author"] = extracted.get("corresponding_author", "")
    row["s2_authors_verified"] = len(s2_authors)
    row["s2_avg_hindex"] = (
        round(sum(a.get("h_index", 0) for a in s2_authors) / len(s2_authors), 1)
        if s2_authors else ""
    )

    flags = criteria.get("author_credibility", {}).get("flags", [])
    row["author_flags"] = "; ".join(flags) if flags else ""
    row["data_quality_warnings"] = " | ".join(result.get("data_quality_warnings", []))

    return row


def main():
    ap = argparse.ArgumentParser(description="Batch evaluate preprints for GISAID eligibility.")
    ap.add_argument("--papers-dir", default="papers", help="Directory with PDF files")
    ap.add_argument("--results-dir", default="results", help="Directory for JSON outputs")
    ap.add_argument("--criteria", default="criteria.yaml", help="Path to criteria.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    ap.add_argument("--summary", default="results_summary.csv", help="Output summary CSV path")
    ap.add_argument("--no-skip", action="store_true", help="Re-evaluate even if result already exists")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    papers_dir = Path(args.papers_dir)
    results_dir = Path(args.results_dir)
    criteria_path = Path(args.criteria)

    if not papers_dir.exists():
        print(f"ERROR: papers directory not found: {papers_dir}", file=sys.stderr)
        sys.exit(1)
    if not criteria_path.exists():
        print(f"ERROR: criteria.yaml not found: {criteria_path}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    rows = process_all(papers_dir, results_dir, criteria_path,
                       args.model, client, skip_existing=not args.no_skip)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.summary, index=False, encoding="utf-8")
        print(f"\n{'='*55}")
        print(f"Total evaluated: {len(df)}")
        if "recommendation" in df.columns:
            print(df["recommendation"].value_counts().to_string())
        print(f"Summary saved → {args.summary}")


if __name__ == "__main__":
    main()
