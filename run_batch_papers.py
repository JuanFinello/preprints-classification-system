#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
import re
from pathlib import Path

import pandas as pd


def detect_publication_type(filename: str) -> str:
    stem = Path(filename).stem.lower()
    parts = stem.split("_")
    last = parts[-1]

    if last.startswith("v") and last[1:].isdigit():
        return "preprint"
    return "article"


def stringify_evidence(obj, met):
    parts = []

    if isinstance(obj, dict):
        parts.append(f"[OBJECTIVE] {obj.get('quote','')}")

    if isinstance(met, dict):
        parts.append(f"[METHOD] {met.get('quote','')}")

    return " || ".join(parts)


def safe_stem(pdf_path: Path) -> str:
    return re.sub(r"[^\w\-\.]+", "_", pdf_path.stem)


def run_single_pdf(runner_script, pdf_path, criteria_path, criterion, python_cmd, json_out_dir):
    out_json = json_out_dir / f"{safe_stem(pdf_path)}.json"

    cmd = [
        python_cmd,
        str(runner_script),
        "--pdf", str(pdf_path),
        "--criteria", str(criteria_path),
        "--criterion", criterion,
        "--out", str(out_json),
    ]

    publication_type = detect_publication_type(pdf_path.name)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return {
            "pdf": pdf_path.name,
            "publication_type": publication_type,
            "error": proc.stderr
        }

    data = json.loads(out_json.read_text())

    result = data.get("result", {})

    obj = result.get("objective_evidence")
    met = result.get("method_evidence")

    return {
        "pdf": pdf_path.name,
        "publication_type": publication_type,
        "paper_type": result.get("paper_type"),
        "score": result.get("score"),
        "justification": result.get("justification"),
        "objective_quote": obj.get("quote") if obj else "",
        "method_quote": met.get("quote") if met else "",
        "evidence": stringify_evidence(obj, met),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--runner-script", default="mvp_preprint_rag.py")
    parser.add_argument("--criteria", default="criteria.yaml")
    parser.add_argument("--criterion", default="research_question_and_methods")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    json_out_dir = Path("batch_json")
    json_out_dir.mkdir(exist_ok=True)

    pdfs = list(papers_dir.glob("*.pdf"))

    rows = []

    for pdf in pdfs:
        print("Processing:", pdf.name)
        row = run_single_pdf(
            Path(args.runner_script),
            pdf,
            Path(args.criteria),
            args.criterion,
            sys.executable,
            json_out_dir
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("batch_results.xlsx", index=False)

    print("DONE")


if __name__ == "__main__":
    main()
