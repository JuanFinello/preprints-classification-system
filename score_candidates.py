#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_candidates.py
Scores the pending "significant preprint" candidates and writes the model's
recommendation into the curator's worklist, so the score is on screen when the
Revision column gets filled in instead of arriving afterwards as an audit.

Runs after filter_significant_preprints.py and before the curator opens the CSV.

Two outputs, on purpose:
  - the worklist CSV gains model_* columns right after Revision (what the
    curator reads)
  - predictions_frozen.csv gains one append-only row per paper, stamped with
    the time and the code revision that produced it (what an audit reads)

The frozen file matters even though the prediction is visible: it records what
the model said before the decision existed, and under which rubric. Rows are
never rewritten, so a later rubric change cannot retroactively edit history.

Usage:
    python score_candidates.py                # score whatever is still pending
    python score_candidates.py --all          # include rows already decided
    python score_candidates.py --rescore      # ignore cached results, call the API again
    python score_candidates.py --dry-run      # report what would run, call nothing

Environment:
    OPENAI_API_KEY   required unless every paper is already scored or --dry-run
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from evaluate_preprint import DEFAULT_MODEL, evaluate_preprint

WORKLIST = Path("../PIPELINE/work/preprints_para_revisar.csv")
PDF_DIRS = [Path("../PIPELINE/sig_preprints_pdf"),
            Path("../Descargar_papers/MEJORADO/results/papers")]
RESULTS_DIR = Path("results_pipeline")
CRITERIA = Path("criteria.yaml")
FROZEN = Path("predictions_frozen.csv")

CRITERION_COLUMNS = ["author_credibility", "research_question_and_methods",
                     "results_and_conclusion", "references"]
MODEL_COLUMNS = (["model_recommendation", "model_average"]
                 + [f"model_{c}" for c in CRITERION_COLUMNS]
                 + ["model_flags", "model_predicted_at", "model_code_rev"])


def _doi_slug(doi: str) -> str:
    """DOI as it appears in PDF and result filenames: every slash an underscore."""
    return doi.strip().replace("/", "_")


def _find_pdf(doi: str) -> Path | None:
    """Candidate PDFs are normally already on disk: these preprints are the ones
    accessions were pulled from, so MEJORADO has them. Downloading is a fallback
    that lives in fetch_significant_preprint_pdfs.py, not here."""
    name = _doi_slug(doi) + ".pdf"
    for d in PDF_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def _code_revision() -> str:
    """Short git SHA of the scoring code, so a stored prediction says which
    rubric and prompts produced it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        rev = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except Exception:
        return "unknown"


def _prediction_row(result: dict) -> dict:
    scoring = result.get("scoring", {})
    criteria = result.get("criteria", {})
    failed = scoring.get("failed_criteria", [])
    invalid = scoring.get("quotes_invalid", [])
    flags = []
    if failed:
        flags.append("FAILED: " + ", ".join(failed))
    if invalid:
        flags.append("unverified quotes: " + ", ".join(invalid))
    if result.get("data_quality_warnings"):
        flags.append(f"{len(result['data_quality_warnings'])} author-data warning(s)")

    row = {
        "model_recommendation": scoring.get("recommendation", ""),
        "model_average": scoring.get("average", ""),
        "model_flags": " | ".join(flags),
    }
    for c in CRITERION_COLUMNS:
        score = criteria.get(c, {}).get("score")
        row[f"model_{c}"] = "" if score is None else score
    return row


def _load_worklist(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "Revision" not in fieldnames or "doi" not in fieldnames:
        sys.exit(f"{path} does not look like a curator worklist "
                 "(no 'Revision' / 'doi' column).")
    return rows, fieldnames


def _output_fieldnames(existing: list[str]) -> list[str]:
    """Model columns sit right after Revision so they are the first thing on
    screen next to the decision being made."""
    out = [c for c in existing if c not in MODEL_COLUMNS]
    at = out.index("Revision") + 1
    return out[:at] + MODEL_COLUMNS + out[at:]


def _append_frozen(records: list[dict]) -> None:
    if not records:
        return
    fields = (["doi", "pdf", "model"] + MODEL_COLUMNS)
    is_new = not FROZEN.exists()
    with FROZEN.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for r in records:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score pending candidates and write the recommendation into the curator worklist.")
    ap.add_argument("--worklist", default=str(WORKLIST), help="Curator CSV to annotate")
    ap.add_argument("--all", action="store_true",
                    help="Also score rows whose Revision is already filled in")
    ap.add_argument("--rescore", action="store_true",
                    help="Call the API again even if a result JSON already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without calling the API or writing")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    args = ap.parse_args()

    worklist = Path(args.worklist)
    if not worklist.exists():
        print(f"No worklist at {worklist} — nothing pending. Nothing to do.")
        return

    lock = worklist.parent / f".~lock.{worklist.name}#"
    if lock.exists():
        sys.exit(f"{worklist.name} is open in LibreOffice ({lock.name} present). "
                 "Close it first, or your edits and this script's will overwrite each other.")

    rows, fieldnames = _load_worklist(worklist)
    todo = [r for r in rows if args.all or not (r.get("Revision") or "").strip()]
    print(f"{len(rows)} row(s) in {worklist.name}, {len(todo)} to score"
          f"{'' if args.all else ' (pending Revision only)'}")

    client = None
    rev = _code_revision()
    now = datetime.now().isoformat(timespec="seconds")
    frozen_records = []
    n_scored = n_cached = n_skipped = 0

    for i, row in enumerate(todo, 1):
        doi = (row.get("doi") or "").strip()
        if not doi:
            continue
        out_json = RESULTS_DIR / (_doi_slug(doi) + ".json")
        pdf = _find_pdf(doi)

        if out_json.exists() and not args.rescore:
            result = json.loads(out_json.read_text(encoding="utf-8"))
            source = "cached"
            n_cached += 1
        elif pdf is None:
            print(f"[{i}/{len(todo)}] NO PDF: {doi} — looked in "
                  f"{', '.join(str(d) for d in PDF_DIRS)}")
            n_skipped += 1
            continue
        elif args.dry_run:
            print(f"[{i}/{len(todo)}] would score: {doi}  ({pdf})")
            n_scored += 1
            continue
        else:
            if client is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    sys.exit("OPENAI_API_KEY not set and there are papers left to score.")
                client = OpenAI(api_key=api_key)
            print(f"[{i}/{len(todo)}] scoring: {doi}")
            result = evaluate_preprint(pdf, CRITERIA, args.model, client)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            source = "scored"
            n_scored += 1

        pred = _prediction_row(result)
        pred["model_predicted_at"] = now
        pred["model_code_rev"] = rev
        row.update(pred)
        frozen_records.append({"doi": doi, "pdf": result.get("source_pdf", ""),
                               "model": result.get("model", args.model), **pred})
        flags = f"  [{pred['model_flags']}]" if pred["model_flags"] else ""
        print(f"    {source}: {pred['model_recommendation']} "
              f"(avg {pred['model_average']}){flags}")

    if args.dry_run:
        print(f"\nDry run: {n_scored} would be scored, {n_cached} already have results, "
              f"{n_skipped} have no PDF. Nothing written.")
        return

    if not frozen_records:
        print("\nNothing to write.")
        return

    backup = worklist.with_suffix(worklist.suffix + ".bak")
    shutil.copy2(worklist, backup)
    with worklist.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_output_fieldnames(fieldnames),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _append_frozen(frozen_records)

    print(f"\n{n_scored} scored, {n_cached} from cache, {n_skipped} without PDF.")
    print(f"{worklist} updated (backup at {backup.name}).")
    print(f"{FROZEN}: {len(frozen_records)} row(s) appended.")


if __name__ == "__main__":
    main()
