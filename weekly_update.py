#!/usr/bin/env python3
"""
Corrida semanal única: sincroniza PDFs nuevos de candidatos "significant
preprint", los evalúa con GPT, genera los prompts de datos de autor para
Claude, y agrega filas vacías al CSV de scores de Claude solo para los
papers nuevos (nunca pisa lo que ya completaste).

Uso: python weekly_update.py
"""
import csv
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path("../PIPELINE").resolve()
RESULTS_DIR = Path("results_pipeline")
SUMMARY_CSV = Path("results_summary_pipeline.csv")
CLAUDE_CSV = Path("claude_scores_pipeline.csv")

CLAUDE_FIELDNAMES = [
    "pdf", "author_credibility", "research_question_and_methods",
    "results_and_conclusion", "references", "feedback", "notes", "date_added",
]


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}  (en {cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def update_claude_template() -> int:
    """Agrega filas vacías para pdfs nuevos. No modifica filas existentes."""
    if not SUMMARY_CSV.exists():
        return 0

    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        gpt_pdfs = [row["pdf"] for row in csv.DictReader(f)]

    existing_rows = []
    existing_pdfs = set()
    if CLAUDE_CSV.exists():
        with CLAUDE_CSV.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
            existing_pdfs = {row["pdf"] for row in existing_rows}

    new_pdfs = [p for p in gpt_pdfs if p not in existing_pdfs]
    if not new_pdfs:
        return 0

    for pdf in new_pdfs:
        existing_rows.append({fn: ("" if fn != "pdf" else pdf) for fn in CLAUDE_FIELDNAMES})

    with CLAUDE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLAUDE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows)

    return len(new_pdfs)


def main() -> None:
    run(
        [sys.executable, "scripts/fetch_significant_preprint_pdfs.py"],
        cwd=PIPELINE_DIR,
    )
    run(
        [sys.executable, "batch_evaluate.py",
         "--papers-dir", str(PIPELINE_DIR / "sig_preprints_pdf"),
         "--results-dir", str(RESULTS_DIR),
         "--summary", str(SUMMARY_CSV)],
        cwd=Path("."),
    )
    run([sys.executable, "export_for_claude.py"], cwd=Path("."))

    n_new = update_claude_template()

    print(f"\n{'='*55}")
    if n_new:
        print(f"{n_new} paper(s) nuevo(s) agregado(s) a {CLAUDE_CSV} (filas vacías).")
        print(f"Completalas con los scores de Claude y después corré compare_scores.py.")
    else:
        print(f"Sin papers nuevos esta semana. {CLAUDE_CSV} sin cambios.")


if __name__ == "__main__":
    main()
