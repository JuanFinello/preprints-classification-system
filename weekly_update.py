#!/usr/bin/env python3
"""
Corrida semanal única: sincroniza PDFs nuevos de candidatos "significant
preprint", los evalúa con GPT, y los puntúa con Claude vía API — todo
automático, sin pasos manuales. Los papers ya puntuados (por cualquiera de
los dos) se saltean.

Uso: python weekly_update.py
"""
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path("../PIPELINE").resolve()
RESULTS_DIR = Path("results_pipeline")
SUMMARY_CSV = Path("results_summary_pipeline.csv")
CLAUDE_CSV = Path("claude_scores_pipeline.csv")


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}  (en {cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


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
    run(
        [sys.executable, "batch_evaluate_claude.py",
         "--pdfs-dir", str(PIPELINE_DIR / "sig_preprints_pdf"),
         "--gpt-results-dir", str(RESULTS_DIR),
         "--csv", str(CLAUDE_CSV)],
        cwd=Path("."),
    )

    print(f"\n{'='*55}")
    print(f"Listo. Corré compare_scores.py para ver el acuerdo GPT vs. Claude.")


if __name__ == "__main__":
    main()
