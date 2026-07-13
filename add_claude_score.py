#!/usr/bin/env python3
"""
Carga interactiva de puntajes de Claude para comparar contra los de GPT.

Flujo: le pasás el PDF a Claude manualmente (claude.ai u otro medio), te da
un puntaje 1-3 para cada criterio de criteria.yaml, y acá lo cargás. Se
guarda en claude_scores_pipeline.csv, una fila por paper.

Uso: python add_claude_score.py
"""
import csv
from datetime import date
from pathlib import Path

SUMMARY_CSV = Path("results_summary_pipeline.csv")
CLAUDE_CSV = Path("claude_scores_pipeline.csv")

CRITERIA = [
    "author_credibility",
    "research_question_and_methods",
    "results_and_conclusion",
    "references",
]
FIELDNAMES = ["pdf", *CRITERIA, "feedback", "notes", "date_added"]


def load_gpt_pdfs() -> list[str]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"No se encontró {SUMMARY_CSV} — corré batch_evaluate.py primero.")
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return [row["pdf"] for row in csv.DictReader(f)]


def load_claude_rows() -> dict[str, dict]:
    if not CLAUDE_CSV.exists():
        return {}
    with CLAUDE_CSV.open(newline="", encoding="utf-8") as f:
        return {row["pdf"]: row for row in csv.DictReader(f)}


def save_claude_rows(rows: dict[str, dict]) -> None:
    with CLAUDE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def prompt_score(criterion: str) -> int:
    while True:
        raw = input(f"  {criterion} (1-3): ").strip()
        if raw in ("1", "2", "3"):
            return int(raw)
        print("  Ingresá 1, 2 o 3.")


def prompt_feedback() -> str:
    raw = input("  feedback (1-3, o Enter para omitir): ").strip()
    return raw if raw in ("1", "2", "3") else ""


def main() -> None:
    gpt_pdfs = load_gpt_pdfs()
    claude_rows = load_claude_rows()

    pending = [p for p in gpt_pdfs if p not in claude_rows]
    done = [p for p in gpt_pdfs if p in claude_rows]
    print(f"Papers evaluados por GPT: {len(gpt_pdfs)} | ya cargados con Claude: {len(done)} | pendientes: {len(pending)}\n")

    if not pending:
        print("No hay papers pendientes de cargar. Podés re-cargar uno existente escribiendo su nombre.")
        options = gpt_pdfs
    else:
        options = pending

    for i, pdf in enumerate(options, 1):
        print(f"  {i}. {pdf}")

    choice = input("\nElegí número, o pegá/escribí el nombre del PDF (Enter para salir): ").strip()
    if not choice:
        return

    if choice.isdigit() and 1 <= int(choice) <= len(options):
        pdf = options[int(choice) - 1]
    else:
        matches = [p for p in gpt_pdfs if choice in p]
        if len(matches) == 1:
            pdf = matches[0]
        elif len(matches) > 1:
            print("Coincide con más de un PDF, sé más específico:")
            for m in matches:
                print(f"  - {m}")
            return
        else:
            print(f"No se encontró un PDF que coincida con '{choice}'.")
            return

    if pdf in claude_rows:
        overwrite = input(f"'{pdf}' ya tiene puntaje de Claude cargado. ¿Sobrescribir? (s/N): ").strip().lower()
        if overwrite != "s":
            return

    print(f"\nCargando puntajes de Claude para: {pdf}")
    scores = {c: prompt_score(c) for c in CRITERIA}
    feedback = prompt_feedback()
    notes = input("  notas (opcional): ").strip()

    claude_rows[pdf] = {
        "pdf": pdf,
        **{c: str(scores[c]) for c in CRITERIA},
        "feedback": feedback,
        "notes": notes,
        "date_added": date.today().isoformat(),
    }
    save_claude_rows(claude_rows)
    print(f"\nGuardado en {CLAUDE_CSV}. Total cargados: {len(claude_rows)}.")


if __name__ == "__main__":
    main()
