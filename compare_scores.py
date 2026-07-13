#!/usr/bin/env python3
"""
Compara los puntajes de GPT (results_summary_pipeline.csv) contra los cargados
manualmente de Claude (claude_scores_pipeline.csv). Corre bien con cualquier
cantidad de papers cargados, aunque sea 1 — el reporte crece a medida
que add_claude_score.py va sumando filas.

Uso: python compare_scores.py
"""
import csv
from collections import Counter
from pathlib import Path

from evaluate_preprint import _recommendation

SUMMARY_CSV = Path("results_summary_pipeline.csv")
CLAUDE_CSV = Path("claude_scores_pipeline.csv")

CRITERIA = [
    "author_credibility",
    "research_question_and_methods",
    "results_and_conclusion",
    "references",
]


def load_gpt_scores() -> dict[str, dict]:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[row["pdf"]] = {
                **{c: int(row[f"{c}_score"]) for c in CRITERIA if row.get(f"{c}_score")},
                "recommendation": row["recommendation"],
            }
        return rows


def load_claude_scores() -> dict[str, dict]:
    if not CLAUDE_CSV.exists():
        return {}
    with CLAUDE_CSV.open(newline="", encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            scores = {c: int(row[c]) for c in CRITERIA if row.get(c)}
            rows[row["pdf"]] = scores
        return rows


def main() -> None:
    if not CLAUDE_CSV.exists():
        print(f"Todavía no hay {CLAUDE_CSV}. Cargá al menos un paper con add_claude_score.py primero.")
        return

    gpt = load_gpt_scores()
    claude = load_claude_scores()
    shared = [pdf for pdf in claude if pdf in gpt]

    if not shared:
        print("No hay papers en común entre GPT y Claude todavía.")
        return

    print(f"Papers comparados: {len(shared)} de {len(gpt)} evaluados por GPT\n")

    print(f"{'criterio':<32}{'acuerdo exacto':>16}{'diff. promedio':>16}")
    print("-" * 64)
    for c in CRITERIA:
        pairs = [(gpt[pdf][c], claude[pdf][c]) for pdf in shared if c in gpt[pdf] and c in claude[pdf]]
        if not pairs:
            continue
        exact = sum(1 for g, cl in pairs if g == cl) / len(pairs)
        mean_diff = sum(abs(g - cl) for g, cl in pairs) / len(pairs)
        print(f"{c:<32}{exact:>15.0%}{mean_diff:>16.2f}")

    print("\nMatriz de confusión por criterio (fila=GPT, columna=Claude):")
    for c in CRITERIA:
        pairs = [(gpt[pdf][c], claude[pdf][c]) for pdf in shared if c in gpt[pdf] and c in claude[pdf]]
        if not pairs:
            continue
        counts = Counter(pairs)
        print(f"\n  {c}")
        print("       Claude:  1    2    3")
        for g in (1, 2, 3):
            row = "  ".join(f"{counts.get((g, cl), 0):>3}" for cl in (1, 2, 3))
            print(f"  GPT {g}:      {row}")

    print("\nRecomendación final (misma lógica de umbrales para ambos):")
    rec_matches = 0
    disagreements = []
    for pdf in shared:
        c_scores = claude[pdf]
        if len(c_scores) < len(CRITERIA):
            continue
        c_avg = round(sum(c_scores.values()) / len(c_scores), 2)
        c_rec = _recommendation(c_avg, c_scores)
        g_rec = gpt[pdf]["recommendation"]
        if c_rec == g_rec:
            rec_matches += 1
        else:
            disagreements.append((pdf, g_rec, c_rec))

    n_full = sum(1 for pdf in shared if len(claude[pdf]) == len(CRITERIA))
    if n_full:
        print(f"  Coinciden: {rec_matches}/{n_full} ({rec_matches/n_full:.0%})")
    if disagreements:
        print("\n  Casos con recomendación distinta (prioridad para revisar el prompt):")
        for pdf, g_rec, c_rec in disagreements:
            print(f"    - {pdf}: GPT={g_rec}  Claude={c_rec}")


if __name__ == "__main__":
    main()
