#!/usr/bin/env python3
"""
Genera un único CSV con los datos de autores/instituciones (ROR + Semantic
Scholar) que Claude no puede conseguir por su cuenta — una fila por paper,
para pegarle junto con el PDF (que sí le subís directo) en el chat.

Usa la MISMA función que arma ese bloque para GPT (_build_author_prompt de
evaluate_preprint.py), así el dato es texto-idéntico al que vio GPT. Se
regenera entero en cada corrida a partir de los JSON en results_pipeline/,
así que siempre refleja el estado actual (no hace falta borrar nada a mano).

Uso:
    python export_for_claude.py
"""
import csv
import json
from pathlib import Path

import yaml

from evaluate_preprint import _build_author_prompt

RESULTS_DIR = Path("results_pipeline")
CRITERIA_PATH = Path("criteria.yaml")
OUT_CSV = Path("claude_author_data.csv")

FIELDNAMES = [
    "pdf", "corresponding_author", "n_authors", "all_authors",
    "n_institutes", "affiliations", "ror_summary", "semantic_scholar_summary",
    "prompt_text",
]


def ror_summary(ror_results: list[dict]) -> str:
    if not ror_results:
        return "Sin datos de ROR"
    return "; ".join(
        f"{r.get('name', '?')} ({', '.join(r.get('type', []))}, {r.get('country', '?')})"
        for r in ror_results
    )


def s2_summary(s2: dict) -> str:
    authors = s2.get("authors", [])
    if not authors:
        return "Sin datos de Semantic Scholar"
    summary = "; ".join(
        f"{a.get('name', '?')}: papers={a.get('paper_count', '?')}, "
        f"h_index={a.get('h_index', '?')}, match_method={a.get('match_method', '?')}"
        for a in authors
    )
    warnings = s2.get("warnings", [])
    if warnings:
        summary += " [WARNINGS: " + " | ".join(warnings) + "]"
    return summary


def author_data_block(author_data: dict, criteria: dict) -> str:
    """
    Igual a _build_author_prompt (mismo texto que vio GPT), pero sin el
    "Return ONLY this JSON..." final: ese pide el schema de UN solo criterio
    (author_credibility), y en el uso real Juan le pide a Claude los 4
    criterios juntos en el mismo mensaje — dejar esa instrucción puesta
    generaba respuestas confundidas (Claude la marcó como contradictoria en
    más de un caso real).
    """
    full = _build_author_prompt(author_data, criteria)
    return full.split("Return ONLY this JSON")[0].strip()


def build_row(pdf_stem: str, criteria: dict) -> dict:
    result_path = RESULTS_DIR / (pdf_stem + ".json")
    gpt_result = json.loads(result_path.read_text(encoding="utf-8"))
    author_data = gpt_result["author_enrichment"]
    extracted = author_data["extracted"]

    return {
        "pdf": pdf_stem + ".pdf",
        "corresponding_author": extracted.get("corresponding_author", ""),
        "n_authors": extracted.get("n_authors", ""),
        "all_authors": "; ".join(extracted.get("all_authors", [])),
        "n_institutes": extracted.get("n_institutes", ""),
        "affiliations": "; ".join(extracted.get("affiliations", [])),
        "ror_summary": ror_summary(author_data.get("ror_results", [])),
        "semantic_scholar_summary": s2_summary(author_data.get("semantic_scholar", {})),
        "prompt_text": author_data_block(author_data, criteria),
    }


def main() -> None:
    criteria = yaml.safe_load(CRITERIA_PATH.read_text(encoding="utf-8"))["criteria"]
    stems = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))

    rows = []
    for stem in stems:
        try:
            rows.append(build_row(stem, criteria))
        except Exception as e:
            print(f"SKIP {stem}.pdf: {e}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} paper(s) → {OUT_CSV}")


if __name__ == "__main__":
    main()
