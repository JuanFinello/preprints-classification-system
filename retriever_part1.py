#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retriever_part1.py
Construye evidencia (contexto) por criterio para la Parte 1 (preprint-only).
No llama a ningún modelo: solo selecciona secciones relevantes.

Uso:
  python retriever_part1.py --sections paper_sections.json --criteria criteria.yaml --out retrieved_context.json
"""

import argparse
import json
from pathlib import Path
import yaml

# Mapeo fijo Parte 1: criterio -> secciones relevantes
PART1_MAP = {
    "research_question_and_methods": ["abstract", "introduction", "methods"],
    "results_and_conclusions": ["abstract", "results", "discussion"],
    "references": ["references"],
}

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Retriever Part 1 (preprint-only)")
    ap.add_argument("--sections", required=True, help="paper_sections.json from split_sections.py")
    ap.add_argument("--criteria", required=True, help="criteria.yaml")
    ap.add_argument("--out", required=True, help="Output JSON with retrieved context")
    args = ap.parse_args()

    sections_path = Path(args.sections)
    criteria_path = Path(args.criteria)
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing: {sections_path}")
    if not criteria_path.exists():
        raise FileNotFoundError(f"Missing: {criteria_path}")

    sec_data = load_json(sections_path)
    sections = sec_data.get("sections", {})

    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    crit_defs = criteria.get("criteria", {})

    out = {
        "source_file": sec_data.get("source_file", ""),
        "part": 1,
        "retrieved": {}
    }

    for crit_key, sec_list in PART1_MAP.items():
        if crit_key not in crit_defs:
            # si el YAML no lo tiene, lo omitimos
            continue

        # armar bloque de evidencia
        evidence_blocks = []
        page_spans = []

        for sname in sec_list:
            s = sections.get(sname)
            if not s:
                continue
            text = (s.get("text") or "").strip()
            if text:
                evidence_blocks.append(f"[{sname.upper()}]\n{text}")
            for span in s.get("page_spans", []):
                page_spans.append(span)

        out["retrieved"][crit_key] = {
            "label": crit_defs[crit_key].get("label", crit_key),
            "definition": {
                "score_1": crit_defs[crit_key].get("score_1", {}).get("description", ""),
                "score_2": crit_defs[crit_key].get("score_2", {}).get("description", ""),
                "score_3": crit_defs[crit_key].get("score_3", {}).get("description", ""),
            },
            "evidence_sections": sec_list,
            "page_spans": page_spans,
            "evidence_text": "\n\n".join(evidence_blocks).strip()
        }

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Wrote retrieved context to: {args.out}")
    print("✓ Criteria prepared:", ", ".join(out["retrieved"].keys()))

if __name__ == "__main__":
    main()

