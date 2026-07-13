#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from openai import OpenAI

OPENAI_API_KEY = "REDACTED_API_KEY"


# ---------------------------
# PDF
# ---------------------------
def extract_text_by_page(pdf_path: Path):
    pages = []
    for i, layout in enumerate(extract_pages(str(pdf_path)), start=1):
        texts = []
        for el in layout:
            if isinstance(el, LTTextContainer):
                texts.append(el.get_text())
        pages.append({"page": i, "text": "\n".join(texts)})
    return pages


def clean_text(text: str):
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("-\n", "")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ---------------------------
# chunking
# ---------------------------
def chunk_pages(pages, chunk_size=5000, overlap=1000):
    full = ""
    markers = []

    for p in pages:
        txt = clean_text(p["text"])
        if not txt:
            continue
        start = len(full)
        full += txt + "\n\n"
        end = len(full)
        markers.append((start, end, p["page"]))

    chunks = []
    i = 0
    cid = 0

    while i < len(full):
        j = min(i + chunk_size, len(full))
        text = full[i:j]

        pages_in_chunk = []
        for s, e, p in markers:
            if not (e < i or s > j):
                pages_in_chunk.append(p)

        chunks.append({
            "chunk_id": cid,
            "text": text.strip(),
            "pages": sorted(set(pages_in_chunk))
        })

        if j == len(full):
            break

        i = j - overlap
        cid += 1

    return chunks


# ---------------------------
# embeddings
# ---------------------------
def embed(client, texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([x.embedding for x in resp.data], dtype=np.float32)


def cosine(q, d):
    q = q / (np.linalg.norm(q) + 1e-8)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    return np.dot(d, q)


# ---------------------------
# retrieval heuristics
# ---------------------------
METHOD_KEYWORDS = [
    "methods", "methodology", "experimental", "experimental methods",
    "materials", "materials and methods", "study design",
    "procedure", "protocol", "measured", "measurement",
    "instrument", "equipment", "assay", "sequencing",
    "thermal conductivity", "specific heat capacity", "emissivity",
    "tga", "dsc", "ftir", "tps", "steady-state"
]

OBJECTIVE_KEYWORDS = [
    "this study aims",
    "the aim of this study",
    "our aim",
    "objective",
    "research objective",
    "we investigate",
    "we evaluated",
    "we examined",
    "we characterized",
    "to address",
    "to determine",
    "to assess",
    "to quantify",
    "to analyze"
]

GENERIC_INTRO_PATTERNS = [
    "is widely used",
    "are widely used",
    "has attracted significant interest",
    "has attracted attention",
    "is indispensable",
    "plays an important role",
    "is commonly employed",
    "is extensively utilized",
    "plenty studies have been conducted",
    "many studies have been conducted"
]


def keyword_bonus(text):
    t = text.lower()
    method_hits = sum(1 for k in METHOD_KEYWORDS if k in t)
    objective_hits = sum(1 for k in OBJECTIVE_KEYWORDS if k in t)
    return min(method_hits * 0.025 + objective_hits * 0.04, 0.22)


def intro_penalty(text):
    t = text.lower()
    hits = sum(1 for p in GENERIC_INTRO_PATTERNS if p in t)
    return min(hits * 0.05, 0.20)


def page_penalty(pages, max_page):
    if not pages:
        return 0.0
    mean_page = sum(pages) / len(pages)
    frac = mean_page / max_page
    if frac > 0.85:
        return 0.22
    elif frac > 0.70:
        return 0.10
    return 0.0


def has_objective_signal(text):
    t = text.lower()
    return any(k in t for k in OBJECTIVE_KEYWORDS)


def has_method_signal(text):
    t = text.lower()
    return any(k in t for k in METHOD_KEYWORDS)


def ensure_coverage(retrieved):
    has_objective = any(has_objective_signal(c["text"]) for c in retrieved)
    has_method = any(has_method_signal(c["text"]) for c in retrieved)
    return has_objective, has_method


def build_retrieval_query(label):
    return f"""
Find the most relevant passages in this scientific preprint for evaluating the criterion
'{label}'.

You must prioritize passages that EXPLICITLY describe:

1. The research objective, aim, or hypothesis of the study
   Examples:
   - "this study aims"
   - "the aim of this study"
   - "we investigate"
   - "to address this gap"

2. The methodology or experimental setup
   Examples:
   - methods
   - materials and methods
   - experimental methods
   - procedure
   - protocol
   - measured using
   - instrument used
   - assay or sequencing approach
   - computational or analytical workflow

PRIORITY ORDER:
- Abstract
- Explicit objective statements in introduction
- Methods / Experimental sections
- Detailed measurement or analytical sections

DEPRIORITIZE:
- General background
- Generic literature review
- Discussion
- Conclusions
- References
- Introductory statements that only say why the topic matters

Return passages that contain CLEAR and EXPLICIT statements, not generic context.
""".strip()


def rank_chunks(chunks, sims, max_page):
    adjusted = []
    for sim, chunk in zip(sims, chunks):
        score = (
            sim
            - page_penalty(chunk["pages"], max_page)
            - intro_penalty(chunk["text"])
            + keyword_bonus(chunk["text"])
        )
        adjusted.append(score)
    return np.array(adjusted)


def select_retrieved_chunks(chunks, adjusted_scores, top_k=5):
    idx = np.argsort(-adjusted_scores)[:top_k]
    return [chunks[i] for i in idx], idx


def retrieve_with_coverage(chunks, adjusted_scores, initial_top_k=5, fallback_top_k=8):
    retrieved, idx = select_retrieved_chunks(chunks, adjusted_scores, top_k=initial_top_k)
    has_obj, has_meth = ensure_coverage(retrieved)

    if has_obj and has_meth:
        return retrieved, idx

    # fallback: abrir un poco más el top_k
    retrieved2, idx2 = select_retrieved_chunks(chunks, adjusted_scores, top_k=fallback_top_k)

    # si aún falta cobertura, forzar inclusión del mejor chunk con señal faltante
    has_obj2, has_meth2 = ensure_coverage(retrieved2)
    final = list(retrieved2)
    final_ids = {c["chunk_id"] for c in final}

    if not has_obj2:
        objective_candidates = [
            (score, chunk) for score, chunk in zip(adjusted_scores, chunks)
            if has_objective_signal(chunk["text"])
        ]
        objective_candidates.sort(key=lambda x: -x[0])
        if objective_candidates:
            best_obj = objective_candidates[0][1]
            if best_obj["chunk_id"] not in final_ids:
                final.append(best_obj)
                final_ids.add(best_obj["chunk_id"])

    if not has_meth2:
        method_candidates = [
            (score, chunk) for score, chunk in zip(adjusted_scores, chunks)
            if has_method_signal(chunk["text"])
        ]
        method_candidates.sort(key=lambda x: -x[0])
        if method_candidates:
            best_meth = method_candidates[0][1]
            if best_meth["chunk_id"] not in final_ids:
                final.append(best_meth)
                final_ids.add(best_meth["chunk_id"])

    # ordenar final por score
    final.sort(key=lambda c: -adjusted_scores[c["chunk_id"]])
    return final[:fallback_top_k], np.array([c["chunk_id"] for c in final[:fallback_top_k]])


# ---------------------------
# criteria
# ---------------------------
def load_criterion(path, key):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    c = data["criteria"][key]
    return {
        "key": key,
        "label": c["label"],
        "score_1": c["score_1"]["description"],
        "score_2": c["score_2"]["description"],
        "score_3": c["score_3"]["description"],
        "evidence_to_look_for": c.get("evidence_to_look_for", []),
    }


# ---------------------------
# prompt
# ---------------------------
def build_prompt(c, chunks):
    evidence = "\n\n".join([
        f"[CHUNK {x['chunk_id']} | pages {x['pages']}]\n{x['text']}"
        for x in chunks
    ])

    return f"""
You are evaluating a scientific preprint for one rubric criterion.

STRICT RULES:
- Use ONLY the evidence below.
- Do NOT use outside knowledge.
- Do NOT infer missing information.
- Every quote must be copied VERBATIM from the evidence.
- Each quote must be short (max 220 characters).
- If you cannot find verbatim evidence, do not invent it.
- If evidence is weak or incomplete, choose the lowest defensible score.
- Return ONLY valid JSON, with no markdown fences.
- Do NOT criticize the paper for missing elements unless there is explicit evidence in the text that they are missing.
- Do NOT assume that sample size, validation, or statistical analysis are required unless the study type clearly demands them.
- Base all criticisms ONLY on what is explicitly present or absent in the provided evidence.
- If clear evidence of both a research objective and a described methodology is present, you should assign a score of 3 unless there is explicit evidence of major missing elements.
- Do NOT default to lower scores when sufficient evidence is present.
- Prefer higher scores when the criterion is clearly satisfied by the evidence.
- You MUST include at least:
  • one quote describing the research objective or aim
  • one quote describing the methodology or experimental setup
- Prefer evidence from methods or experimental sections over introduction/background.
- Do NOT base your evaluation only on objectives or introductory text if methodological evidence is available.

CRITERION:
{c['label']}

SCORING DEFINITIONS:
1 = {c['score_1']}
2 = {c['score_2']}
3 = {c['score_3']}

EVIDENCE TO LOOK FOR:
{", ".join(c["evidence_to_look_for"])}

EVIDENCE FROM PREPRINT:
{evidence}

Return exactly this JSON schema:
{{
  "criterion": "{c['key']}",
  "score": 1,
  "justification": "brief explanation grounded only in the evidence",
  "evidence": [
    {{
      "chunk_id": 0,
      "pages": [1],
      "quote": "exact short quote copied from the evidence"
    }}
  ]
}}

ADDITIONAL RULE:
- Score 3 requires:
  • at least one clear quote stating the research objective or aim
  • at least one clear quote describing the methodology or experimental setup
  • the methodology must contain concrete technical details (e.g., measurement techniques, instruments, procedures)

- Do NOT assign score 3 if:
  • the objective is vague or implicit
  • the methodology is only mentioned without describing how it was performed

- Do NOT penalize for missing elements such as sample size, controls, or statistical analysis unless they are explicitly discussed in the evidence.

- If objective and method are both present but the methodology lacks concrete technical detail, assign score 2.
""".strip()


# ---------------------------
# output cleaning
# ---------------------------
def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


# ---------------------------
# evidence validation
# ---------------------------
def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_evidence(parsed, retrieved):
    full_text = "\n\n".join(c["text"] for c in retrieved)
    norm_full = normalize_for_match(full_text)

    if "evidence" not in parsed or not isinstance(parsed["evidence"], list):
        parsed["evidence"] = []
        parsed["warning"] = "No evidence field returned by model."
        return parsed

    valid = []
    for e in parsed["evidence"]:
        quote = str(e.get("quote", "")).strip()
        if not quote:
            continue

        norm_quote = normalize_for_match(quote)

        if norm_quote and norm_quote in norm_full:
            valid.append(e)
            continue

        if len(norm_quote) >= 60:
            words = norm_quote.split()
            partial = " ".join(words[: min(12, len(words))])
            if partial and partial in norm_full:
                valid.append(e)
                continue

    if not valid:
        parsed["warning"] = "No evidence matched retrieved text."

    parsed["evidence"] = valid
    return parsed


def downgrade_score_if_needed(parsed):
    score = parsed.get("score")
    evidence = parsed.get("evidence", [])

    if not isinstance(score, int):
        parsed["score"] = 1
        parsed["warning"] = parsed.get("warning", "") + " Invalid score returned; forced to 1."
        return parsed

    n_ev = len(evidence)

    if n_ev == 0:
        parsed["score"] = 1
        parsed["justification"] = (
            "Score was downgraded automatically because no valid verbatim evidence "
            "was found in the retrieved text."
        )
        return parsed

    if score == 3 and n_ev < 2:
        parsed["score"] = 2
        parsed["justification"] = (
            "Original score 3 was downgraded to 2 because fewer than two valid "
            "verbatim evidence quotes were found."
        )
        return parsed

    return parsed


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--criterion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    args = ap.parse_args()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    )

    pdf_path = Path(args.pdf)
    criteria_path = Path(args.criteria)

    if not pdf_path.exists():
        raise FileNotFoundError(f"No existe PDF: {pdf_path}")
    if not criteria_path.exists():
        raise FileNotFoundError(f"No existe criteria.yaml: {criteria_path}")

    # 1) extraer texto
    pages = extract_text_by_page(pdf_path)

    # 2) chunking
    chunks = chunk_pages(pages)

    # 3) criterio
    crit = load_criterion(criteria_path, args.criterion)

    # 4) embeddings
    texts = [c["text"] for c in chunks]
    vecs = embed(client, texts, model=args.embedding_model)

    # 5) retrieval
    query = build_retrieval_query(crit["label"])
    qvec = embed(client, [query], model=args.embedding_model)[0]
    sims = cosine(qvec, vecs)

    max_page = max(p["page"] for p in pages) if pages else 1
    adjusted = rank_chunks(chunks, sims, max_page)
    retrieved, idx = retrieve_with_coverage(
        chunks,
        adjusted,
        initial_top_k=args.top_k,
        fallback_top_k=max(args.top_k + 3, 8)
    )

    # 6) prompt
    prompt = build_prompt(crit, retrieved)

    # 7) scoring
    resp = client.responses.create(
        model=args.llm_model,
        input=prompt
    )

    raw = clean_json(resp.output_text)

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "criterion": args.criterion,
            "score": 1,
            "justification": "Model output could not be parsed as valid JSON.",
            "evidence": [],
            "raw_output": raw
        }

    # 8) validar evidencia
    parsed = validate_evidence(parsed, retrieved)

    # 9) bajar score si falta grounding
    parsed = downgrade_score_if_needed(parsed)

    # 10) flags útiles para debug
    has_obj, has_meth = ensure_coverage(retrieved)

    out = {
        "source_pdf": pdf_path.name,
        "criterion": args.criterion,
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved],
        "retrieved_pages": [c["pages"] for c in retrieved],
        "retrieval_debug": {
            "has_objective_chunk": has_obj,
            "has_method_chunk": has_meth,
            "n_retrieved": len(retrieved)
        },
        "result": parsed
    }

    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✓ Resultado guardado en {args.out}")


if __name__ == "__main__":
    main()
