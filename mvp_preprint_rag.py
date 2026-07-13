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
    "statistical analysis", "analysis was performed", "randomized",
    "cohort study", "retrospective cohort", "prospective study",
    "participants", "sample", "sampling", "dataset",
    "thermal conductivity", "specific heat capacity", "emissivity",
    "tga", "dsc", "ftir", "tps", "steady-state"
]

OBJECTIVE_KEYWORDS = [
    "this study aims",
    "the aim of this study",
    "our aim",
    "objective",
    "research objective",
    "primary aim",
    "secondary aim",
    "we investigate",
    "we evaluated",
    "we examined",
    "we characterized",
    "to address",
    "to determine",
    "to assess",
    "to quantify",
    "to analyze",
    "we sought to answer",
    "purpose"
]

REVIEW_KEYWORDS = [
    "review",
    "this review",
    "we review",
    "we summarize",
    "we discuss",
    "scope of this review",
    "this article examines",
    "comprehensive analysis",
    "narrative review",
    "literature review",
    "updated review"
]

QUALITATIVE_KEYWORDS = [
    "focus group",
    "focus groups",
    "interview",
    "interviews",
    "participants",
    "transcribed",
    "qualitative",
    "semi-structured",
    "thematic analysis",
    "nvivo",
    "coded",
    "verbatim"
]

COMPUTATIONAL_KEYWORDS = [
    "pipeline",
    "workflow",
    "alignment",
    "genome assembly",
    "phylogenetic analysis",
    "bioinformatics",
    "computational analysis",
    "dataset",
    "software",
    "model",
    "algorithm"
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
    review_hits = sum(1 for k in REVIEW_KEYWORDS if k in t)
    qualitative_hits = sum(1 for k in QUALITATIVE_KEYWORDS if k in t)
    computational_hits = sum(1 for k in COMPUTATIONAL_KEYWORDS if k in t)

    return min(
        method_hits * 0.025
        + objective_hits * 0.04
        + review_hits * 0.03
        + qualitative_hits * 0.03
        + computational_hits * 0.03,
        0.25
    )


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
    all_methodish = METHOD_KEYWORDS + REVIEW_KEYWORDS + QUALITATIVE_KEYWORDS + COMPUTATIONAL_KEYWORDS
    return any(k in t for k in all_methodish)


def ensure_coverage(retrieved):
    has_objective = any(has_objective_signal(c["text"]) for c in retrieved)
    has_method = any(has_method_signal(c["text"]) for c in retrieved)
    return has_objective, has_method


def build_retrieval_query(label):
    return f"""
Find the most relevant passages in this scientific preprint for evaluating the criterion
'{label}'.

You must prioritize passages that EXPLICITLY describe:

1. The research objective, aim, scope, or hypothesis of the study
2. The methodology, study design, experimental setup, or equivalent analytical approach

PRIORITY ORDER:
- Abstract
- Explicit objective statements in introduction
- Methods / Experimental / Analysis sections
- Study design / Participants / Statistical analysis sections
- For reviews: explicit scope, framing, or synthesis statements

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

    retrieved2, idx2 = select_retrieved_chunks(chunks, adjusted_scores, top_k=fallback_top_k)

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

STRICT BUT REALISTIC RULES:
- Use ONLY the evidence below.
- Do NOT use outside knowledge.
- Do NOT invent information.
- Quotes must be copied VERBATIM from the evidence (max 220 characters each).
- Prefer explicit evidence, but you may interpret standard scientific writing structures when clearly supported by the text.
- If evidence is partial or implicit, you may assign an intermediate score (2).
- Score 3 requires strong support.
- Do NOT penalize a paper for lacking elements that are not expected for its paper type.

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

STEP 1: IDENTIFY PAPER TYPE
Determine the most likely paper type from the evidence only.

Possible paper types:
- original research
- review
- qualitative study
- observational study
- computational/bioinformatics study
- unclear

Use common cues such as:
- "Review", "This review", "we review", "we summarize" -> likely review
- interviews, focus groups, thematic analysis -> likely qualitative study
- cohort, retrospective, registry, population-based -> likely observational study
- pipeline, software, genome assembly, phylogenetic analysis -> likely computational/bioinformatics study
- experiment, assay, randomized design, measurements, instruments -> likely original research

STEP 2: EVALUATE RELATIVE TO PAPER TYPE
Evaluate the clarity and quality of the research objective and the methods
(or equivalent analytical approach) relative to the paper type.

GENERAL GUIDANCE:
- Research objective may appear as: aim, objective, purpose, we investigate, we evaluate, this study explores.
- Methodology may appear as: methods, materials and methods, approach, experimental design, data collection, analysis pipeline.
- The mere presence of an objective and a method mention is NOT automatically enough for score 3.

FOR ORIGINAL RESEARCH:
- Look for a clear objective and a clear description of how the study was conducted.
- Strong methodological support includes procedures, measurements, instruments, datasets, analytical/statistical approach, or participants when relevant.

FOR REVIEW PAPERS:
- Do NOT require an original experimental setup.
- Instead, look for:
  - a clear scope or purpose of the review
  - a clear framing, synthesis approach, or organizing logic
- A review can score:
  - 3 if its scope is explicit and its analytical/synthesis approach is clearly structured
  - 2 if its scope is clear but the review approach is only partly described or fairly generic
  - 1 if both scope and organizing approach are vague or unsupported

FOR QUALITATIVE OR OBSERVATIONAL STUDIES:
- Do NOT require laboratory methods.
- Accept interviews, focus groups, cohorts, registries, sampling strategy, observational design, or analytic approach as methodology.

FOR COMPUTATIONAL / BIOINFORMATICS STUDIES:
- Do NOT require wet-lab methods.
- Accept datasets, workflows, pipelines, software tools, models, or analytic procedures as methodology.

SCORING LOGIC:
- Score 1:
  - Objective or methodology/analytical approach is missing
  - OR cannot be reasonably supported by the evidence.

- Score 2:
  - Objective OR methodology/analytical approach is clearly supported
  - OR both are present but weak, implicit, generic, or only partly connected
  - OR the paper type is clear, but the methodological/analytical detail is only moderate

- Score 3:
  - Clear objective/scope AND clear methodology/analytical approach
  - AND the description is strong enough for that paper type
  - AND objective and method are both directly supported by strong evidence

MANDATORY EVIDENCE COVERAGE:
- You MUST include:
  - objective_evidence: one quote supporting the objective or scope
  - method_evidence: one quote supporting the methodology or analytical approach
- If both objective/scope and methodology/approach are present in the evidence, you are NOT allowed to ignore either one.
- Do NOT use generic background or broad contextual statements as evidence if more direct evidence is available.

IMPORTANT:
- If the paper is a review, do not lower the score just because it lacks original experiments.
- If the paper is qualitative, do not lower the score just because it lacks instruments or assays.
- If the paper is computational, do not lower the score just because it lacks wet-lab methods.
- Prefer direct objective/method statements over general background text.
- If unsure between 2 and 3, choose 2.

Return exactly this JSON schema:
{{
  "criterion": "{c['key']}",
  "paper_type": "original research",
  "score": 1,
  "justification": "brief explanation grounded only in the evidence",
  "objective_evidence": {{
    "chunk_id": 0,
    "pages": [1],
    "quote": "exact short quote copied from the evidence"
  }},
  "method_evidence": {{
    "chunk_id": 1,
    "pages": [3],
    "quote": "exact short quote copied from the evidence"
  }}
}}
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


def _validate_single_evidence(ev, norm_full):
    if not isinstance(ev, dict):
        return None

    quote = str(ev.get("quote", "")).strip()
    if not quote:
        return None

    norm_quote = normalize_for_match(quote)

    if norm_quote and norm_quote in norm_full:
        return ev

    if len(norm_quote) >= 60:
        words = norm_quote.split()
        partial = " ".join(words[: min(12, len(words))])
        if partial and partial in norm_full:
            return ev

    return None


def validate_evidence(parsed, retrieved):
    full_text = "\n\n".join(c["text"] for c in retrieved)
    norm_full = normalize_for_match(full_text)

    obj = parsed.get("objective_evidence")
    met = parsed.get("method_evidence")

    valid_obj = _validate_single_evidence(obj, norm_full)
    valid_met = _validate_single_evidence(met, norm_full)

    parsed["objective_evidence"] = valid_obj
    parsed["method_evidence"] = valid_met

    if not valid_obj or not valid_met:
        parsed["warning"] = "Missing or invalid objective/method evidence."

    return parsed


def downgrade_score_if_needed(parsed):
    score = parsed.get("score")

    obj = parsed.get("objective_evidence")
    met = parsed.get("method_evidence")

    if not isinstance(score, int):
        parsed["score"] = 1
        parsed["warning"] = parsed.get("warning", "") + " Invalid score returned; forced to 1."
        return parsed

    if not obj and not met:
        parsed["score"] = 1
        parsed["justification"] = (
            "Score downgraded to 1 because no valid objective or method evidence was found."
        )
        return parsed

    if score == 3 and (not obj or not met):
        parsed["score"] = 2
        parsed["justification"] = (
            "Score 3 downgraded to 2 because objective and method evidence were not both present."
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

    pages = extract_text_by_page(pdf_path)
    chunks = chunk_pages(pages)
    crit = load_criterion(criteria_path, args.criterion)

    texts = [c["text"] for c in chunks]
    vecs = embed(client, texts, model=args.embedding_model)

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

    prompt = build_prompt(crit, retrieved)

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
            "paper_type": "unclear",
            "score": 1,
            "justification": "Model output could not be parsed as valid JSON.",
            "objective_evidence": None,
            "method_evidence": None,
            "raw_output": raw
        }

    parsed = validate_evidence(parsed, retrieved)
    parsed = downgrade_score_if_needed(parsed)

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
