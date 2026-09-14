#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_preprint_claude.py
Evaluates a single preprint PDF against the GISAID rubric using the Claude API,
mirroring evaluate_preprint.py's GPT flow so the two are comparable.

No PDF file is ever sent to the API — only pre-extracted text (pdfminer,
via evaluate_preprint.extract_pdf), the same content GPT sees. Author
enrichment (ROR + Semantic Scholar) is NOT recomputed here either: it's
reused from the matching GPT result, since it's already text-extraction-based
and re-running it would just add cost without changing the signal.

Environment:
    ANTHROPIC_API_KEY   required
"""

import json
import re
from pathlib import Path

import anthropic

from evaluate_preprint import (
    _build_author_prompt,
    _build_content_prompt,
    _build_feedback_prompt,
    _compose_author_credibility_result,
    _compose_research_question_result,
    _doi_from_pdf_path,
    _failed_author_credibility_result,
    _failed_content_result,
    _apply_reference_score,
    compute_scores,
    extract_pdf,
    fetch_prereview_data,
    validate_quotes,
    verify_references,
)

DEFAULT_MODEL = "claude-sonnet-5"


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return raw


def _llm(client: anthropic.Anthropic, model: str, prompt: str, max_tokens: int = 16000) -> str:
    """Una llamada a la API. Devuelve el texto del bloque `text`.

    max_tokens tiene que cubrir thinking + respuesta: en claude-sonnet-5 el
    thinking adaptativo está PRENDIDO por defecto (omitir `thinking` no lo
    apaga), así que un presupuesto chico se consume pensando y la respuesta
    llega sin ningún bloque `text`. Con max_tokens=2000 eso pasaba SIEMPRE:
    stop_reason='max_tokens', output_tokens=2000, todos thinking_tokens, y el
    único bloque devuelto era ('thinking', 0 chars). El texto vacío hacía
    fallar json.loads() y cada criterio quedaba con score=1 "Parse error"
    (bug real, detectado 2026-08-25 — todos los scores de Claude eran basura).
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    if not text:
        # No degradar en silencio a score=1: sin texto no hay evaluación.
        raise RuntimeError(
            f"Respuesta sin bloque de texto (stop_reason={resp.stop_reason}, "
            f"output_tokens={resp.usage.output_tokens}, max_tokens={max_tokens}). "
            "Si stop_reason es 'max_tokens', subí max_tokens: el presupuesto se "
            "agotó en thinking antes de escribir la respuesta."
        )
    return text


def evaluate_content_claude(eval_text: str, criteria: dict, client: anthropic.Anthropic, model: str) -> dict:
    prompt = _build_content_prompt(eval_text, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=16000))
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_content_result(f"could not parse response as JSON ({e})", raw)

    result = {
        "research_question_and_methods": _compose_research_question_result(parsed),
        "results_and_conclusion": parsed.get("results_and_conclusion", {}),
        "references": parsed.get("references", {}),
    }

    _apply_reference_score(result["references"])
    return result


def evaluate_author_credibility_claude(author_data: dict, criteria: dict, client: anthropic.Anthropic, model: str) -> dict:
    prompt = _build_author_prompt(author_data, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=16000))
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_author_credibility_result(f"could not parse response as JSON ({e})", raw)

    return _compose_author_credibility_result(parsed)


def evaluate_feedback_claude(doi: str, criteria: dict, client: anthropic.Anthropic, model: str) -> dict:
    """Same approach as evaluate_feedback (GPT side): score from real
    PREreview text when any exists for this DOI; stays manual (score: None)
    otherwise — see evaluate_preprint.evaluate_feedback for the reasoning."""
    prereview_data = fetch_prereview_data(doi)
    if prereview_data["n_reviews"] == 0:
        return {
            "score": None,
            "justification": "Not evaluated automatically. Requires manual check of preprint server.",
            "prereview_data": prereview_data,
        }

    prompt = _build_feedback_prompt(criteria, [rv["text"] for rv in prereview_data["reviews"]])
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=16000))
    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "score": None,
            "justification": "Parse error scoring PREreview evidence — requires manual check.",
            "prereview_data": prereview_data,
        }

    score = parsed.get("score")
    return {
        "score": score if isinstance(score, int) else None,
        "justification": parsed.get("justification", ""),
        "prereview_data": prereview_data,
    }


def evaluate_preprint_claude(pdf_path: Path, criteria_path: Path, model: str,
                              client: anthropic.Anthropic, author_data: dict) -> dict:
    """Full evaluation pipeline for a single PDF, Claude-scored.

    author_data comes from the matching GPT result's "author_enrichment" —
    not recomputed here, see module docstring. Only extracted text is sent
    to the API — no PDF file is ever uploaded or attached to a request.
    """
    import yaml
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))["criteria"]

    pdf_data = extract_pdf(pdf_path)

    content_results = evaluate_content_claude(pdf_data["eval_text"], criteria, client, model)
    author_results = evaluate_author_credibility_claude(author_data, criteria, client, model)

    criterion_results = {**author_results, **content_results}
    criterion_results["feedback"] = evaluate_feedback_claude(_doi_from_pdf_path(pdf_path), criteria, client, model)

    criterion_results = validate_quotes(criterion_results, pdf_data["full_text"])
    criterion_results["references"]["citation_verification"] = verify_references(pdf_data["full_text"])
    scoring = compute_scores(criterion_results)

    checked = ["research_question_and_methods", "results_and_conclusion", "references"]
    invalid = [k for k in checked if not criterion_results.get(k, {}).get("quote_valid", True)]
    scoring["quotes_invalid"] = invalid

    return {
        "source_pdf": pdf_path.name,
        "model": model,
        "n_pages": pdf_data["n_pages"],
        "author_enrichment": author_data,
        "criteria": criterion_results,
        "scoring": scoring,
    }
