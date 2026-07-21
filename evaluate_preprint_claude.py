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
    _score_from_reference_count,
    compute_scores,
    extract_pdf,
    validate_quotes,
)

DEFAULT_MODEL = "claude-sonnet-5"


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return raw


def _llm(client: anthropic.Anthropic, model: str, prompt: str, max_tokens: int = 1024) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def evaluate_content_claude(eval_text: str, criteria: dict, client: anthropic.Anthropic, model: str) -> dict:
    prompt = _build_content_prompt(eval_text, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=1500))
    try:
        result = json.loads(raw)
    except Exception:
        return {
            "research_question_and_methods": {"score": 1, "justification": "Parse error", "quote": ""},
            "results_and_conclusion": {"score": 1, "justification": "Parse error", "quote": ""},
            "references": {"score": 1, "justification": "Parse error", "quote": ""},
        }

    ref = result.get("references", {})
    ref_count = ref.get("reference_count")
    if isinstance(ref_count, int):
        ref["score"] = _score_from_reference_count(ref_count)
    else:
        ref["score"] = 1
        ref["justification"] = (ref.get("justification", "") + " [no reference_count reported]").strip()
    return result


def evaluate_author_credibility_claude(author_data: dict, criteria: dict, client: anthropic.Anthropic, model: str) -> dict:
    prompt = _build_author_prompt(author_data, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=512))
    try:
        return json.loads(raw)
    except Exception:
        return {"author_credibility": {"score": 1, "justification": "Parse error", "flags": []}}


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
    criterion_results["feedback"] = {
        "score": None,
        "justification": "Not evaluated automatically. Requires manual check of preprint server.",
    }

    criterion_results = validate_quotes(criterion_results, pdf_data["full_text"])
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
