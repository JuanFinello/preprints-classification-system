#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_preprint_deepseek.py
Pilot script to evaluate a single preprint PDF against the GISAID rubric using
the DeepSeek API, mirroring evaluate_preprint_claude.py's text-only flow so
results are comparable to GPT/Claude.

Purpose right now: check whether DeepSeek-V4 will even score virology/
dual-use preprints without refusing (Claude's API refuses these — see
evaluate_preprint_claude.py history) before considering it for the pipeline.
Not wired into batch_evaluate.py / the CSV pipeline yet.

No PDF file is ever sent to the API — only pre-extracted text (pdfminer, via
evaluate_preprint.extract_pdf), same as the Claude path. Author enrichment
(ROR + Semantic Scholar) is reused from the matching GPT result rather than
recomputed, same reasoning as evaluate_preprint_claude.py.

The DeepSeek API is OpenAI-SDK-compatible (base_url override only), but uses
the plain "max_tokens" param rather than GPT-5.x's "max_completion_tokens" —
kept as its own _llm() rather than reusing evaluate_preprint._llm for that
reason.

Environment:
    DEEPSEEK_API_KEY   required
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml
from openai import OpenAI

from evaluate_preprint import (
    _build_author_prompt,
    _build_content_prompt,
    _build_feedback_prompt,
    _compose_author_credibility_result,
    _compose_research_question_result,
    _doi_from_pdf_path,
    _failed_author_credibility_result,
    _failed_content_result,
    _score_from_reference_count,
    compute_scores,
    extract_pdf,
    fetch_prereview_data,
    validate_quotes,
    verify_references,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash" for the cheaper/faster tier


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return raw


def _llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def evaluate_content_deepseek(eval_text: str, criteria: dict, client: OpenAI, model: str) -> dict:
    prompt = _build_content_prompt(eval_text, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=2000))
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_content_result(f"could not parse response as JSON ({e})", raw)

    result = {
        "research_question_and_methods": _compose_research_question_result(parsed),
        "results_and_conclusion": parsed.get("results_and_conclusion", {}),
        "references": parsed.get("references", {}),
    }

    ref = result["references"]
    ref_count = ref.get("reference_count")
    if isinstance(ref_count, int):
        ref["score"] = _score_from_reference_count(ref_count)
    else:
        ref["score"] = 1
        ref["justification"] = (ref.get("justification", "") + " [no reference_count reported]").strip()
    return result


def evaluate_author_credibility_deepseek(author_data: dict, criteria: dict, client: OpenAI, model: str) -> dict:
    prompt = _build_author_prompt(author_data, criteria)
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=2000))
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_author_credibility_result(f"could not parse response as JSON ({e})", raw)

    return _compose_author_credibility_result(parsed)


def evaluate_feedback_deepseek(doi: str, criteria: dict, client: OpenAI, model: str) -> dict:
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
    raw = _strip_json_fences(_llm(client, model, prompt, max_tokens=1500))
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


def evaluate_preprint_deepseek(pdf_path: Path, criteria_path: Path, model: str,
                                client: OpenAI, author_data: dict) -> dict:
    """Full evaluation pipeline for a single PDF, DeepSeek-scored.

    author_data comes from the matching GPT result's "author_enrichment" —
    not recomputed here, see module docstring. Only extracted text is sent
    to the API — no PDF file is ever uploaded or attached to a request.
    """
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))["criteria"]

    pdf_data = extract_pdf(pdf_path)

    content_results = evaluate_content_deepseek(pdf_data["eval_text"], criteria, client, model)
    author_results = evaluate_author_credibility_deepseek(author_data, criteria, client, model)

    criterion_results = {**author_results, **content_results}
    criterion_results["feedback"] = evaluate_feedback_deepseek(_doi_from_pdf_path(pdf_path), criteria, client, model)

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


# ─── CLI (pilot: run against one PDF at a time, reusing an existing GPT result
# for author_data) ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pilot: evaluate one preprint PDF with DeepSeek-V4, checking it scores "
                    "(rather than refuses) dual-use/virology content before wider adoption."
    )
    ap.add_argument("--pdf", required=True, help="Path to the preprint PDF")
    ap.add_argument("--gpt-result", required=True,
                    help="Path to the matching GPT result JSON (results_pipeline/*.json), "
                         "used only for its author_enrichment data")
    ap.add_argument("--criteria", default="criteria.yaml", help="Path to criteria.yaml")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model to use")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    gpt_result_path = Path(args.gpt_result)
    criteria_path = Path(args.criteria)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not gpt_result_path.exists():
        raise FileNotFoundError(f"GPT result not found: {gpt_result_path}")
    if not criteria_path.exists():
        raise FileNotFoundError(f"criteria.yaml not found: {criteria_path}")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    gpt_result = json.loads(gpt_result_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    result = evaluate_preprint_deepseek(
        pdf_path, criteria_path, args.model, client,
        author_data=gpt_result["author_enrichment"],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    scoring = result["scoring"]
    print(f"✓ {pdf_path.name}")
    print(f"  Scores: {scoring['per_criterion']}")
    print(f"  Average: {scoring['average']} / 3.0  |  Recommendation: {scoring['recommendation']}")
    invalid = scoring.get("quotes_invalid", [])
    if invalid:
        print(f"  ⚠ Quotes not verified: {', '.join(invalid)}")


if __name__ == "__main__":
    main()
